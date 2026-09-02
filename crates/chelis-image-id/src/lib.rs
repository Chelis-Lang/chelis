//! Identify the object image that the *running chelis code* is loaded from,
//! and derive a build-identity discriminator for it.
//!
//! This exists for chelis#1156: compiled-context caches must be keyed on the
//! compiler BUILD, not on `COMPILER_VERSION`, which names a release. Two
//! binaries built from different commits share one `workspace.package.version`
//! until the next bump, so a version-keyed cache lets them read each other's
//! entries and check programs under the other build's type semantics.
//!
//! Two things make that harder than reading `current_exe()`:
//!
//! 1. **The running code is not always the process image.** `chelis-python` is
//!    built `crate-type = ["cdylib", "rlib"]` and loaded into a Python
//!    interpreter, so `std::env::current_exe()` there names `python3`, not the
//!    extension module. Keying on it would both merge two different
//!    `chelis_python` builds under one interpreter (the chelis#1156 defect,
//!    unfixed on that surface) and split one build across two interpreters or
//!    virtualenvs for no semantic reason. [`running_image_path`] asks the
//!    dynamic loader which object a symbol *in this crate* came from, so a
//!    static binary reports itself and a `cdylib` reports the shared object.
//!
//! 2. **Hashing the whole image is too expensive to do per process.** A debug
//!    `chelis` is roughly 138MB, and SHA-256 over it costs about 250ms on an
//!    M-series laptop, which is more than an entire warm-cache invocation. The
//!    linker has already computed a content id for exactly this purpose, so
//!    [`identify`] reads that instead and falls back to the digest only when
//!    the image carries none.
//!
//! The crate is deliberately separate from `chelis-compiler-api`, which sets
//! `#![forbid(unsafe_code)]`. `dladdr` is FFI and cannot be called from there,
//! and `forbid` cannot be relaxed locally by design. Keeping the one unsafe
//! call here preserves that guarantee for the compiler crate. `chelis-reef`
//! also cannot depend on `chelis-compiler-api` (it sits below it), so a
//! standalone crate is the only home both consumers can share.

use std::path::{Path, PathBuf};

/// A build-identity discriminator for one object image.
///
/// Both variants answer the same question ("which build is this?"), and both
/// are derived from the image's content rather than from its filesystem
/// metadata, so neither is affected by mtime normalization (Nix canonicalizes
/// every store file to mtime 1; `cp -p`, `rsync -t`, and tar/OCI layers
/// preserve timestamps across genuinely different builds).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageId {
    /// The linker's own content id: a Mach-O `LC_UUID` or an ELF
    /// `NT_GNU_BUILD_ID` note.
    ///
    /// # What this covers, exactly
    ///
    /// Both are hashes the linker computes over the **mapped image** - the code
    /// and data the process actually executes and reads. Neither covers the
    /// whole file: the symbol table, the debug map, and the code signature all
    /// sit outside them.
    ///
    /// That is the right granularity for keying a cache of type-checking
    /// results, and it is deliberately NOT the same as "these two files are
    /// byte-identical". Measured on this workspace, both directions:
    ///
    /// - **It tracks every semantic change.** Editing one live string constant
    ///   in `chelis-compiler-api` and relinking changed the `LC_UUID`
    ///   (`793D66F4...` to `4E4BD4F1...`). Semantics are determined by code and
    ///   data, so a change that leaves the mapped image identical cannot change
    ///   behaviour.
    /// - **It ignores rebuild churn that cannot affect behaviour.** Adding an
    ///   unreferenced `pub fn` to a dependency crate produced a binary of
    ///   identical length whose SHA-256 differed, because the added symbol was
    ///   dead-stripped and the only surviving difference was cargo's rlib
    ///   metadata hash inside the debug-map paths (1347 occurrences of one
    ///   6-byte token). The `LC_UUID` was unchanged, correctly: those two
    ///   binaries compute the same answers, so sharing a cache entry between
    ///   them is right, and a whole-file digest would have forced a pointless
    ///   recompile.
    ///
    /// It is also stable across relinks of unchanged input: rebuilding an
    /// earlier source reproduced its original `LC_UUID` exactly.
    ///
    /// # Callers must not assume byte equality
    ///
    /// Because this deliberately excludes non-executed regions, two files with
    /// different bytes can share an id. Do not use [`ImageId`] as a file
    /// checksum; it answers "would these two images behave the same?", which is
    /// the question a semantics cache needs and a checksum is only a proxy for.
    ///
    /// It is read from the image header, so obtaining it costs microseconds
    /// regardless of image size.
    LinkerBuildId(Vec<u8>),
    /// SHA-256 over the entire image, used when the image carries no linker id.
    ///
    /// ELF `NT_GNU_BUILD_ID` depends on the link being driven with
    /// `--build-id`, which most toolchains default to but none guarantee, so
    /// this fallback carries real weight on Linux in a way it does not on macOS.
    ContentDigest([u8; 32]),
}

impl ImageId {
    /// A short, stable tag naming how this id was derived.
    ///
    /// Callers that build a cache key from an [`ImageId`] must include this, so
    /// that a linker id and a digest can never be confused for one another.
    pub fn scheme(&self) -> &'static str {
        match self {
            ImageId::LinkerBuildId(_) => "bid",
            ImageId::ContentDigest(_) => "sha256",
        }
    }

    /// The id's bytes, lowercase hex.
    pub fn hex(&self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let bytes: &[u8] = match self {
            ImageId::LinkerBuildId(b) => b,
            ImageId::ContentDigest(d) => d,
        };
        let mut out = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0xf) as usize] as char);
        }
        out
    }
}

/// One identified object image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunningImage {
    /// Where the image was read from. Recorded for diagnostics only: it is
    /// deliberately NOT part of any cache key, so relocating an identical
    /// binary keeps its cache rather than orphaning it.
    pub path: PathBuf,
    /// The image's byte length.
    pub len: u64,
    /// The build-identity discriminator.
    pub id: ImageId,
}

/// Return the exact per-build identity shared by every compiler cache layer.
///
/// All workspace crates have one release version, so the running image's
/// content identity plus this crate's package version is the compiler build
/// identity. Keeping the formatter here prevents lower-level caches such as
/// `chelis-reef` from approximating `chelis-compiler-api`'s fingerprint or
/// creating a dependency cycle to obtain it.
pub fn build_fingerprint() -> &'static str {
    static FINGERPRINT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    FINGERPRINT.get_or_init(|| {
        let image = running_image();
        if image.is_none() {
            // The degraded identity is safe because it forces misses, but it
            // is expensive enough to diagnose rather than hiding the cause.
            use std::io::Write as _;
            let _ = writeln!(
                std::io::stderr(),
                "chelis: warning: could not identify the running compiler image, so \
                 compiled-context caches cannot be shared between invocations and \
                 every run will rebuild them (chelis#1156)."
            );
        }
        fingerprint_string(image.as_ref())
    })
}

/// Format one image-identification result as the workspace build identity.
///
/// This is public for cross-crate negative controls; production callers use
/// [`build_fingerprint`], which pins one result for the process.
#[doc(hidden)]
pub fn fingerprint_string(image: Option<&RunningImage>) -> String {
    const COMPILER_VERSION: &str = env!("CARGO_PKG_VERSION");
    match image {
        Some(image) => format!(
            "{COMPILER_VERSION}+{scheme}.{id}.{len:x}",
            scheme = image.id.scheme(),
            id = image.id.hex(),
            len = image.len
        ),
        None => {
            // A per-process random discriminator fails toward misses instead
            // of ever sharing an unidentified build's semantic cache.
            use std::hash::{BuildHasher, Hasher};
            let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
            hasher.write_u32(std::process::id());
            let nonce = hasher.finish();
            format!(
                "{COMPILER_VERSION}+degraded.{pid:x}.{nonce:x}",
                pid = std::process::id()
            )
        }
    }
}

/// The path of the object image containing this crate's code.
///
/// Asks the dynamic loader (`dladdr`) which object a symbol defined *here* was
/// loaded from. For an ordinary statically linked binary that is the executable
/// itself; for a `cdylib` loaded into another process it is the shared object,
/// which is the whole point (see the module docs).
///
/// Falls back to [`std::env::current_exe`] when `dladdr` is unavailable or
/// yields nothing usable.
pub fn running_image_path() -> Option<PathBuf> {
    dladdr_self().or_else(|| std::env::current_exe().ok())
}

/// Every candidate path for the running image, best first.
///
/// [`running_image`] walks these in order so that a `dladdr` result which
/// cannot actually be opened (a deleted or replaced shared object) still falls
/// through to the executable rather than degrading the fingerprint.
fn running_image_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = dladdr_self() {
        out.push(p);
    }
    if let Ok(exe) = std::env::current_exe()
        && !out.contains(&exe)
    {
        out.push(exe);
    }
    out
}

#[cfg(unix)]
fn dladdr_self() -> Option<PathBuf> {
    use std::ffi::{CStr, OsStr};
    use std::os::unix::ffi::OsStrExt;

    // A function defined in THIS crate. Whatever object the loader says this
    // address came from is the object this code is running inside.
    let probe = dladdr_self as *const () as *const libc::c_void;
    let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();

    // SAFETY: `probe` is a valid, non-null code address (the address of a
    // function in this crate) and `info` is a valid, properly aligned,
    // writable `Dl_info` that `dladdr` fully initializes on success. We only
    // read `info` when `dladdr` returned nonzero, which is its documented
    // success condition. `dli_fname` then points to a NUL-terminated string
    // owned by the loader and valid for the lifetime of the loaded object,
    // which outlives the borrow taken here because we copy out immediately.
    let filled = unsafe { libc::dladdr(probe, info.as_mut_ptr()) };
    if filled == 0 {
        return None;
    }
    // SAFETY: `dladdr` returned success, so `info` is initialized.
    let info = unsafe { info.assume_init() };
    if info.dli_fname.is_null() {
        return None;
    }
    // SAFETY: see above; `dli_fname` is a NUL-terminated C string from the
    // loader, and the bytes are copied into an owned `PathBuf` before return.
    let name = unsafe { CStr::from_ptr(info.dli_fname) };
    let bytes = name.to_bytes();
    if bytes.is_empty() {
        return None;
    }
    Some(PathBuf::from(OsStr::from_bytes(bytes)))
}

#[cfg(not(unix))]
fn dladdr_self() -> Option<PathBuf> {
    // No `dladdr` equivalent is wired up yet. `current_exe` is correct for a
    // statically linked binary, which is every non-unix chelis build today.
    None
}

/// Identify one object image at `path`: its byte length and its build id.
///
/// Prefers the linker's content id and falls back to a SHA-256 of the whole
/// file. Returns `None` only when the file cannot be read at all.
pub fn identify(path: &Path) -> Option<(u64, ImageId)> {
    let file = std::fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    if let Some(id) = linker_build_id(&file, len) {
        return Some((len, ImageId::LinkerBuildId(id)));
    }
    let digest = sha256_file(&file)?;
    Some((len, ImageId::ContentDigest(digest)))
}

/// The running image, fully identified.
pub fn running_image() -> Option<RunningImage> {
    for path in running_image_candidates() {
        if let Some((len, id)) = identify(&path) {
            return Some(RunningImage { path, len, id });
        }
    }
    None
}

/// The linker's content id for an already-open image, if it carries one.
///
/// Deliberately avoids `object::File::parse`, which eagerly builds the symbol
/// table: on a 138MB debug `chelis` that costs about 50ms, most of it work this
/// crate throws away. Dispatching on the magic bytes and reading only the
/// header, the Mach-O load commands, or the ELF program headers and note
/// segment brings the same answer down to microseconds.
///
/// Reads through `object`'s `ReadCache`, which pulls only the byte ranges the
/// parse actually touches, so this stays a header read rather than a whole-file
/// read. Deliberately not `mmap`: a mapping of a file that is truncated
/// underneath raises `SIGBUS`, and killing the compiler to save a few
/// microseconds is a bad trade.
///
/// `len` is the image's real byte length, and every format arm checks that the
/// structures the header claims actually fit inside it before trusting the id.
/// A truncated image must fall through to the digest, and that has to hold on
/// each format for its own reason rather than by accident: a truncated Mach-O
/// usually fails earlier anyway, because `sizeofcmds` runs past the end of the
/// file and the load-command read errors out, but an ELF keeps its program
/// headers and its `PT_NOTE` in the first few hundred bytes, so a truncation
/// that destroys almost the whole image still yields a perfectly readable
/// build-id note. Without this check the same truncation is rejected on one
/// platform and accepted on the other.
///
/// Any parse failure returns `None` and routes the caller to the digest, so a
/// malformed or unfamiliar image degrades to the slow-but-total path instead of
/// producing a wrong id.
fn linker_build_id(file: &std::fs::File, len: u64) -> Option<Vec<u8>> {
    use object::{Endianness, FileKind};

    let data = object::ReadCache::new(file);
    match FileKind::parse(&data).ok()? {
        FileKind::MachO32 => macho_uuid::<object::macho::MachHeader32<Endianness>, _>(&data, len),
        FileKind::MachO64 => macho_uuid::<object::macho::MachHeader64<Endianness>, _>(&data, len),
        FileKind::Elf32 => elf_build_id::<object::elf::FileHeader32<Endianness>, _>(&data, len),
        FileKind::Elf64 => elf_build_id::<object::elf::FileHeader64<Endianness>, _>(&data, len),
        // Fat/universal Mach-O, PE, and everything else fall through to the
        // digest. A fat binary has no single image id to report.
        _ => None,
    }
}

/// The `LC_UUID` load command's payload, if the Mach-O image carries one and
/// its segments all fit within `len`.
///
/// `LC_UUID` is a hash the linker computes over the mapped image, so it is a
/// content id and not merely a build label. See [`ImageId::LinkerBuildId`] for
/// exactly what that does and does not cover.
fn macho_uuid<'data, M, R>(data: R, len: u64) -> Option<Vec<u8>>
where
    M: object::read::macho::MachHeader,
    R: object::ReadRef<'data>,
{
    use object::read::macho::Segment as _;

    let header = M::parse(data, 0).ok()?;
    if !header.is_supported() {
        return None;
    }
    let endian = header.endian().ok()?;

    let mut uuid: Option<Vec<u8>> = None;
    let mut claimed_end: u64 = 0;
    let mut commands = header.load_commands(endian, data, 0).ok()?;
    while let Ok(Some(command)) = commands.next() {
        if let Ok(Some((segment, _))) = M::Segment::from_command(command) {
            let (offset, size) = segment.file_range(endian);
            claimed_end = claimed_end.max(offset.saturating_add(size));
        } else if let Ok(Some(command_uuid)) = command.uuid() {
            uuid.get_or_insert_with(|| command_uuid.uuid.to_vec());
        }
    }
    if claimed_end > len {
        return None;
    }
    uuid
}

/// The `NT_GNU_BUILD_ID` note's descriptor, if the ELF image carries one and
/// the structures its header claims all fit within `len`.
///
/// Read from the program headers rather than the section headers: section
/// headers live at the end of the file and are stripped from some shipped
/// images, while the note segment is mapped and near the front.
///
/// Emitting this note requires the link to be driven with `--build-id`. Most
/// toolchains default to it, but none guarantee it, so a `None` here is an
/// ordinary outcome and routes the caller to the digest.
fn elf_build_id<'data, E, R>(data: R, len: u64) -> Option<Vec<u8>>
where
    E: object::read::elf::FileHeader,
    R: object::ReadRef<'data>,
{
    use object::read::elf::ProgramHeader as _;

    let header = E::parse(data).ok()?;
    if !header.is_supported() {
        return None;
    }
    let endian = header.endian().ok()?;
    let program_headers = header.program_headers(endian, data).ok()?;

    // Everything the header says is in this file must actually be in it. A
    // `PT_NOTE` sitting in the first page survives a truncation that removes
    // every `PT_LOAD` byte, so the note alone proves nothing about the image.
    let mut claimed_end: u64 = 0;
    for segment in program_headers {
        let (offset, size) = segment.file_range(endian);
        claimed_end = claimed_end.max(offset.saturating_add(size));
    }
    // Section headers are optional (a fully stripped image reports none), but
    // when the header claims them they are part of the file too.
    let section_offset: u64 = header.e_shoff(endian).into();
    if section_offset != 0 {
        let table =
            u64::from(header.e_shnum(endian)).saturating_mul(u64::from(header.e_shentsize(endian)));
        claimed_end = claimed_end.max(section_offset.saturating_add(table));
    }
    if claimed_end > len {
        return None;
    }

    for segment in program_headers {
        let Ok(Some(mut notes)) = segment.notes(endian, data) else {
            continue;
        };
        while let Ok(Some(note)) = notes.next() {
            if note.name() == object::elf::ELF_NOTE_GNU
                && note.n_type(endian) == object::elf::NT_GNU_BUILD_ID
            {
                let desc = note.desc();
                if !desc.is_empty() {
                    return Some(desc.to_vec());
                }
            }
        }
    }
    None
}

/// SHA-256 over a whole file, or `None` if it cannot be read.
///
/// Chunked rather than `io::copy` into the hasher so this does not depend on
/// `sha2`'s `std` feature surviving feature unification.
fn sha256_file(file: &std::fs::File) -> Option<[u8; 32]> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Seek, SeekFrom};

    let mut file = file.try_clone().ok()?;
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => hasher.update(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return None,
        }
    }
    Some(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The running test binary must be locatable and identifiable.
    #[test]
    fn the_running_image_is_identifiable() {
        let image = running_image().expect("the test binary must be identifiable");
        assert!(image.len > 0, "an image cannot be zero bytes");
        assert!(!image.id.hex().is_empty(), "an id must have bytes");
    }

    /// On every platform chelis builds for today, the running test binary is a
    /// Mach-O or an ELF that carries a linker id. If this starts failing the
    /// fallback is still correct, but the fast path has silently stopped
    /// applying and every process is paying a full hash again, so this is
    /// worth knowing about.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn the_running_image_carries_a_linker_build_id() {
        let image = running_image().expect("identifiable");
        assert!(
            matches!(image.id, ImageId::LinkerBuildId(_)),
            "expected a linker build id on this platform, got {:?}; the whole-file \
             digest fallback is correct but costs a full hash per process",
            image.id.scheme()
        );
    }

    /// `dladdr` must resolve a symbol in this crate. For the test binary that
    /// is the test binary itself, which is also what `current_exe` reports;
    /// the case where they differ is a `cdylib` and is covered by the
    /// `chelis-python` integration test.
    #[cfg(unix)]
    #[test]
    fn dladdr_resolves_this_crates_own_object() {
        let via_loader = dladdr_self().expect("dladdr must resolve a local symbol");
        assert!(
            via_loader.exists(),
            "dladdr reported {via_loader:?}, which does not exist"
        );
    }

    /// The id must be a function of CONTENT, not of path or timestamp. This is
    /// the chelis#1156 property: Nix canonicalizes every store file's mtime to
    /// 1 second past the epoch, so an id derived from metadata collapses.
    #[test]
    fn the_id_ignores_path_and_mtime() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        let exe = running_image_path().expect("locatable");
        std::fs::copy(&exe, &a).expect("copy a");
        std::fs::copy(&exe, &b).expect("copy b");
        set_mtime(&a, 1);
        set_mtime(&b, 1_755_000_000);

        let (len_a, id_a) = identify(&a).expect("identify a");
        let (len_b, id_b) = identify(&b).expect("identify b");
        assert_eq!(len_a, len_b);
        assert_eq!(id_a, id_b, "identical content must identify identically");
    }

    /// The converse: different content must not share an id even when byte
    /// length and mtime match exactly. That pair is the realized chelis#1156
    /// collision shape, not a hypothetical one.
    #[test]
    fn different_content_at_one_length_and_mtime_differs() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        std::fs::write(&a, b"build one, semantics A").expect("write a");
        std::fs::write(&b, b"build two, semantics B").expect("write b");
        assert_eq!(
            std::fs::metadata(&a).expect("meta a").len(),
            std::fs::metadata(&b).expect("meta b").len(),
            "fixture must hold length constant, or it tests nothing"
        );
        set_mtime(&a, 1);
        set_mtime(&b, 1);

        // Neither is a real object file, so both take the digest fallback.
        let (_, id_a) = identify(&a).expect("identify a");
        let (_, id_b) = identify(&b).expect("identify b");
        assert!(matches!(id_a, ImageId::ContentDigest(_)));
        assert_ne!(id_a, id_b);
    }

    /// A synthetic, minimal ELF64 carrying a `NT_GNU_BUILD_ID` note.
    ///
    /// The ELF arm would otherwise be exercised only on Linux CI, and that gap
    /// is exactly how PR #1161 shipped a truncation check that held on Mach-O
    /// and not on ELF. Building the bytes here tests it on every platform.
    ///
    /// Layout: 64-byte header, one 56-byte `PT_NOTE` program header at offset
    /// 64, and the note itself at offset 120. `e_shoff` is 0, i.e. fully
    /// stripped of section headers, which is also the shape that forces the
    /// build id to be found through the program headers.
    fn synthetic_elf64_with_build_id(build_id: &[u8]) -> Vec<u8> {
        synthetic_elf64(build_id, None)
    }

    /// As above, but optionally also declaring a `PT_LOAD` of `load_filesz`
    /// bytes. A real binary always has one, and it is what makes a truncation
    /// detectable when the note itself survives.
    fn synthetic_elf64(build_id: &[u8], load_filesz: Option<u64>) -> Vec<u8> {
        let phnum: u16 = if load_filesz.is_some() { 2 } else { 1 };
        let note_offset: u64 = 64 + 56 * u64::from(phnum);
        let mut note = Vec::new();
        note.extend_from_slice(&4u32.to_le_bytes()); // n_namesz: "GNU\0"
        note.extend_from_slice(&(build_id.len() as u32).to_le_bytes()); // n_descsz
        note.extend_from_slice(&3u32.to_le_bytes()); // n_type: NT_GNU_BUILD_ID
        note.extend_from_slice(b"GNU\0");
        note.extend_from_slice(build_id);
        while note.len() % 4 != 0 {
            note.push(0);
        }
        let note_len = note.len() as u64;

        let mut out = Vec::new();
        // --- ELF64 header ---
        out.extend_from_slice(&[0x7f, b'E', b'L', b'F']);
        out.push(2); // ELFCLASS64
        out.push(1); // ELFDATA2LSB
        out.push(1); // EV_CURRENT
        out.push(0); // ELFOSABI_SYSV
        out.push(0); // ABI version
        out.extend_from_slice(&[0; 7]); // padding
        out.extend_from_slice(&3u16.to_le_bytes()); // e_type: ET_DYN
        out.extend_from_slice(&62u16.to_le_bytes()); // e_machine: EM_X86_64
        out.extend_from_slice(&1u32.to_le_bytes()); // e_version
        out.extend_from_slice(&0u64.to_le_bytes()); // e_entry
        out.extend_from_slice(&64u64.to_le_bytes()); // e_phoff
        out.extend_from_slice(&0u64.to_le_bytes()); // e_shoff: stripped
        out.extend_from_slice(&0u32.to_le_bytes()); // e_flags
        out.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
        out.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
        out.extend_from_slice(&phnum.to_le_bytes()); // e_phnum
        out.extend_from_slice(&0u16.to_le_bytes()); // e_shentsize
        out.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
        out.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx
        assert_eq!(out.len(), 64, "ELF64 header must be 64 bytes");

        // --- optional PT_LOAD, mirroring a real binary ---
        if let Some(filesz) = load_filesz {
            out.extend_from_slice(&1u32.to_le_bytes()); // p_type: PT_LOAD
            out.extend_from_slice(&5u32.to_le_bytes()); // p_flags: PF_R | PF_X
            out.extend_from_slice(&0u64.to_le_bytes()); // p_offset
            out.extend_from_slice(&0u64.to_le_bytes()); // p_vaddr
            out.extend_from_slice(&0u64.to_le_bytes()); // p_paddr
            out.extend_from_slice(&filesz.to_le_bytes()); // p_filesz
            out.extend_from_slice(&filesz.to_le_bytes()); // p_memsz
            out.extend_from_slice(&4096u64.to_le_bytes()); // p_align
        }

        // --- one PT_NOTE program header ---
        out.extend_from_slice(&4u32.to_le_bytes()); // p_type: PT_NOTE
        out.extend_from_slice(&4u32.to_le_bytes()); // p_flags: PF_R
        out.extend_from_slice(&note_offset.to_le_bytes()); // p_offset
        out.extend_from_slice(&0u64.to_le_bytes()); // p_vaddr
        out.extend_from_slice(&0u64.to_le_bytes()); // p_paddr
        out.extend_from_slice(&note_len.to_le_bytes()); // p_filesz
        out.extend_from_slice(&note_len.to_le_bytes()); // p_memsz
        out.extend_from_slice(&4u64.to_le_bytes()); // p_align
        assert_eq!(
            out.len() as u64,
            note_offset,
            "program headers must end exactly at the note offset"
        );

        out.extend_from_slice(&note);
        out
    }

    /// The ELF arm must actually extract `NT_GNU_BUILD_ID`.
    #[test]
    fn an_elf_image_yields_its_gnu_build_id() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let p = dir.path().join("synthetic.so");
        let want: Vec<u8> = (0u8..16).collect();
        std::fs::write(&p, synthetic_elf64_with_build_id(&want)).expect("write");

        let (_, id) = identify(&p).expect("identify");
        assert_eq!(
            id.scheme(),
            "bid",
            "the ELF note must be found, not hashed around"
        );
        assert_eq!(id, ImageId::LinkerBuildId(want));
    }

    /// Two ELF images differing only in their build-id note must not share an
    /// id, the ELF-side counterpart of the Mach-O `LC_UUID` property.
    #[test]
    fn elf_images_with_different_build_ids_differ() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let (a, b) = (dir.path().join("a.so"), dir.path().join("b.so"));
        std::fs::write(&a, synthetic_elf64_with_build_id(&[0xaa; 16])).expect("write a");
        std::fs::write(&b, synthetic_elf64_with_build_id(&[0xbb; 16])).expect("write b");
        assert_eq!(
            std::fs::metadata(&a).expect("meta a").len(),
            std::fs::metadata(&b).expect("meta b").len(),
            "fixture must hold length constant, or it tests nothing"
        );
        assert_ne!(identify(&a).expect("a").1, identify(&b).expect("b").1);
    }

    /// A truncated ELF must fall back to the digest even though its note is
    /// intact.
    ///
    /// This is the case that made PR #1161 red on Linux and green on macOS. An
    /// ELF keeps its program headers and `PT_NOTE` in the first few hundred
    /// bytes, so a truncation that destroys the entire mapped image still
    /// leaves a perfectly readable build-id note behind; a Mach-O of the same
    /// shape fails earlier because `sizeofcmds` runs off the end. Only the
    /// explicit extent check makes the two agree.
    #[test]
    fn a_truncated_elf_falls_back_to_the_digest_despite_an_intact_note() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let full = synthetic_elf64_with_build_id(&[0xcd; 16]);
        let p = dir.path().join("truncated.so");
        // Cut inside the note: the header and program header still parse, and
        // the note's own offset still looks valid from the header alone.
        std::fs::write(&p, &full[..full.len() - 8]).expect("write");

        let (_, id) = identify(&p).expect("identify");
        assert_eq!(
            id.scheme(),
            "sha256",
            "a truncated ELF must not report a linker id just because its \
             note happens to survive in the first page"
        );
    }

    /// The exact shape that made PR #1161 red on Linux and green on macOS.
    ///
    /// `a_truncated_object_image_falls_back_to_the_digest` truncates the real
    /// test binary to 2048 bytes. On Mach-O that already fails, because
    /// `sizeofcmds` runs past the end of the file. On ELF it does not: the
    /// header, the program headers, and the whole `PT_NOTE` all live in the
    /// first few hundred bytes, so the build id is still perfectly readable
    /// after the entire mapped image has been cut away. This reproduces that
    /// layout directly rather than relying on a host to have it.
    #[test]
    fn a_real_shaped_elf_truncated_to_one_page_falls_back_to_the_digest() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        // A 4 MiB PT_LOAD, like any real binary, with the note early.
        let full = synthetic_elf64(&[0xef; 16], Some(4 * 1024 * 1024));
        assert!(
            full.len() < 2048,
            "the note must sit inside the surviving prefix, or this proves nothing"
        );

        let intact = dir.path().join("intact.so");
        std::fs::write(&intact, &full).expect("write intact");
        // The intact-note prefix, exactly as a 2048-byte truncation would leave it.
        let truncated = dir.path().join("truncated.so");
        let mut prefix = full.clone();
        prefix.resize(2048, 0);
        std::fs::write(&truncated, &prefix).expect("write truncated");

        // Sanity: the note really does survive the cut, so the only thing that
        // can reject this image is the declared-extent check.
        let (_, intact_id) = identify(&intact).expect("intact");
        assert_eq!(
            intact_id.scheme(),
            "sha256",
            "the intact fixture itself declares 4 MiB it does not have, so it \
             must also be rejected; if this ever says bid the check is broken"
        );

        let (_, id) = identify(&truncated).expect("truncated");
        assert_eq!(
            id.scheme(),
            "sha256",
            "an ELF whose PT_LOAD runs past the end of the file must not report \
             a linker id, however intact its note looks"
        );
    }

    /// A file that is not an object image at all must degrade to the digest
    /// rather than failing or returning a constant.
    #[test]
    fn a_non_object_file_falls_back_to_the_digest() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let p = dir.path().join("not-an-object");
        std::fs::write(&p, b"#!/bin/sh\necho hello\n").expect("write");
        let (_, id) = identify(&p).expect("identify");
        assert_eq!(id.scheme(), "sha256");
    }

    /// A truncated object image must not yield a partial or bogus linker id.
    #[test]
    fn a_truncated_object_image_falls_back_to_the_digest() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let p = dir.path().join("truncated");
        let exe = running_image_path().expect("locatable");
        let bytes = std::fs::read(&exe).expect("read exe");
        // Keep enough to look like a header, far too little to be valid.
        std::fs::write(&p, &bytes[..2048]).expect("write");
        let (_, id) = identify(&p).expect("identify");
        assert_eq!(
            id.scheme(),
            "sha256",
            "a truncated image must not produce a linker id"
        );
    }

    /// The two schemes must be distinguishable in a cache key, or a linker id
    /// and a digest that happened to share a hex prefix would collide.
    #[test]
    fn the_schemes_are_tagged_distinctly() {
        let bid = ImageId::LinkerBuildId(vec![0xab; 16]);
        let dig = ImageId::ContentDigest([0xab; 32]);
        assert_ne!(bid.scheme(), dig.scheme());
    }

    /// Set a file's mtime to `secs` past the epoch. `1` is the value Nix
    /// stamps on every store file.
    fn set_mtime(path: &Path, secs: u64) {
        let f = std::fs::File::options()
            .write(true)
            .open(path)
            .expect("open for set_modified");
        f.set_modified(std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs))
            .expect("set_modified");
        assert_eq!(
            std::fs::metadata(path)
                .expect("meta")
                .modified()
                .expect("mtime")
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .expect("post-epoch")
                .as_secs(),
            secs,
            "the fixture must actually control mtime, or these tests are vacuous"
        );
    }
}
