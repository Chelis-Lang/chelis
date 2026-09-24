use crate::*;
use object::{Object, ObjectSection};

pub const RECORD_LENGTH: usize = 248;
const MAGIC: &[u8; 16] = b"CHELIS-RT-ID\0\0\0\0";

fn kind_byte(kind: RecordKind) -> u8 {
    match kind {
        RecordKind::Runtime => 0,
        RecordKind::Cli => 1,
        RecordKind::Python => 2,
    }
}

pub fn encode_record(
    descriptor: &Descriptor,
    kind: RecordKind,
) -> Result<[u8; RECORD_LENGTH], RecordError> {
    if descriptor.schema_version != 1 {
        return Err(RecordError::Unsupported {
            reason: format!("descriptor schema {}", descriptor.schema_version),
        });
    }
    let mut bytes = [0; RECORD_LENGTH];
    bytes[..16].copy_from_slice(MAGIC);
    bytes[16..18].copy_from_slice(&descriptor.schema_version.to_le_bytes());
    bytes[18] = kind_byte(kind);
    bytes[20..24].copy_from_slice(&224u32.to_le_bytes());
    for (chunk, digest) in bytes[24..].as_chunks_mut::<32>().0.iter_mut().zip([
        descriptor.recipe,
        descriptor.source,
        descriptor.interface,
        descriptor.target,
        descriptor.features,
        descriptor.compile,
        descriptor.toolchain,
    ]) {
        chunk.copy_from_slice(digest.as_bytes());
    }
    Ok(bytes)
}

pub fn decode_record(bytes: &[u8], kind: RecordKind) -> Result<Descriptor, RecordError> {
    if bytes.len() < RECORD_LENGTH {
        return Err(RecordError::Truncated {
            reason: format!("expected {RECORD_LENGTH} bytes, got {}", bytes.len()),
        });
    }
    if bytes.len() != RECORD_LENGTH {
        return Err(RecordError::Malformed {
            reason: "noncanonical record length or trailing bytes".into(),
        });
    }
    if &bytes[..16] != MAGIC {
        return Err(RecordError::Malformed {
            reason: "invalid magic".into(),
        });
    }
    let version = u16::from_le_bytes([bytes[16], bytes[17]]);
    if version != 1 {
        return Err(RecordError::Unsupported {
            reason: format!("record version {version}"),
        });
    }
    if bytes[18] > 2 {
        return Err(RecordError::Unsupported {
            reason: format!("record kind {}", bytes[18]),
        });
    }
    if bytes[18] != kind_byte(kind) {
        return Err(RecordError::Malformed {
            reason: "record kind does not match its section role".into(),
        });
    }
    if bytes[19] != 0 {
        return Err(RecordError::Malformed {
            reason: "nonzero reserved byte".into(),
        });
    }
    if bytes[20..24] != 224u32.to_le_bytes() {
        return Err(RecordError::Malformed {
            reason: "noncanonical payload length".into(),
        });
    }
    fn digest(bytes: &[u8], offset: usize) -> ContentDigest {
        let mut value = [0; 32];
        value.copy_from_slice(&bytes[offset..offset + 32]);
        ContentDigest::from_bytes(value)
    }
    Ok(Descriptor {
        schema_version: version,
        recipe: digest(bytes, 24),
        source: digest(bytes, 56),
        interface: digest(bytes, 88),
        target: digest(bytes, 120),
        features: digest(bytes, 152),
        compile: digest(bytes, 184),
        toolchain: digest(bytes, 216),
    })
}

fn section_names(kind: RecordKind) -> (&'static str, &'static str) {
    match kind {
        RecordKind::Runtime => (".chelis.runtime.id", "__ch_rt_id"),
        RecordKind::Cli => (".chelis.runtime.expect.cli", "__ch_rt_cli"),
        RecordKind::Python => (".chelis.runtime.expect.python", "__ch_rt_py"),
    }
}

fn provenance_names(kind: RecordKind) -> (&'static str, &'static str) {
    match kind {
        RecordKind::Runtime => (".chelis.runtime.provenance", "__ch_rt_prov"),
        RecordKind::Cli => (".chelis.runtime.provenance.cli", "__ch_prv_cli"),
        RecordKind::Python => (".chelis.runtime.provenance.python", "__ch_prv_py"),
    }
}

#[derive(Default)]
struct NativeRecords {
    identity: Option<Descriptor>,
    provenance: Option<BuildProvenance>,
}

impl NativeRecords {
    fn provenance(self) -> Result<BuildProvenance, RecordError> {
        let identity = self.identity.ok_or(RecordError::Missing)?;
        let provenance = self.provenance.ok_or(RecordError::Missing)?;
        if let BuildProvenance::SealedDistribution { source_closure } = &provenance
            && *source_closure != identity.source
        {
            return Err(RecordError::Malformed {
                reason: "sealed provenance does not bind the retained source closure".into(),
            });
        }
        Ok(provenance)
    }
}

fn object_error(error: object::Error) -> RecordError {
    RecordError::Malformed {
        reason: error.to_string(),
    }
}

// Bounds checks distinguish truncated containers from structurally malformed
// ones before object performs its complete, format-aware section traversal.
fn bounded_range(bytes: &[u8], offset: u64, size: u64, what: &str) -> Result<(), RecordError> {
    if offset
        .checked_add(size)
        .is_none_or(|end| end > bytes.len() as u64)
    {
        return Err(RecordError::Truncated {
            reason: what.into(),
        });
    }
    Ok(())
}

fn integer(bytes: &[u8], offset: usize, width: usize, little: bool) -> u64 {
    let mut value = 0;
    for index in 0..width {
        let shift = if little { index } else { width - 1 - index };
        value |= u64::from(bytes[offset + index]) << (shift * 8);
    }
    value
}

fn native_bounds(bytes: &[u8], macho: bool, minimum: usize) -> Result<(), RecordError> {
    if macho {
        let little = matches!(
            &bytes[..4],
            [0xce, 0xfa, 0xed, 0xfe] | [0xcf, 0xfa, 0xed, 0xfe]
        );
        let count = integer(bytes, 16, 4, little);
        let size = integer(bytes, 20, 4, little);
        bounded_range(bytes, minimum as u64, size, "Mach-O load commands")?;
        if count > size / 8 {
            return Err(RecordError::Malformed {
                reason: "Mach-O load command count exceeds command bytes".into(),
            });
        }
        let end = minimum + size as usize;
        let alignment = if minimum == 32 { 8 } else { 4 };
        let mut offset = minimum;
        for _ in 0..count {
            if end - offset < 8 {
                return Err(RecordError::Malformed {
                    reason: "incomplete Mach-O load command header".into(),
                });
            }
            let command_size = integer(bytes, offset + 4, 4, little) as usize;
            if command_size < 8
                || !command_size.is_multiple_of(alignment)
                || command_size > end - offset
            {
                return Err(RecordError::Malformed {
                    reason: "invalid Mach-O load command extent".into(),
                });
            }
            offset += command_size;
        }
        if offset != end {
            return Err(RecordError::Malformed {
                reason: "unvisited Mach-O load command bytes".into(),
            });
        }
    } else {
        let little = match bytes[5] {
            1 => true,
            2 => false,
            value => {
                return Err(RecordError::Unsupported {
                    reason: format!("ELF byte order {value}"),
                });
            }
        };
        if bytes[6] != 1 || integer(bytes, 20, 4, little) != 1 {
            return Err(RecordError::Unsupported {
                reason: "ELF version".into(),
            });
        }
        let (width, program_offset, section_offset, sizes) = if minimum == 64 {
            (8, 32, 40, 54)
        } else {
            (4, 28, 32, 42)
        };
        let phoff = integer(bytes, program_offset, width, little);
        let shoff = integer(bytes, section_offset, width, little);
        let phsize = integer(bytes, sizes, 2, little);
        let phcount = integer(bytes, sizes + 2, 2, little);
        let shsize = integer(bytes, sizes + 4, 2, little);
        let mut shcount = integer(bytes, sizes + 6, 2, little);
        if shcount == 0 && shoff != 0 {
            // ELF extended section counts live in the null section's sh_size.
            let section_size = if width == 8 { 64 } else { 40 };
            bounded_range(bytes, shoff, section_size, "ELF null section header")?;
            shcount = integer(
                bytes,
                shoff as usize + if width == 8 { 32 } else { 20 },
                width,
                little,
            );
        }
        let section_bytes = shsize
            .checked_mul(shcount)
            .ok_or_else(|| RecordError::Malformed {
                reason: "ELF section count overflow".into(),
            })?;
        bounded_range(bytes, shoff, section_bytes, "ELF section headers")?;
        bounded_range(bytes, phoff, phsize * phcount, "ELF program headers")?;
    }
    Ok(())
}

fn archive_bounds(bytes: &[u8]) -> Result<(), RecordError> {
    let mut offset = 8usize;
    while offset < bytes.len() {
        bounded_range(bytes, offset as u64, 60, "archive member header")?;
        let header = &bytes[offset..offset + 60];
        if &header[58..] != b"`\n" {
            return Err(RecordError::Malformed {
                reason: "archive member header terminator".into(),
            });
        }
        let size = std::str::from_utf8(&header[48..58])
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .ok_or_else(|| RecordError::Malformed {
                reason: "archive member size".into(),
            })?;
        bounded_range(bytes, (offset + 60) as u64, size, "archive member data")?;
        offset += 60 + size as usize;
        if size % 2 != 0 {
            bounded_range(bytes, offset as u64, 1, "archive member alignment")?;
            offset += 1;
        }
    }
    Ok(())
}

fn native_record(
    bytes: &[u8],
    kind: RecordKind,
    retained: &mut NativeRecords,
) -> Result<(), RecordError> {
    if bytes.len() < 16 {
        return Err(RecordError::Truncated {
            reason: "native file header".into(),
        });
    }
    let format = object::FileKind::parse(bytes).map_err(object_error)?;
    let (minimum, macho) = match format {
        object::FileKind::Elf32 => (52, false),
        object::FileKind::Elf64 => (64, false),
        object::FileKind::MachO32 => (28, true),
        object::FileKind::MachO64 => (32, true),
        _ => {
            return Err(RecordError::Unsupported {
                reason: format!("native format {format:?}"),
            });
        }
    };
    if bytes.len() < minimum {
        return Err(RecordError::Truncated {
            reason: "native file header".into(),
        });
    }
    native_bounds(bytes, macho, minimum)?;
    let file = object::File::parse(bytes).map_err(object_error)?;
    let (elf_name, macho_name) = section_names(kind);
    let (elf_provenance, macho_provenance) = provenance_names(kind);
    for section in file.sections() {
        if let Some((offset, size)) = section.file_range() {
            bounded_range(bytes, offset, size, "native section data")?;
        }
        let name = section.name().map_err(object_error)?;
        let identity = name == if macho { macho_name } else { elf_name };
        let provenance = name
            == if macho {
                macho_provenance
            } else {
                elf_provenance
            };
        if !identity && !provenance {
            continue;
        }
        if (identity && retained.identity.is_some())
            || (provenance && retained.provenance.is_some())
        {
            return Err(RecordError::Duplicate);
        }
        if macho && section.segment_name().map_err(object_error)? != Some("__DATA") {
            return Err(RecordError::Malformed {
                reason: "identity section is outside __DATA".into(),
            });
        }
        let (offset, size) = section.file_range().ok_or_else(|| RecordError::Malformed {
            reason: "identity section has no file data".into(),
        })?;
        if offset
            .checked_add(size)
            .is_none_or(|end| end > bytes.len() as u64)
        {
            return Err(RecordError::Truncated {
                reason: "identity section extends beyond native image".into(),
            });
        }
        if section
            .compressed_file_range()
            .map_err(object_error)?
            .format
            != object::CompressionFormat::None
        {
            return Err(RecordError::Unsupported {
                reason: "compressed identity section".into(),
            });
        }
        let data = section.data().map_err(object_error)?;
        if identity {
            let (records, trailing) = data.as_chunks::<RECORD_LENGTH>();
            if records.len() > 1
                && trailing.is_empty()
                && records
                    .iter()
                    .all(|record| decode_record(record, kind).is_ok())
            {
                return Err(RecordError::Duplicate);
            }
            retained.identity = Some(decode_record(data, kind)?);
        } else {
            retained.provenance =
                Some(
                    decode_provenance(data).map_err(|error| RecordError::Malformed {
                        reason: error.to_string(),
                    })?,
                );
        }
    }
    Ok(())
}

pub fn decode_image(bytes: &[u8], kind: RecordKind) -> Result<Descriptor, RecordError> {
    let mut retained = NativeRecords::default();
    native_record(bytes, kind, &mut retained)?;
    retained.identity.ok_or(RecordError::Missing)
}

pub fn decode_image_provenance(
    bytes: &[u8],
    kind: RecordKind,
) -> Result<BuildProvenance, RecordError> {
    let mut retained = NativeRecords::default();
    native_record(bytes, kind, &mut retained)?;
    retained.provenance()
}

fn archive_records(bytes: &[u8]) -> Result<NativeRecords, RecordError> {
    if bytes.len() < 8 {
        return Err(RecordError::Truncated {
            reason: "archive header".into(),
        });
    }
    if bytes.starts_with(b"!<thin>\n") {
        return Err(RecordError::Unsupported {
            reason: "thin archives require external data".into(),
        });
    }
    if !bytes.starts_with(b"!<arch>\n") {
        return Err(RecordError::Unsupported {
            reason: "not a Unix native archive".into(),
        });
    }
    archive_bounds(bytes)?;
    let archive = object::read::archive::ArchiveFile::parse(bytes).map_err(object_error)?;
    if archive.is_thin() {
        return Err(RecordError::Unsupported {
            reason: "thin archive".into(),
        });
    }
    let mut retained = NativeRecords::default();
    for member in archive.members() {
        let member = member.map_err(object_error)?;
        let (offset, size) = member.file_range();
        if offset
            .checked_add(size)
            .is_none_or(|end| end > bytes.len() as u64)
        {
            return Err(RecordError::Truncated {
                reason: "archive member extends beyond archive".into(),
            });
        }
        native_record(
            member.data(bytes).map_err(object_error)?,
            RecordKind::Runtime,
            &mut retained,
        )?;
    }
    Ok(retained)
}

pub fn decode_archive(bytes: &[u8]) -> Result<Descriptor, RecordError> {
    archive_records(bytes)?.identity.ok_or(RecordError::Missing)
}

pub fn decode_archive_provenance(bytes: &[u8]) -> Result<BuildProvenance, RecordError> {
    archive_records(bytes)?.provenance()
}
