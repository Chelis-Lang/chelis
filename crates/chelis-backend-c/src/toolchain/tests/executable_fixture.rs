//! Executable scripts used by the toolchain unit tests.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

pub(super) fn write_executable(path: &Path, contents: impl AsRef<[u8]>) {
    write_executable_while(path, contents.as_ref(), || {});
}

fn test_name(name: &str) -> String {
    format!("{}::{name}", module_path!().split_once("::").unwrap().1)
}

fn write_executable_while(path: &Path, contents: &[u8], while_writing: impl FnOnce()) {
    use std::process::{Command, Stdio};

    // A concurrent fork inherits even CLOEXEC descriptors until it execs. Only
    // the writer child opens this inode for writing; the parent can keep spawning
    // unrelated processes without handing them an executable's write handle.
    let test = test_name("writer_child");
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test, "--nocapture", "--quiet"])
        .env("CHELIS_EXECUTABLE_FIXTURE_PATH", path)
        .env(
            "CHELIS_EXECUTABLE_FIXTURE_LENGTH",
            contents.len().to_string(),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    loop {
        line.clear();
        if output.read_line(&mut line).unwrap() == 0 {
            let result = child.wait_with_output().unwrap();
            panic!("fixture writer did not open {}: {result:?}", path.display());
        }
        if line == "fixture writer ready\n" {
            break;
        }
    }
    child.stdout = Some(output.into_inner());
    // The regression parks an unrelated child here, while the writer owns the
    // open file. This handshake makes the fork/exec lifetime test deterministic.
    while_writing();
    let written = child.stdin.take().unwrap().write_all(contents);
    let result = child.wait_with_output().unwrap();
    assert!(result.status.success(), "fixture writer: {result:?}");
    written.unwrap();
}

/// The negative control deliberately creates an inheritable write handle. Keep
/// the lifetime probes in their own test processes so a sibling probe cannot
/// inherit that handle and prolong its lifetime beyond the selected handshake.
#[cfg(target_os = "linux")]
fn in_lifetime_child(name: &str) -> bool {
    let test = test_name(name);
    if std::env::var("CHELIS_FIXTURE_LIFETIME_CHILD").as_deref() == Ok(&test) {
        return false;
    }
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test, "--nocapture"])
        .env("CHELIS_FIXTURE_LIFETIME_CHILD", &test)
        .output()
        .unwrap();
    assert!(result.status.success(), "{test}: {result:?}");
    true
}

/// The sole writer of executable fixtures. It spawns no children, and its exit
/// closes every write handle before the parent can execute the script.
#[test]
fn writer_child() {
    use std::io::Read;

    let Some(path) = std::env::var_os("CHELIS_EXECUTABLE_FIXTURE_PATH") else {
        return;
    };
    let length: u64 = std::env::var("CHELIS_EXECUTABLE_FIXTURE_LENGTH")
        .unwrap()
        .parse()
        .unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.set_permissions(std::fs::Permissions::from_mode(0o755))
        .unwrap();
    println!("fixture writer ready");
    std::io::stdout().flush().unwrap();
    // A concurrent fork can inherit the parent's pipe writer too. Read exactly
    // the declared payload, never wait for EOF from unrelated pre-exec children.
    let copied = std::io::copy(&mut std::io::stdin().lock().take(length), &mut file).unwrap();
    assert_eq!(copied, length, "truncated executable fixture");
}

#[test]
fn writes_exact_bytes_and_edits_the_same_inode() {
    use std::os::unix::fs::MetadataExt;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wrapper with spaces");
    let mut inode = None;
    for (script, code) in [("#!/bin/sh\nexit 0\n", 0), ("#!/bin/sh\nexit 7\n", 7)] {
        write_executable(&path, script);
        assert_eq!(std::fs::read(&path).unwrap(), script.as_bytes());
        let metadata = path.metadata().unwrap();
        assert_eq!(metadata.permissions().mode() & 0o777, 0o755);
        if let Some(previous) = inode {
            assert_eq!(metadata.ino(), previous, "the edit stays in place");
        }
        inode = Some(metadata.ino());
        assert_eq!(
            std::process::Command::new(&path).status().unwrap().code(),
            Some(code)
        );
    }
}

/// A concurrent spawn stopped between fork and exec. The handshakes, rather
/// than timing or repeated executions, select the inherited-descriptor window.
#[cfg(target_os = "linux")]
struct BeforeExec {
    release: std::os::unix::net::UnixStream,
    thread: Option<std::thread::JoinHandle<()>>,
}

#[cfg(target_os = "linux")]
impl BeforeExec {
    fn hold() -> Self {
        use std::io::Read;
        use std::os::unix::net::UnixStream;
        use std::os::unix::process::CommandExt;
        use std::process::Command;

        let (mut ready, mut signal) = UnixStream::pair().unwrap();
        let (release, mut wait) = UnixStream::pair().unwrap();
        let thread = std::thread::spawn(move || {
            let mut command = Command::new("/bin/true");
            // SAFETY: the child only performs socket read/write syscalls before
            // exec; it takes no locks, allocates nothing, and calls no Rust code
            // that inspects process-global state.
            unsafe {
                command.pre_exec(move || {
                    signal.write_all(b"r")?;
                    let mut byte = [0];
                    wait.read_exact(&mut byte)?;
                    Ok(())
                });
            }
            assert!(command.status().unwrap().success());
        });
        let mut byte = [0];
        ready.read_exact(&mut byte).unwrap();
        assert_eq!(&byte, b"r");
        Self {
            release,
            thread: Some(thread),
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for BeforeExec {
    fn drop(&mut self) {
        let released = self.release.write_all(b"x");
        let joined = self.thread.take().unwrap().join();
        if !std::thread::panicking() {
            released.unwrap();
            joined.unwrap();
        }
    }
}

#[cfg(target_os = "linux")]
#[test]
fn executable_is_ready_while_an_unrelated_child_is_before_exec() {
    if in_lifetime_child("executable_is_ready_while_an_unrelated_child_is_before_exec") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wrapper");
    // Both first publication and an edited fixture must be ready to execute.
    for script in [b"#!/bin/sh\nexit 0\n".as_slice(), b"#!/bin/sh\nexit 7\n"] {
        let mut inherited = None;
        write_executable_while(&path, script, || inherited = Some(BeforeExec::hold()));
        assert_eq!(std::fs::read(&path).unwrap(), script);
        assert_eq!(path.metadata().unwrap().permissions().mode() & 0o777, 0o755);
        let result = std::process::Command::new(&path).status();
        // Always release and reap the parked child, including the red control.
        drop(inherited);
        let expected = if script.ends_with(b"0\n") { 0 } else { 7 };
        assert_eq!(
            result.expect("the published script is executable").code(),
            Some(expected)
        );
    }
}

#[cfg(target_os = "linux")]
#[test]
fn renaming_does_not_release_an_inherited_writable_inode() {
    use std::os::unix::fs::MetadataExt;

    if in_lifetime_child("renaming_does_not_release_an_inherited_writable_inode") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unpublished");
    let renamed = dir.path().join("published");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(b"#!/bin/sh\nexit 0\n").unwrap();
    file.set_permissions(std::fs::Permissions::from_mode(0o755))
        .unwrap();
    let inode = file.metadata().unwrap().ino();
    let inherited = BeforeExec::hold();
    drop(file);
    let original = std::process::Command::new(&path).status();
    std::fs::rename(&path, &renamed).unwrap();
    assert_eq!(renamed.metadata().unwrap().ino(), inode);
    let published = std::process::Command::new(&renamed).status();
    drop(inherited);
    for result in [original, published] {
        assert_eq!(result.unwrap_err().raw_os_error(), Some(26), "ETXTBSY");
    }
    assert!(
        std::process::Command::new(&renamed)
            .status()
            .unwrap()
            .success()
    );
}
