//! Executable scripts used by the toolchain unit tests.

use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

pub(super) fn write_executable(path: &Path, contents: impl AsRef<[u8]>) {
    write_executable_while(path, contents.as_ref(), || {});
}

fn write_executable_while(path: &Path, contents: &[u8], while_writing: impl FnOnce()) {
    let mut file = std::fs::File::create(path).unwrap();
    while_writing();
    file.write_all(contents).unwrap();
    drop(file);
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
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
        assert_eq!(result.expect("the published script is executable").code(), Some(expected));
    }
}

#[cfg(target_os = "linux")]
#[test]
fn renaming_does_not_release_an_inherited_writable_inode() {
    use std::os::unix::fs::MetadataExt;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unpublished");
    let renamed = dir.path().join("published");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(b"#!/bin/sh\nexit 0\n").unwrap();
    file.set_permissions(std::fs::Permissions::from_mode(0o755)).unwrap();
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
    assert!(std::process::Command::new(&renamed).status().unwrap().success());
}
