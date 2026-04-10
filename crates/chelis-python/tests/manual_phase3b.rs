use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[test]
#[ignore = "manual phase 3b acceptance gate: requires py/.venv with PyTorch and installs bindings/python into that venv"]
fn phase3b_python_manual_acceptance_oracle() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let python = repo_root.join("py/.venv/bin/python");
    assert!(python.exists(), "expected {}", python.display());

    let status = Command::new("uv")
        .arg("pip")
        .arg("install")
        .arg("--python")
        .arg(&python)
        .arg("-e")
        .arg(repo_root.join("bindings/python"))
        .current_dir(&repo_root)
        .status()
        .expect("install chelis package");
    assert!(status.success(), "uv pip install failed");

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test port");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let mut tide = Command::new("cargo")
        .arg("run")
        .arg("-p")
        .arg("chelis-cli")
        .arg("--")
        .arg("tide")
        .arg("serve")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .current_dir(&repo_root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn tide server");

    wait_for_tide(port, &mut tide);

    let status = Command::new(&python)
        .arg(repo_root.join("bindings/python/tests/manual_phase3b.py"))
        .env("CHELIS_TIDE_URL", format!("http://127.0.0.1:{port}"))
        .current_dir(&repo_root)
        .status()
        .expect("run manual python acceptance");

    let _ = tide.kill();
    let _ = tide.wait();

    assert!(status.success(), "manual phase 3b python acceptance failed");
}

fn wait_for_tide(port: u16, tide: &mut Child) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if let Some(status) = tide.try_wait().expect("poll tide server") {
            panic!("tide server exited early with status {status}");
        }
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
    let _ = tide.kill();
    panic!("timed out waiting for tide server on port {port}");
}
