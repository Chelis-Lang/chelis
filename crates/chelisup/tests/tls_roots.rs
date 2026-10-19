//! `chelisup install` trusts the system roots and `SSL_CERT_FILE`, not only
//! the roots compiled into the binary. The download goes to a local TLS
//! server whose certificate only a generated root trusts, so no network is
//! needed.
#![cfg(unix)]

mod common;

use common::{bin, build_fixture_tarball};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// A TLS server on 127.0.0.1 whose certificate, for `localhost`, chains
/// only to a root generated here. `respond` maps a request path to a status
/// and a body. The server answers one request on each connection and records
/// `<status> <path>` for each answered request.
struct TlsServer {
    port: u16,
    root_pem: String,
    answered: Arc<Mutex<Vec<String>>>,
}

impl TlsServer {
    fn start(respond: impl Fn(&str) -> (u16, Vec<u8>) + Send + 'static) -> Self {
        let root_key = rcgen::KeyPair::generate().expect("root key");
        let mut root_params = rcgen::CertificateParams::new(Vec::new()).expect("root params");
        root_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        root_params
            .distinguished_name
            .push(rcgen::DnType::CommonName, "chelis test root");
        let root_cert = root_params
            .self_signed(&root_key)
            .expect("root certificate");
        let root = rcgen::Issuer::new(root_params, root_key);

        let leaf_key = rcgen::KeyPair::generate().expect("leaf key");
        let leaf_params =
            rcgen::CertificateParams::new(vec!["localhost".to_string()]).expect("leaf params");
        let leaf_cert = leaf_params
            .signed_by(&leaf_key, &root)
            .expect("leaf certificate");

        let config = rustls::ServerConfig::builder()
            .with_safe_defaults()
            .with_no_client_auth()
            .with_single_cert(
                vec![rustls::Certificate(leaf_cert.der().to_vec())],
                rustls::PrivateKey(leaf_key.serialize_der()),
            )
            .expect("server config");
        let config = Arc::new(config);
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        let answered = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&answered);
        std::thread::spawn(move || {
            for tcp in listener.incoming().flatten() {
                let _ = tcp.set_read_timeout(Some(Duration::from_secs(10)));
                let Ok(conn) = rustls::ServerConnection::new(Arc::clone(&config)) else {
                    continue;
                };
                let mut tls = rustls::StreamOwned::new(conn, tcp);
                // A client that rejects the certificate ends the handshake
                // here, and no request arrives.
                let Some(path) = read_request_path(&mut tls) else {
                    continue;
                };
                let (status, body) = respond(&path);
                let head = format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                if tls.write_all(head.as_bytes()).is_err() || tls.write_all(&body).is_err() {
                    continue;
                }
                tls.conn.send_close_notify();
                let _ = tls.flush();
                log.lock().expect("log").push(format!("{status} {path}"));
            }
        });
        Self {
            port,
            root_pem: root_cert.pem(),
            answered,
        }
    }

    fn answered(&self) -> Vec<String> {
        self.answered.lock().expect("log").clone()
    }
}

/// Read one HTTP request head and return its path. `None` when the TLS
/// handshake or the read fails.
fn read_request_path(stream: &mut impl Read) -> Option<String> {
    let mut head = Vec::new();
    let mut buf = [0u8; 1024];
    while !head.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut buf).ok()?;
        if n == 0 {
            return None;
        }
        head.extend_from_slice(&buf[..n]);
    }
    let head = String::from_utf8_lossy(&head);
    let line = head.lines().next()?;
    line.split(' ').nth(1).map(str::to_string)
}

/// Run `chelisup install <version>` against the HTTPS release base, with
/// `SSL_CERT_FILE` set to `root_file` or removed.
fn install(
    home: &std::path::Path,
    base: &str,
    root_file: Option<&std::path::Path>,
) -> std::process::Output {
    let mut cmd = Command::new(bin());
    cmd.args(["install", "0.1.0"])
        .current_dir(home)
        .env("CHELIS_HOME", home)
        .env("CHELISUP_RELEASE_BASE", base)
        .env_remove("CHELIS_TOOLCHAIN")
        .env_remove("SSL_CERT_FILE");
    if let Some(root_file) = root_file {
        cmd.env("SSL_CERT_FILE", root_file);
    }
    cmd.output().expect("run chelisup")
}

#[test]
fn install_trusts_the_root_that_ssl_cert_file_names() {
    let slug = chelisup::install::detect_slug().expect("host platform must be supported in CI");
    let build = chelisup::install::release_build("0.1.0", slug, chelisup::install::host_is_musl())
        .expect("release build");
    let release = tempfile::tempdir().expect("tempdir");
    let tarball = build_fixture_tarball(release.path(), "0.1.0", &build);
    let asset = format!("/{}", tarball.file_name().expect("name").to_string_lossy());
    let bytes = std::fs::read(&tarball).expect("read tarball");
    let served = asset.clone();
    let server = TlsServer::start(move |path| {
        if path == served {
            (200, bytes.clone())
        } else {
            (404, Vec::new())
        }
    });
    let base = format!("https://localhost:{}", server.port);
    let scratch = tempfile::tempdir().expect("tempdir");
    let root_file = scratch.path().join("root.pem");
    std::fs::write(&root_file, &server.root_pem).expect("write root");

    let home = tempfile::tempdir().expect("tempdir");
    let untrusted = install(home.path(), &base, None);
    let stderr = String::from_utf8_lossy(&untrusted.stderr);
    assert!(!untrusted.status.success(), "stderr: {stderr}");
    assert!(
        stderr.contains("network error fetching") && stderr.contains("certificate"),
        "without SSL_CERT_FILE the download must fail on the certificate, stderr: {stderr}"
    );
    assert!(!home.path().join("toolchains/0.1.0").exists());
    assert!(server.answered().is_empty(), "{:?}", server.answered());

    // A file that does not exist stops the client build, and the error names
    // the variable and the file.
    let missing_file = scratch.path().join("missing.pem");
    let missing = install(home.path(), &base, Some(&missing_file));
    let stderr = String::from_utf8_lossy(&missing.stderr);
    assert!(!missing.status.success(), "stderr: {stderr}");
    assert!(
        stderr.contains("could not build the HTTP client")
            && stderr.contains(&format!("SSL_CERT_FILE={}", missing_file.display())),
        "stderr: {stderr}"
    );
    assert!(server.answered().is_empty(), "{:?}", server.answered());

    let trusted = install(home.path(), &base, Some(&root_file));
    assert!(
        trusted.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&trusted.stderr)
    );
    assert!(home.path().join("toolchains/0.1.0/bin/chelis").is_file());
    assert_eq!(server.answered(), [format!("200 {asset}")]);
}
