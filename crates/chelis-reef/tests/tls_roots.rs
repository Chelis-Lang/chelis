//! Reef GitHub fetches trust the system roots and `SSL_CERT_FILE`, not only
//! the roots compiled into the binary. The fetch goes to a local TLS server
//! whose certificate only a generated root trusts, so no network is needed.
//!
//! This target holds one test because the test sets process environment.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chelis_reef::{GitHubFetchError, install_from_github};

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

const METADATA_PATH: &str = "/repos/chelis-lang/tls-probe/releases/tags/v0.1.0";

#[test]
fn github_fetch_trusts_the_root_that_ssl_cert_file_names() {
    let server = TlsServer::start(|path| {
        if path == METADATA_PATH {
            (200, br#"{"assets": []}"#.to_vec())
        } else {
            (404, Vec::new())
        }
    });
    let dir = tempfile::tempdir().expect("tempdir");
    let root_file = dir.path().join("root.pem");
    std::fs::write(&root_file, &server.root_pem).expect("write root");
    let registry = dir.path().join("registry");

    // SAFETY: this target holds only this test, and the server thread reads
    // no environment, so no other thread reads or writes it now.
    unsafe {
        std::env::set_var(
            "CHELIS_REEF_GITHUB_BASE_API",
            format!("https://localhost:{}", server.port),
        );
        // A token stops the `gh auth token` fallback from running.
        std::env::set_var("GITHUB_TOKEN", "test-token");
        std::env::remove_var("SSL_CERT_FILE");
    }
    let untrusted = install_from_github("chelis-lang/tls-probe@v0.1.0", &registry);
    match &untrusted {
        Err(GitHubFetchError::Network { message, .. }) => assert!(
            message.contains("certificate"),
            "the fetch must fail on the certificate, got: {message}"
        ),
        other => panic!("without SSL_CERT_FILE the fetch must fail on TLS, got: {other:?}"),
    }
    assert!(server.answered().is_empty(), "{:?}", server.answered());

    // A file that does not exist stops the client build, and the error names
    // the variable and the file.
    let missing_file = dir.path().join("missing.pem");
    // SAFETY: as above.
    unsafe {
        std::env::set_var("SSL_CERT_FILE", &missing_file);
    }
    let missing = install_from_github("chelis-lang/tls-probe@v0.1.0", &registry);
    match &missing {
        Err(GitHubFetchError::Io { message }) => assert!(
            message.starts_with("build http client: ")
                && message.contains(&format!("SSL_CERT_FILE={}", missing_file.display())),
            "{message}"
        ),
        other => panic!("a missing SSL_CERT_FILE must stop the client build, got: {other:?}"),
    }
    assert!(server.answered().is_empty(), "{:?}", server.answered());

    // SAFETY: as above.
    unsafe {
        std::env::set_var("SSL_CERT_FILE", &root_file);
    }
    let trusted = install_from_github("chelis-lang/tls-probe@v0.1.0", &registry);
    match &trusted {
        // The metadata fetch succeeded: the empty asset list is the error.
        Err(GitHubFetchError::ReleaseAssetNotFound { asset_name, .. }) => assert!(
            asset_name.contains("release has no attached assets"),
            "{asset_name}"
        ),
        other => panic!("with SSL_CERT_FILE the metadata fetch must succeed, got: {other:?}"),
    }
    assert_eq!(server.answered(), [format!("200 {METADATA_PATH}")]);
}
