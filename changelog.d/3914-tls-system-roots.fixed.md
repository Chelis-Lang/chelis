Reef fetches and `chelisup install` trust the system certificate roots as well
as the bundled webpki roots. When `SSL_CERT_FILE` names a PEM file, its
certificates replace the system roots. Thus a TLS proxy, a private certificate
authority, and NixOS `security.pki` work. Previously both programs trusted only
the roots in the binary.

When `SSL_CERT_FILE` names a file that does not exist, the fetch stops, and
the error names the variable and the file. A pure Nix shell without `cacert`
sets `SSL_CERT_FILE=/no-cert-file.crt`. There, unset the variable or set it to
a CA bundle. See [#3914](https://github.com/Chelis-Lang/chelis/issues/3914).
