Reef fetches and `chelisup install` trust the system certificate roots as well
as the bundled webpki roots. When `SSL_CERT_FILE` names a PEM file, its
certificates replace the system roots. Thus a TLS proxy, a private certificate
authority, and NixOS `security.pki` work. Previously both programs trusted only
the roots in the binary. See
[#3914](https://github.com/Chelis-Lang/chelis/issues/3914).
