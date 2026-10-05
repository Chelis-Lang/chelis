//! Name-service lookups in the static Linux build.
//!
//! A statically linked glibc still reads `/etc/nsswitch.conf`. For a service
//! it does not carry (systemd's `resolve`, `myhostname`, and `mymachines`, or
//! `sss` and `ldap`) it `dlopen`s the host's `libnss_<service>.so.2`, which
//! links the host's `libc.so.6`: a second C library this process cannot host.
//! On Fedora 44 and Arch Linux, whose `hosts` line names those services, the
//! first host lookup dies with SIGFPE. The static build therefore points the
//! databases it uses at the services built into libc, so lookups read
//! `/etc/hosts`, `/etc/passwd`, `/etc/group`, and the name servers in
//! `/etc/resolv.conf`, and never load a plugin. Dynamic builds keep the host's
//! configuration.

/// Restrict this process's host, user, and group lookups to glibc's built-in
/// `files` and `dns` services. It changes nothing except in the static Linux
/// build. Call it before the first lookup.
pub fn use_builtin_services() {
    // Every Linux glibc build compiles this, so CI checks it; only the static
    // build runs it.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    if cfg!(target_feature = "crt-static") {
        use std::ffi::{c_char, c_int};

        unsafe extern "C" {
            /// `<nss.h>`: replace the services glibc consults for one database.
            fn __nss_configure_lookup(dbname: *const c_char, service_line: *const c_char) -> c_int;
        }

        for (database, services) in [
            (c"hosts", c"files dns"),
            (c"passwd", c"files"),
            (c"group", c"files"),
        ] {
            // SAFETY: both arguments are NUL-terminated static strings, and
            // glibc parses the service line into storage of its own.
            let status = unsafe { __nss_configure_lookup(database.as_ptr(), services.as_ptr()) };
            debug_assert_eq!(status, 0, "glibc rejected the {database:?} services");
        }
    }
}
