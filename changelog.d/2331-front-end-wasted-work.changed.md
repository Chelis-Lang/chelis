`chelis build` no longer formats an ownership-verification diagnostic that the
success path discards, and no longer desugars a single-file program twice. The
diagnostic text, the accepted and rejected programs, and the emitted artifacts
are unchanged; only the wasted work is gone. See
[#2331](https://github.com/Chelis-Lang/chelis/issues/2331).
