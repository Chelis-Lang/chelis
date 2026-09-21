`chelis build` no longer formats an ownership-verification diagnostic that the
success path discards, and no longer desugars a single-file program twice. The
first helps every build; the second helps only a build outside a reef package,
because a package build's second pass already covers just the entry module. The
emitted artifacts are unchanged, and the live-owner overflow diagnostic is
byte-identical on the path that raises it. See
[#2331](https://github.com/Chelis-Lang/chelis/issues/2331).
