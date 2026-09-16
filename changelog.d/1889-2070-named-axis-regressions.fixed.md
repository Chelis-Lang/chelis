Generic helper lowering now derives runtime extent witnesses and result claims
from the authored signature, so caller-side labels cannot merge unrelated
parameter binders or distinct rank-spread axes. This repairs the two named-axis
regressions introduced by #2070; #1889's callable-alias and worker/disk
residuals remain open. See [#1889](https://github.com/Chelis-Lang/chelis/issues/1889).
