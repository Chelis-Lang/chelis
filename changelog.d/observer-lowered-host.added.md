The opt-in native emission observer now exposes the initial host-lowering snapshot
alongside the selected verified payload, so consumers can inspect scalar losses
pruned from tuple-gradient artifacts. Ordinary compilation and wire output are
unchanged; the additional snapshot is not a certificate or an emitted function.
