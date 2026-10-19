Reef locks retain a binary entry for each declared platform. Fetch plans and offline imports select only the build-platform artifact. Use `reef fetch-plan --platform <slug>` to select another platform.

Reef rejects duplicate binary identities and inconsistent hash pins before an install can replace bytes. See [#3923](https://github.com/Chelis-Lang/chelis/issues/3923).
