# Third-party notices

geg's own source code is MIT-licensed. Its dependencies and model weights retain their own licenses. Exact dependency versions are recorded in `Cargo.lock`.

## nucleo and nucleo-matcher

- nucleo 0.5.0: <https://crates.io/crates/nucleo/0.5.0>
- nucleo-matcher 0.3.1: <https://crates.io/crates/nucleo-matcher/0.3.1>
- License: Mozilla Public License 2.0 (MPL-2.0), <https://www.mozilla.org/MPL/2.0/>
- Upstream: <https://github.com/helix-editor/nucleo>
- Corresponding unmodified source is available from the versioned crates.io pages above, including the upstream license and copyright notices. `cargo vendor` can also retrieve the exact source for a locked build.

These unmodified libraries implement asynchronous fuzzy matching and fuzzy scoring. geg's own terminal UI and integration are independently written; the dependencies are not relicensed under geg's MIT license.

## Design references, not bundled code

The dust and fzf projects informed algorithm and interaction design. Their source code is not copied or bundled:

- dust: <https://github.com/bootandy/dust>
- fzf: <https://github.com/junegunn/fzf>

## Models and native libraries

EmbeddingGemma/GGUF weights are provisioned separately, not included in this crate or binary. Follow the applicable model terms. llama.cpp, SQLite, and other transitive dependencies retain the licenses distributed with their source packages; geg does not claim ownership of them.
