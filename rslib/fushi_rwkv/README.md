# fushi_rwkv (vendored)

Hayaki's Engine patch uses this crate to run the RWKV review model. It is a copy of
`native/fushi_rwkv/src` from [Fushi](https://github.com/hajisensai/Fushi) at commit
`07940eaef04a871fb3d89457b320abfe6ae2b752`, licensed AGPL-3.0-or-later.

Changes from that commit:

- `src/snapshot_tests.rs` is removed, because it needs model weights on disk.
- `Cargo.toml` is new. It builds an rlib only and takes `rayon` from the Anki workspace.

Fushi's own provenance notes for the inference core are in `UPSTREAM-FUSHI.md`.
To update, copy `src` again from a newer Fushi commit, repeat the changes above,
and record the commit here.
