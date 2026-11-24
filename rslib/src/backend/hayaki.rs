// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Hayaki: RWKV scheduling status and preparation (ADR-0009).

use std::path::Path;
use std::time::Duration;

use anki_proto::hayaki::rwkv_status_response::State;
use anki_proto::hayaki::PrepareRwkvRequest;
use anki_proto::hayaki::RwkvStatusResponse;
use fushi_rwkv::Model;
use sha2::Digest;

use super::Backend;
use crate::prelude::*;
use crate::scheduler::rwkv;
use crate::scheduler::rwkv::Status;
use crate::services::BackendHayakiService;

fn status_response(enabled: bool, status: Status) -> RwkvStatusResponse {
    let mut out = RwkvStatusResponse {
        enabled,
        ..Default::default()
    };
    match status {
        Status::NotPrepared => out.set_state(State::NotPrepared),
        Status::Preparing { done, total } => {
            out.set_state(State::Preparing);
            out.done = done;
            out.total = total;
        }
        Status::Ready => out.set_state(State::Ready),
        Status::Failed(error) => {
            out.set_state(State::Failed);
            out.error = error;
        }
    }
    out
}

fn weights_are_valid(bytes: &[u8]) -> bool {
    bytes.len() == rwkv::WEIGHTS_LEN
        && hex::encode(sha2::Sha256::digest(bytes)) == rwkv::WEIGHTS_SHA256
}

impl Backend {
    /// The weights from `dir`, downloading them first if they are missing or
    /// damaged.
    fn rwkv_weights(&self, dir: &Path) -> Result<Vec<u8>> {
        let path = dir.join(rwkv::WEIGHTS_FILE_NAME);
        if let Ok(bytes) = std::fs::read(&path) {
            if weights_are_valid(&bytes) {
                return Ok(bytes);
            }
        }
        let bytes = self.runtime_handle().block_on(async {
            let response = self
                .web_client()
                .get(rwkv::WEIGHTS_URL)
                .timeout(Duration::from_secs(120))
                .send()
                .await?
                .error_for_status()?;
            response.bytes().await
        });
        let bytes = bytes.or_invalid("RWKV weights download")?;
        if !weights_are_valid(&bytes) {
            invalid_input!("RWKV weights download: size or SHA-256 does not match");
        }
        std::fs::create_dir_all(dir)?;
        let tmp = path.with_extension("part");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &path)?;
        Ok(bytes.to_vec())
    }
}

impl BackendHayakiService for Backend {
    fn rwkv_status(&self) -> Result<RwkvStatusResponse> {
        self.with_col(|col| {
            let (enabled, status) = col.rwkv_status();
            Ok(status_response(enabled, status))
        })
    }

    fn prepare_rwkv(&self, input: PrepareRwkvRequest) -> Result<RwkvStatusResponse> {
        let bytes = self.rwkv_weights(Path::new(&input.weights_dir))?;
        let model = Model::from_bytes(&bytes).or_invalid("RWKV weights")?;
        rwkv::set_loaded_model(model.clone());
        self.with_col(|col| {
            col.rwkv_prepare(model)?;
            let (enabled, status) = col.rwkv_status();
            Ok(status_response(enabled, status))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_weights_of_the_wrong_size_or_hash() {
        assert!(!weights_are_valid(b"not weights"));
        assert!(!weights_are_valid(&vec![0u8; rwkv::WEIGHTS_LEN]));
    }

    #[test]
    fn downloads_and_verifies_the_pinned_weights() -> Result<()> {
        if std::env::var("ONLINE_TESTS").is_err() {
            println!("test disabled; ONLINE_TESTS not set");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let backend = Backend::new(I18n::template_only(), false);
        let bytes = backend.rwkv_weights(dir.path())?;
        assert!(weights_are_valid(&bytes));
        assert!(Model::from_bytes(&bytes).is_ok());
        // The second call reads the file instead of downloading.
        assert_eq!(backend.rwkv_weights(dir.path())?, bytes);
        Ok(())
    }
}
