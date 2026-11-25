// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Hayaki: RWKV scheduling status and preparation (ADR-0009).

use std::path::Path;
use std::time::Duration;

use anki_proto::zz_hayaki::rwkv_status_response::State;
use anki_proto::zz_hayaki::PrepareRwkvRequest;
use anki_proto::zz_hayaki::RwkvModel;
use anki_proto::zz_hayaki::RwkvStatusResponse;
use anki_proto::zz_hayaki::RwkvWeights;
use anki_proto::zz_hayaki::SetRwkvModelRequest;

use super::Backend;
use crate::prelude::*;
use crate::scheduler::rwkv::set_loaded_model;
use crate::scheduler::rwkv::LoadedModel;
use crate::scheduler::rwkv::ModelSize;
use crate::scheduler::rwkv::Status;
use crate::services::BackendHayakiService;

fn proto_model(size: ModelSize) -> RwkvModel {
    match size {
        ModelSize::Big => RwkvModel::Big,
        ModelSize::Small => RwkvModel::Small,
        ModelSize::Tiny => RwkvModel::Tiny,
    }
}

fn model_size(model: RwkvModel) -> ModelSize {
    match model {
        RwkvModel::Big => ModelSize::Big,
        RwkvModel::Small => ModelSize::Small,
        RwkvModel::Tiny => ModelSize::Tiny,
    }
}

fn status_response((enabled, size, status): (bool, ModelSize, Status)) -> RwkvStatusResponse {
    let weights = size.weights();
    let mut out = RwkvStatusResponse {
        enabled,
        weights: Some(RwkvWeights {
            file_name: weights.file_name.into(),
            url: weights.url.into(),
            sha256: weights.sha256.into(),
            size: weights.len as u64,
        }),
        ..Default::default()
    };
    out.set_model(proto_model(size));
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

impl Backend {
    /// `size`'s weights from `dir`, downloading them first if they are
    /// missing or damaged.
    fn rwkv_weights(&self, dir: &Path, size: ModelSize) -> Result<Vec<u8>> {
        let weights = size.weights();
        let path = dir.join(weights.file_name);
        if let Ok(bytes) = std::fs::read(&path) {
            if weights.is_valid(&bytes) {
                return Ok(bytes);
            }
        }
        let bytes = self.runtime_handle().block_on(async {
            let response = self
                .web_client()
                .get(weights.url)
                .timeout(Duration::from_secs(600))
                .send()
                .await?
                .error_for_status()?;
            response.bytes().await
        });
        let bytes = bytes.or_invalid("RWKV weights download")?;
        if !weights.is_valid(&bytes) {
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
        self.with_col(|col| Ok(status_response(col.rwkv_status())))
    }

    fn prepare_rwkv(&self, input: PrepareRwkvRequest) -> Result<RwkvStatusResponse> {
        // The download runs without holding the collection.
        let size = self.with_col(|col| Ok(col.rwkv_size()))?;
        let bytes = self.rwkv_weights(Path::new(&input.weights_dir), size)?;
        let model = LoadedModel::from_bytes(size, &bytes).or_invalid("RWKV weights")?;
        self.with_col(|col| {
            // The choice may have changed during the download.
            if col.rwkv_size() == size {
                set_loaded_model(model.clone());
            }
            col.rwkv_prepare(model)?;
            Ok(status_response(col.rwkv_status()))
        })
    }

    fn set_rwkv_model(&self, input: SetRwkvModelRequest) -> Result<RwkvStatusResponse> {
        // An unknown value from a newer app must not sync as small.
        let Ok(model) = RwkvModel::try_from(input.model) else {
            invalid_input!("unknown RWKV model {}", input.model);
        };
        self.with_col(|col| {
            col.rwkv_set_size(model_size(model))?;
            Ok(status_response(col.rwkv_status()))
        })
    }
}

#[cfg(test)]
mod tests {
    use fushi_rwkv::ModelKind;
    use fushi_rwkv::RwkvRuntime;

    use super::*;

    #[test]
    fn every_size_maps_to_its_own_proto_model() {
        for size in ModelSize::ALL {
            assert_eq!(model_size(proto_model(size)), size);
        }
        assert_eq!(model_size(RwkvModel::default()), ModelSize::default());
    }

    #[test]
    fn downloads_and_verifies_the_pinned_weights() -> Result<()> {
        if std::env::var("ONLINE_TESTS").is_err() {
            println!("test disabled; ONLINE_TESTS not set");
            return Ok(());
        }
        let dir = tempfile::tempdir()?;
        let backend = Backend::new(I18n::template_only(), false);
        for size in ModelSize::ALL {
            let bytes = backend.rwkv_weights(dir.path(), size)?;
            assert!(size.weights().is_valid(&bytes));
            let loaded = LoadedModel::from_bytes(size, &bytes)?;
            let kind = if size == ModelSize::Big {
                ModelKind::Large
            } else {
                ModelKind::Small
            };
            assert_eq!(loaded.model.kind(), kind);
            assert!(RwkvRuntime::new(&loaded.model, size.precision(), 0).is_ok());
            // The second call reads the file instead of downloading.
            assert_eq!(backend.rwkv_weights(dir.path(), size)?, bytes);
        }
        Ok(())
    }
}
