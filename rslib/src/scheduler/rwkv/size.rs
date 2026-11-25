// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

//! Hayaki: the three RWKV models a collection can choose between.

use fushi_rwkv::Model;
use fushi_rwkv::StatePrecision;
use serde::Deserialize;
use serde::Serialize;

/// The collection config key choosing the model, so the choice syncs.
/// Absent or unknown means small.
pub const SIZE_CONFIG_KEY: &str = "hayakiRwkvModel";

/// The models are JSchoreels/anki-rwkv's. Their weights are for personal and
/// research use; see WEIGHTS-NOTICE.md in that repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    Big,
    #[default]
    Small,
    Tiny,
}

/// Where the app gets a model's weights, and how it checks them.
#[derive(Debug, PartialEq, Eq)]
pub struct WeightsFile {
    pub file_name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub len: usize,
}

impl WeightsFile {
    pub fn is_valid(&self, bytes: &[u8]) -> bool {
        use sha2::Digest;
        bytes.len() == self.len && hex::encode(sha2::Sha256::digest(bytes)) == self.sha256
    }
}

/// sm99, the 2.76M-parameter network trained for the immediate head on
/// show-time intervals. Only its f32 file loads as the big network.
const BIG: WeightsFile = WeightsFile {
    file_name: "rwkv-sm99-weights.f32.bin",
    url: "https://raw.githubusercontent.com/JSchoreels/anki/aecc5b09dde64cd14d7e0a135abe6c542afe13cf/qt/aqt/rwkv_inference/sm99/weights.f32.bin",
    sha256: "682ba1ff207c51af7de5e2abeb25593362aa706b292cbcc338a1591fad0d935c",
    len: 11_082_637,
};

/// sm56, the small network Hayaki shipped first.
const SMALL: WeightsFile = WeightsFile {
    file_name: "rwkv-sm56-weights.int8.bin",
    url: "https://raw.githubusercontent.com/JSchoreels/anki/001440b03395f24ffc1714b18ad3d9d4d89d2067/qt/aqt/rwkv_inference/sm56/weights.int8.bin",
    sha256: "479b3e2bc6a1c900ba8a4a45d629965c6df2a20cc8ff294583f5aeaeea3b4fdd",
    len: 436_391,
};

/// sm130io, the six-step network the upstream fork calls Tiny, trained for
/// the immediate head on show-time intervals.
const TINY: WeightsFile = WeightsFile {
    file_name: "rwkv-sm130io-weights.int8.bin",
    url: "https://raw.githubusercontent.com/JSchoreels/anki/aecc5b09dde64cd14d7e0a135abe6c542afe13cf/qt/aqt/rwkv_inference/sm130io/weights.int8.bin",
    sha256: "24b39f61129bd9eb45156f45a32e749e1a0cf8d75e9271c2ed7038f9d6265771",
    len: 238_585,
};

impl ModelSize {
    pub const ALL: [ModelSize; 3] = [ModelSize::Big, ModelSize::Small, ModelSize::Tiny];

    pub fn weights(self) -> &'static WeightsFile {
        match self {
            ModelSize::Big => &BIG,
            ModelSize::Small => &SMALL,
            ModelSize::Tiny => &TINY,
        }
    }

    /// Card and note state precision. Small keeps the int4 state it shipped
    /// with; the big network refuses int4, and tiny has no int4 measurement.
    pub fn precision(self) -> StatePrecision {
        match self {
            ModelSize::Small => StatePrecision::Int4,
            ModelSize::Big | ModelSize::Tiny => StatePrecision::Int8,
        }
    }
}

/// A parsed network and the size it was loaded as.
#[derive(Clone)]
pub struct LoadedModel {
    pub size: ModelSize,
    pub model: Model,
}

impl LoadedModel {
    /// Parses weights already checked against `size`'s file.
    pub fn from_bytes(size: ModelSize, bytes: &[u8]) -> std::io::Result<Self> {
        Ok(Self {
            size,
            model: Model::from_bytes(bytes)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_config_value_is_the_lowercase_name() {
        for (size, name) in ModelSize::ALL.iter().zip(["big", "small", "tiny"]) {
            assert_eq!(serde_json::to_string(size).unwrap(), format!("\"{name}\""));
        }
        assert_eq!(ModelSize::default(), ModelSize::Small);
    }

    #[test]
    fn every_size_has_its_own_weights_file() {
        let names: std::collections::HashSet<_> = ModelSize::ALL
            .iter()
            .map(|s| s.weights().file_name)
            .collect();
        assert_eq!(names.len(), ModelSize::ALL.len());
        assert!(!SMALL.is_valid(b"not weights"));
        assert!(!SMALL.is_valid(&vec![0u8; SMALL.len]));
    }
}
