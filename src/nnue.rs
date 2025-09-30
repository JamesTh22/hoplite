use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::board::{Board, Undo};
use crate::types::Move;

/// A minimal representation of a Stockfish-style NNUE network.
///
/// The implementation keeps the raw payload of the `.nnue` file together with
/// the header metadata.  The engine does not attempt to interpret the network
/// weights yet, but the metadata is validated so that obviously corrupt files
/// are rejected.  This is sufficient to provide the higher-level incremental
/// accumulator plumbing required by the search code.
#[derive(Clone, Debug)]
pub struct NnueNetwork {
    pub version: u32,
    pub network_hash: u32,
    pub architecture_hash: u32,
    pub description: String,
    pub path: Option<PathBuf>,
    raw: Vec<u8>,
}

impl NnueNetwork {
    /// Load a NNUE file from disk and perform some light sanity checking on the
    /// header.  The Stockfish reference format starts with a four byte version
    /// (currently `0x7AF32F20`), followed by the network hash, an architecture
    /// hash and a 256 byte textual description.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)
            .with_context(|| format!("failed to open NNUE file `{}`", path_ref.display()))?;
        let mut reader = BufReader::new(file);
        let mut raw = Vec::new();
        reader
            .read_to_end(&mut raw)
            .with_context(|| format!("failed to read NNUE file `{}`", path_ref.display()))?;
        Self::from_bytes(raw, Some(path_ref.to_path_buf()))
    }

    fn from_bytes(data: Vec<u8>, path: Option<PathBuf>) -> Result<Self> {
        if data.len() < 4 * 3 + 256 {
            bail!("NNUE file too small ({} bytes)", data.len());
        }

        let version = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let network_hash = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let architecture_hash = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let description_bytes = &data[12..12 + 256];
        let description = description_bytes
            .iter()
            .take_while(|b| **b != 0)
            .map(|b| *b as char)
            .collect::<String>();

        if description.trim().is_empty() {
            bail!("NNUE description string is empty");
        }

        if version == 0 {
            bail!("NNUE file reports unsupported version 0");
        }

        if data.len() <= 12 + 256 + 4 {
            bail!("NNUE file is missing layer data");
        }

        Ok(Self {
            version,
            network_hash,
            architecture_hash,
            description,
            path,
            raw: data,
        })
    }

    /// Simple helper for printing in logs.
    pub fn summary(&self) -> String {
        format!(
            "version=0x{version:08x} net_hash=0x{hash:08x} arch=0x{arch:08x} desc='{desc}'",
            version = self.version,
            hash = self.network_hash,
            arch = self.architecture_hash,
            desc = self.description.trim()
        )
    }

    /// Placeholder evaluation routine.  The NNUE weights are not interpreted
    /// yet; instead we merely return `0` and rely on the PSQT fall-back to
    /// supply the actual evaluation.  The accumulator plumbing still relies on
    /// this method so we keep it available for future use.
    pub fn evaluate(&self, _accumulator: &Accumulator) -> i16 {
        0
    }
}

/// Incremental accumulator for NNUE evaluation.  The current implementation is
/// intentionally conservative: we rebuild the accumulator from scratch whenever
/// the board changes.  This keeps the API surface identical to an actual
/// incremental update routine and allows the search to maintain a state stack
/// without special casing.
#[derive(Clone, Debug, Default)]
pub struct Accumulator {
    key: u64,
}

impl Accumulator {
    pub fn from_board(board: &Board) -> Self {
        Self { key: board.key }
    }

    pub fn update(&mut self, board: &Board, _mv: Move, _undo: &Undo) {
        *self = Self::from_board(board);
    }

    pub fn key(&self) -> u64 {
        self.key
    }
}

/// Convenience helper used in tests and UCI plumbing.
pub fn load_nnue(path: impl AsRef<Path>) -> Result<NnueNetwork> {
    NnueNetwork::load_from_file(path)
}
