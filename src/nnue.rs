use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use binread::BinRead;
use nnue::stockfish::halfkp::{
    scale_nn_to_centipawns, SfHalfKpFullModel, SfHalfKpModel, SfHalfKpState,
};
use nnue::{Color as NnueColor, Piece as NnuePiece, Square as NnueSquare};

use crate::board::{Board, Undo};
use crate::types::{Move, PieceKind, Side};

/// Representation of a Stockfish-style NNUE network backed by the `nnue`
/// library.  The loader keeps track of the header metadata together with the
/// parsed model so that the rest of the engine can evaluate positions without
/// keeping any global mutable state.
#[derive(Clone)]
pub struct NnueNetwork {
    pub version: u32,
    pub network_hash: u32,
    pub architecture_hash: u32,
    pub description: String,
    pub path: Option<PathBuf>,
    model: SfHalfKpModel,
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

        let mut cursor = Cursor::new(&data);
        let full_model = SfHalfKpFullModel::read(&mut cursor)
            .with_context(|| "failed to parse NNUE payload as a HalfKP network")?;

        Ok(Self {
            version,
            network_hash,
            architecture_hash,
            description,
            path,
            model: full_model.model,
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

    pub fn evaluate(&self, board: &Board, acc: &mut Accumulator) -> Option<i16> {
        if acc.key != board.key {
            acc.key = board.key;
            acc.value = None;
        }

        if let Some(score) = acc.value {
            return Some(score);
        }

        let eval = self.forward(board)?;
        acc.value = Some(eval);
        Some(eval)
    }

    fn forward(&self, board: &Board) -> Option<i16> {
        let (white_king_sq, black_king_sq) = find_kings(board)?;

        let white_king = square_to_nnue(white_king_sq);
        let black_king = square_to_nnue(black_king_sq);

        let mut state = self.model.new_state(white_king, black_king);
        populate_state(board, &mut state);

        let stm = side_to_nnue(board.stm);
        let raw = state.activate(stm)[0];
        let centipawns = scale_nn_to_centipawns(raw);
        let clamped = centipawns.clamp(-32000, 32000);
        Some(clamped as i16)
    }
}

/// Incremental accumulator for NNUE evaluation.  The current implementation is
/// intentionally conservative: we rebuild the accumulator from scratch whenever
/// the board changes.  This keeps the API surface identical to an actual
/// incremental update routine and allows the search to maintain a state stack
/// without special casing.  The accumulator also caches the most recent
/// evaluation so that repeated probes of the same board avoid re-running the
/// forward pass.
#[derive(Clone, Debug, Default)]
pub struct Accumulator {
    key: u64,
    value: Option<i16>,
}

impl Accumulator {
    pub fn from_board(board: &Board) -> Self {
        Self {
            key: board.key,
            value: None,
        }
    }

    pub fn update(&mut self, board: &Board, _mv: Move, _undo: &Undo) {
        self.key = board.key;
        self.value = None;
    }

    pub fn key(&self) -> u64 {
        self.key
    }
}

/// Convenience helper used in tests and UCI plumbing.
pub fn load_nnue(path: impl AsRef<Path>) -> Result<NnueNetwork> {
    NnueNetwork::load_from_file(path)
}

fn find_kings(board: &Board) -> Option<(u8, u8)> {
    let mut white = None;
    let mut black = None;

    for sq in 0..64u8 {
        if let Some(piece) = board.piece_at(sq) {
            match (piece.side, piece.kind) {
                (Side::White, PieceKind::King) => white = Some(sq),
                (Side::Black, PieceKind::King) => black = Some(sq),
                _ => {}
            }
        }
    }

    match (white, black) {
        (Some(w), Some(b)) => Some((w, b)),
        _ => None,
    }
}

fn populate_state(board: &Board, state: &mut SfHalfKpState<'_>) {
    for sq in 0..64u8 {
        if let Some(piece) = board.piece_at(sq) {
            if let Some(nnue_piece) = piece_kind_to_nnue(piece.kind) {
                let square = square_to_nnue(sq);
                let piece_color = side_to_nnue(piece.side);
                for color in NnueColor::ALL.iter().copied() {
                    state.add(color, nnue_piece, piece_color, square);
                }
            }
        }
    }
}

fn piece_kind_to_nnue(kind: PieceKind) -> Option<NnuePiece> {
    match kind {
        PieceKind::Pawn => Some(NnuePiece::Pawn),
        PieceKind::Knight => Some(NnuePiece::Knight),
        PieceKind::Bishop => Some(NnuePiece::Bishop),
        PieceKind::Rook => Some(NnuePiece::Rook),
        PieceKind::Queen => Some(NnuePiece::Queen),
        PieceKind::King => None,
    }
}

fn side_to_nnue(side: Side) -> NnueColor {
    match side {
        Side::White => NnueColor::White,
        Side::Black => NnueColor::Black,
    }
}

fn square_to_nnue(sq: u8) -> NnueSquare {
    NnueSquare::from_index(sq as usize)
}
