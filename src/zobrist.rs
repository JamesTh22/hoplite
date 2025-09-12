
use rand::{RngCore, SeedableRng};
use rand::rngs::StdRng;
use crate::types::{Side, PieceKind, Piece};

#[derive(Clone)]
pub struct Zobrist {
    pub psq: [[[u64; 64]; 6]; 2],
    pub stm: u64,
}

impl Zobrist {
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE_BABE_F005_u64);
        let mut psq = [[[0u64; 64]; 6]; 2];
        for side in 0..2 {
            for kind in 0..6 {
                for sq in 0..64 {
                    psq[side][kind][sq] = rng.next_u64();
                }
            }
        }
        let stm = rng.next_u64();
        Self { psq, stm }
    }
    #[inline]
    pub fn piece_key(&self, p: Piece, sq: u8) -> u64 {
        let s = if matches!(p.side, Side::White) {0} else {1};
        let k = match p.kind {
            PieceKind::Pawn => 0, PieceKind::Knight => 1, PieceKind::Bishop => 2,
            PieceKind::Rook => 3, PieceKind::Queen => 4, PieceKind::King => 5,
        };
        self.psq[s][k][sq as usize]
    }
}
