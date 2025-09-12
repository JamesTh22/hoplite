
#![allow(dead_code)]

use crate::types::{Piece, PieceKind, Side, Move};
use crate::zobrist::Zobrist;

#[derive(Clone)]
pub struct Board {
    pub pieces: [Option<Piece>; 64],
    pub stm: Side,
    pub castle: u8,  // bit0 K, bit1 Q, bit2 k, bit3 q
    pub ep: Option<u8>,
    pub halfmove: u32,
    pub fullmove: u32,
    pub key: u64,
    pub(crate) zob: Zobrist,
}

impl Board {
    pub fn pawn_key(&self) -> u64 {
        // Use Zobrist table but only XOR pawns (both sides)
        let mut k = 0u64;
        for i in 0..64u8 {
            if let Some(p)=self.piece_at(i) {
                if matches!(p.kind, crate::types::PieceKind::Pawn) {
                    k ^= self.zob.psq[p.side as usize][p.kind as usize][i as usize];
                }
            }
        }
        k
    }

    pub fn new_start() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    pub fn from_fen(fen: &str) -> anyhow::Result<Self> {
        let zob = Zobrist::new();
        let mut pieces = [None; 64];
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() != 6 { anyhow::bail!("bad FEN"); }
        let mut idx: i32 = 56; // a8
        for c in parts[0].chars() {
            match c {
                '/' => idx -= 16,
                '1'..='8' => { idx += c as i32 - '0' as i32; }
                p => {
                    let side = if p.is_ascii_uppercase() { Side::White } else { Side::Black };
                    let kind = match p.to_ascii_lowercase() {
                        'p' => PieceKind::Pawn, 'n' => PieceKind::Knight, 'b' => PieceKind::Bishop,
                        'r' => PieceKind::Rook, 'q' => PieceKind::Queen, 'k' => PieceKind::King,
                        _ => anyhow::bail!("bad piece"),
                    };
                    pieces[idx as usize] = Some(Piece{side, kind});
                    idx += 1;
                }
            }
        }
        let stm = if parts[1] == "w" { Side::White } else { Side::Black };
        let mut castle = 0u8;
        if parts[2].contains('K') { castle |= 1; }
        if parts[2].contains('Q') { castle |= 2; }
        if parts[2].contains('k') { castle |= 4; }
        if parts[2].contains('q') { castle |= 8; }
        let ep = if parts[3] != "-" {
            let b = parts[3].as_bytes();
            let f = (b[0]-b'a') as i32;
            let r = (b[1]-b'1') as i32;
            Some((r*8+f) as u8)
        } else { None };
        let halfmove: u32 = parts[4].parse().unwrap_or(0);
        let fullmove: u32 = parts[5].parse().unwrap_or(1);
        let mut b = Self { pieces, stm, castle, ep, halfmove, fullmove, key:0, zob };
        b.recompute_key();
        Ok(b)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::new();
        for r in (0..8).rev() {
            let mut empty = 0;
            for f in 0..8 {
                let i = (r*8+f) as usize;
                match self.pieces[i] {
                    None => empty += 1,
                    Some(p) => {
                        if empty>0 { s.push_str(&empty.to_string()); empty=0; }
                        let c = match p.kind {
                            PieceKind::Pawn=>'p', PieceKind::Knight=>'n', PieceKind::Bishop=>'b',
                            PieceKind::Rook=>'r', PieceKind::Queen=>'q', PieceKind::King=>'k'
                        };
                        s.push(if p.side==Side::White { c.to_ascii_uppercase() } else { c });
                    }
                }
            }
            if empty>0 { s.push_str(&empty.to_string()); }
            if r>0 { s.push('/'); }
        }
        s.push(' ');
        s.push(if self.stm==Side::White {'w'} else {'b'});
        s.push(' ');
        let mut c=String::new();
        if self.castle&1!=0 { c.push('K'); }
        if self.castle&2!=0 { c.push('Q'); }
        if self.castle&4!=0 { c.push('k'); }
        if self.castle&8!=0 { c.push('q'); }
        if c.is_empty() { c.push('-'); }
        s.push_str(&c);
        s.push(' ');
        if let Some(ep)=self.ep {
            let f = (ep % 8) as u8; let r = (ep / 8) as u8;
            s.push((b'a'+f) as char); s.push((b'1'+r) as char);
        } else { s.push_str("-"); }
        s.push(' '); s.push_str(&self.halfmove.to_string());
        s.push(' '); s.push_str(&self.fullmove.to_string());
        s
    }

    #[inline]
    pub fn piece_at(&self, sq: u8) -> Option<Piece> { self.pieces[sq as usize] }

    fn recompute_key(&mut self) {
        let mut k=0u64;
        for i in 0..64 {
            if let Some(p)=self.pieces[i] { k ^= self.zob.piece_key(p, i as u8); }
        }
        if self.stm==Side::White { k ^= self.zob.stm; }
        self.key = k;
    }

    pub fn make_move(&mut self, mv: Move) -> Undo {
        let from = mv.from as usize; let to = mv.to as usize;
        let moving = self.pieces[from];
        let mut captured = self.pieces[to];
        let old_ep = self.ep;
        let old_half = self.halfmove;
        let old_full = self.fullmove;
        let old_castle = self.castle;

        // reset ep
        self.ep = None;
        // halfmove/fullmove update
        // reset on pawn move or capture
        if let Some(p)=moving {
            if matches!(p.kind, crate::types::PieceKind::Pawn) || captured.is_some() { self.halfmove = 0; } else { self.halfmove += 1; }
        } else { self.halfmove += 1; }
        if matches!(self.stm, crate::types::Side::Black) { self.fullmove += 1; }
        // special move flags
        let mut is_castle = false;
        let mut castle_rook_from: Option<usize> = None;
        let mut castle_rook_to: Option<usize> = None;
        let mut is_ep = false;
        let mut ep_captured_sq: Option<usize> = None;
        let side = self.stm;

        if let Some(p)=moving {
            // Castling: king moves two files from e to g/c (or e8 to g8/c8)
            if matches!(p.kind, PieceKind::King) {
                if (mv.from==4 && (mv.to==6 || mv.to==2) && side==Side::White)
                    || (mv.from==60 && (mv.to==62 || mv.to==58) && side==Side::Black) {
                    is_castle = true;
                    if mv.to == 6 { castle_rook_from = Some(7); castle_rook_to = Some(5); }   // white king-side
                    if mv.to == 2 { castle_rook_from = Some(0); castle_rook_to = Some(3); }   // white queen-side
                    if mv.to == 62 { castle_rook_from = Some(63); castle_rook_to = Some(61); } // black king-side
                    if mv.to == 58 { castle_rook_from = Some(56); castle_rook_to = Some(59); } // black queen-side
                }
                // King moved: clear castling rights
                match side {
                    Side::White => { self.castle &= !(1|2); }
                    Side::Black => { self.castle &= !(4|8); }
                }
            }
            // Pawn logic: set EP after double step, detect EP capture
            if matches!(p.kind, PieceKind::Pawn) {
                let from_rank = (mv.from / 8) as i32;
                let to_rank = (mv.to / 8) as i32;
                if (from_rank - to_rank).abs() == 2 {
                    let mid = ((from_rank + to_rank)/2) as u8 * 8 + (mv.from % 8);
                    self.ep = Some(mid);
                }
                if captured.is_none() && Some(mv.to)==old_ep {
                    is_ep = true;
                    let dir = if side==Side::White { -1 } else { 1 };
                    let cap_rank = (mv.to / 8) as i32 + dir;
                    let cap_file = (mv.to % 8) as i32;
                    let cap_sq = (cap_rank * 8 + cap_file) as usize;
                    ep_captured_sq = Some(cap_sq);
                    captured = self.pieces[cap_sq];
                }
            }
        }

        // Zobrist: remove moving piece from from-square and any captured from to-square
        if let Some(p)=moving { self.key ^= self.zob.piece_key(p, mv.from); }
        if let Some(c)=captured {
            if is_ep {
                self.key ^= self.zob.piece_key(c, ep_captured_sq.unwrap() as u8);
            } else {
                self.key ^= self.zob.piece_key(c, mv.to);
            }
        }

        // move piece
        self.pieces[to] = moving;
        self.pieces[from] = None;

        // handle en passant capture: remove the pawn behind
        if is_ep {
            if let Some(csq)=ep_captured_sq {
                self.pieces[csq] = None;
            }
        }

        // promotion
        if let Some(promo)=mv.promo {
            if let Some(mut p)=self.pieces[to] {
                self.key ^= self.zob.piece_key(p, mv.to);
                p.kind = promo;
                self.pieces[to] = Some(p);
                self.key ^= self.zob.piece_key(p, mv.to);
            }
        } else {
            if let Some(p)=self.pieces[to] {
                self.key ^= self.zob.piece_key(p, mv.to);
            }
        }

        // rook movement for castling
        if is_castle {
            if let (Some(rf), Some(rt)) = (castle_rook_from, castle_rook_to) {
                if let Some(rp)=self.pieces[rf] {
                    self.key ^= self.zob.piece_key(rp, rf as u8);
                    self.pieces[rt] = self.pieces[rf];
                    self.pieces[rf] = None;
                    if let Some(rp2)=self.pieces[rt] {
                        self.key ^= self.zob.piece_key(rp2, rt as u8);
                    }
                }
            }
        }

        // update castling rights when rooks move or are captured
        match mv.from {
            0 => { self.castle &= !2; }, 7 => { self.castle &= !1; },
            56 => { self.castle &= !8; }, 63 => { self.castle &= !4; },
            _ => {}
        }
        match mv.to {
            0 => { self.castle &= !2; }, 7 => { self.castle &= !1; },
            56 => { self.castle &= !8; }, 63 => { self.castle &= !4; },
            _ => {}
        }

        // side to move
        self.key ^= self.zob.stm;
        self.stm = self.stm.flip();

        Undo {
            captured,
            from: mv.from, to: mv.to, promo: mv.promo,
            old_ep, old_half, old_full, old_castle,
            ep_captured_sq,
            castle_rook_from, castle_rook_to,
            is_castle, is_ep,
        }
    }

    pub fn unmake_move(&mut self, mv: Move, u: Undo) {
        // side to move back
        self.stm = self.stm.flip();
        self.key ^= self.zob.stm;

        // undo rook from castling first
        if u.is_castle {
            if let (Some(rf), Some(rt)) = (u.castle_rook_from, u.castle_rook_to) {
                if let Some(rp)=self.pieces[rt] {
                    self.key ^= self.zob.piece_key(rp, rt as u8);
                    self.pieces[rf] = self.pieces[rt];
                    self.pieces[rt] = None;
                    if let Some(rp2)=self.pieces[rf] {
                        self.key ^= self.zob.piece_key(rp2, rf as u8);
                    }
                }
            }
        }

        // undo piece move
        let from = mv.from as usize; let to = mv.to as usize;
        let moving = self.pieces[to];
        if let Some(mut p)=moving {
            self.key ^= self.zob.piece_key(p, mv.to);
            if u.promo.is_some() { p.kind = PieceKind::Pawn; }
            self.pieces[from] = Some(p);
            self.pieces[to] = u.captured;
            self.key ^= self.zob.piece_key(p, mv.from);
            if let Some(c)=u.captured {
                let cap_sq = if let Some(csq)=u.ep_captured_sq { csq as u8 } else { mv.to };
                self.key ^= self.zob.piece_key(c, cap_sq);
            }
        }

        // restore ep captured pawn if any
        if let Some(csq)=u.ep_captured_sq {
            self.pieces[csq] = u.captured;
            self.pieces[mv.to as usize] = None;
        }

        self.ep = u.old_ep;
        self.halfmove = u.old_half;
        self.fullmove = u.old_full;
        self.castle = u.old_castle;
    }

    pub fn in_check(&self, side: Side) -> bool {
        let king_sq = (0u8..64).find(|&i| {
            if let Some(p)=self.piece_at(i) {
                p.side==side && matches!(p.kind, PieceKind::King)
            } else { false }
        });
        let Some(ksq)=king_sq else { return false; };
        crate::movegen::square_attacked(self, ksq, side.flip())
    }
}

pub struct Undo {
    pub captured: Option<Piece>,
    pub from: u8, pub to: u8, pub promo: Option<PieceKind>,
    pub old_ep: Option<u8>,
    pub old_half: u32,
    pub old_full: u32,
    pub old_castle: u8,
    pub ep_captured_sq: Option<usize>,
    pub castle_rook_from: Option<usize>,
    pub castle_rook_to: Option<usize>,
    pub is_castle: bool,
    pub is_ep: bool,
}
