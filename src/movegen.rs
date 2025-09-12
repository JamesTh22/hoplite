use crate::board::Board;
use crate::types::{sq, Move, PieceKind, Side};

use lazy_static::lazy_static;

lazy_static! {
    // Pre-computed knight move targets for each square as bitboards
    static ref KNIGHT_TARGETS: [u64; 64] = {
        let mut arr = [0u64; 64];
        for s in 0u8..64 {
            let f = (s % 8) as i32;
            let r = (s / 8) as i32;
            for (df, dr) in [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)] {
                if let Some(t) = sq(f+df, r+dr) {
                    arr[s as usize] |= 1u64 << t;
                }
            }
        }
        arr
    };

    // Pre-computed king move targets for each square as bitboards
    static ref KING_TARGETS: [u64; 64] = {
        let mut arr = [0u64; 64];
        for s in 0u8..64 {
            let f = (s % 8) as i32;
            let r = (s / 8) as i32;
            for df in -1..=1 {
                for dr in -1..=1 {
                    if df == 0 && dr == 0 { continue; }
                    if let Some(t) = sq(f+df, r+dr) {
                        arr[s as usize] |= 1u64 << t;
                    }
                }
            }
        }
        arr
    };
}

#[inline]
fn add(moves: &mut Vec<Move>, from: u8, to: u8) {
    moves.push(Move {
        from,
        to,
        promo: None,
    });
}

pub fn legal_moves(b: &Board) -> Vec<Move> {
    let mut ms = Vec::with_capacity(64);
    pseudo_legal(b, &mut ms);
    // filter illegal (leaves king in check)
    let mut legal = Vec::with_capacity(ms.len());
    for m in ms.into_iter() {
        let mut bb = b.clone();
        let u = bb.make_move(m);
        if !bb.in_check(b.stm) {
            legal.push(m);
        }
        bb.unmake_move(m, u);
    }
    legal
}

pub fn pseudo_legal(b: &Board, out: &mut Vec<Move>) {
    let side = b.stm;
    for sqi in 0u8..64 {
        if let Some(p) = b.piece_at(sqi) {
            if p.side != side {
                continue;
            }
            match p.kind {
                PieceKind::Pawn => pawn_moves(b, sqi, p.side, out),
                PieceKind::Knight => knight_moves(b, sqi, p.side, out),
                PieceKind::Bishop => {
                    slider_moves(b, sqi, p.side, out, &[(1, 1), (1, -1), (-1, 1), (-1, -1)])
                }
                PieceKind::Rook => {
                    slider_moves(b, sqi, p.side, out, &[(1, 0), (-1, 0), (0, 1), (0, -1)])
                }
                PieceKind::Queen => slider_moves(
                    b,
                    sqi,
                    p.side,
                    out,
                    &[
                        (1, 1),
                        (1, -1),
                        (-1, 1),
                        (-1, -1),
                        (1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1),
                    ],
                ),
                PieceKind::King => king_moves(b, sqi, p.side, out),
            }
        }
    }
    // Castling
    castling_moves(b, side, out);
}

fn pawn_moves(b: &Board, s: u8, side: Side, out: &mut Vec<Move>) {
    let file = (s % 8) as i32;
    let rank = (s / 8) as i32;
    let dir = if side == Side::White { 1 } else { -1 };
    // single push
    if let Some(t) = sq(file, rank + dir) {
        if b.piece_at(t).is_none() {
            // promotion?
            if (side == Side::White && (rank + dir) == 7)
                || (side == Side::Black && (rank + dir) == 0)
            {
                for pk in [
                    PieceKind::Queen,
                    PieceKind::Rook,
                    PieceKind::Bishop,
                    PieceKind::Knight,
                ] {
                    out.push(Move {
                        from: s,
                        to: t,
                        promo: Some(pk),
                    });
                }
            } else {
                add(out, s, t);
                // double push
                let start_rank = if side == Side::White { 1 } else { 6 };
                if rank == start_rank {
                    if let Some(t2) = sq(file, rank + 2 * dir) {
                        if b.piece_at(t2).is_none() {
                            add(out, s, t2);
                        }
                    }
                }
            }
        }
    }
    // captures (including en passant)
    for df in [-1, 1] {
        if let Some(t) = sq(file + df, rank + dir) {
            // normal capture
            if let Some(p) = b.piece_at(t) {
                if p.side != side {
                    if (side == Side::White && (rank + dir) == 7)
                        || (side == Side::Black && (rank + dir) == 0)
                    {
                        for pk in [
                            PieceKind::Queen,
                            PieceKind::Rook,
                            PieceKind::Bishop,
                            PieceKind::Knight,
                        ] {
                            out.push(Move {
                                from: s,
                                to: t,
                                promo: Some(pk),
                            });
                        }
                    } else {
                        add(out, s, t);
                    }
                }
            } else if Some(t) == b.ep {
                // en passant target square
                add(out, s, t);
            }
        }
    }
}

fn knight_moves(b: &Board, s: u8, side: Side, out: &mut Vec<Move>) {
    let mut bb = KNIGHT_TARGETS[s as usize];
    while bb != 0 {
        let t = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        match b.piece_at(t) {
            None => add(out, s, t),
            Some(p) if p.side != side => add(out, s, t),
            _ => {}
        }
    }
}

fn king_moves(b: &Board, s: u8, side: Side, out: &mut Vec<Move>) {
    let mut bb = KING_TARGETS[s as usize];
    while bb != 0 {
        let t = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        match b.piece_at(t) {
            None => add(out, s, t),
            Some(p) if p.side != side => add(out, s, t),
            _ => {}
        }
    }
}

fn castling_moves(b: &Board, side: Side, out: &mut Vec<Move>) {
    // squares: white king e1 (4), rooks h1 (7), a1 (0); black king e8 (60), rooks h8 (63), a8 (56)
    let (_rank, e, f, g, d, c, bfile, h, a) = match side {
        Side::White => (0u8, 4u8, 5u8, 6u8, 3u8, 2u8, 1u8, 7u8, 0u8),
        Side::Black => (7u8, 60u8, 61u8, 62u8, 59u8, 58u8, 57u8, 63u8, 56u8),
    };
    // check king presence
    if let Some(k) = b.piece_at(e) {
        if k.side == side && matches!(k.kind, PieceKind::King) {
            // king-side
            let k_right = match side {
                Side::White => b.castle & 1 != 0,
                Side::Black => b.castle & 4 != 0,
            };
            if k_right
                && b.piece_at(f).is_none()
                && b.piece_at(g).is_none()
                && !crate::movegen::square_attacked(b, e, side.flip())
                && !crate::movegen::square_attacked(b, f, side.flip())
                && !crate::movegen::square_attacked(b, g, side.flip())
                && matches!(b.piece_at(h), Some(p) if p.side==side && matches!(p.kind, PieceKind::Rook))
            {
                out.push(Move {
                    from: e,
                    to: g,
                    promo: None,
                });
            }
            // queen-side
            let q_right = match side {
                Side::White => b.castle & 2 != 0,
                Side::Black => b.castle & 8 != 0,
            };
            if q_right
                && b.piece_at(d).is_none()
                && b.piece_at(c).is_none()
                && b.piece_at(bfile).is_none()
                && !crate::movegen::square_attacked(b, e, side.flip())
                && !crate::movegen::square_attacked(b, d, side.flip())
                && !crate::movegen::square_attacked(b, c, side.flip())
                && matches!(b.piece_at(a), Some(p) if p.side==side && matches!(p.kind, PieceKind::Rook))
            {
                out.push(Move {
                    from: e,
                    to: c,
                    promo: None,
                });
            }
        }
    }
}

fn slider_moves(b: &Board, s: u8, side: Side, out: &mut Vec<Move>, dirs: &[(i32, i32)]) {
    let f = (s % 8) as i32;
    let r = (s / 8) as i32;
    for (df, dr) in dirs {
        let mut nf = f + df;
        let mut nr = r + dr;
        loop {
            if let Some(t) = sq(nf, nr) {
                if let Some(p) = b.piece_at(t) {
                    if p.side != side {
                        add(out, s, t);
                    }
                    break;
                } else {
                    add(out, s, t);
                }
                nf += df;
                nr += dr;
            } else {
                break;
            }
        }
    }
}

pub fn square_attacked(b: &Board, s: u8, by: Side) -> bool {
    let f = (s % 8) as i32;
    let r = (s / 8) as i32;
    // pawn attacks
    let dir = if by == Side::White { -1 } else { 1 };
    for df in [-1, 1] {
        if let Some(t) = sq(f + df, r + dir) {
            if let Some(p) = b.piece_at(t) {
                if p.side == by && matches!(p.kind, PieceKind::Pawn) {
                    return true;
                }
            }
        }
    }
    // knight
    let mut nbb = KNIGHT_TARGETS[s as usize];
    while nbb != 0 {
        let t = nbb.trailing_zeros() as u8;
        nbb &= nbb - 1;
        if let Some(p) = b.piece_at(t) {
            if p.side == by && matches!(p.kind, PieceKind::Knight) {
                return true;
            }
        }
    }
    // king
    let mut kbb = KING_TARGETS[s as usize];
    while kbb != 0 {
        let t = kbb.trailing_zeros() as u8;
        kbb &= kbb - 1;
        if let Some(p) = b.piece_at(t) {
            if p.side == by && matches!(p.kind, PieceKind::King) {
                return true;
            }
        }
    }
    // sliders
    if slider_attack(b, s, by, &[(1, 1), (1, -1), (-1, 1), (-1, -1)], true) {
        return true;
    }
    if slider_attack(b, s, by, &[(1, 0), (-1, 0), (0, 1), (0, -1)], false) {
        return true;
    }
    false
}

fn slider_attack(b: &Board, s: u8, by: Side, dirs: &[(i32, i32)], diag: bool) -> bool {
    let f = (s % 8) as i32;
    let r = (s / 8) as i32;
    for (df, dr) in dirs {
        let mut nf = f + df;
        let mut nr = r + dr;
        loop {
            if let Some(t) = sq(nf, nr) {
                if let Some(p) = b.piece_at(t) {
                    if p.side == by {
                        if diag {
                            if matches!(p.kind, PieceKind::Bishop | PieceKind::Queen) {
                                return true;
                            }
                        } else {
                            if matches!(p.kind, PieceKind::Rook | PieceKind::Queen) {
                                return true;
                            }
                        }
                    }
                    break;
                }
                nf += df;
                nr += dr;
            } else {
                break;
            }
        }
    }
    false
}
