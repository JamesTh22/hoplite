use crate::board::Board;
use crate::types::{sq, Move, PieceKind, Side};

mod movecache;
use lazy_static::lazy_static;

const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

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

    static ref ROOK_MASKS: [u64; 64] = generate_masks(&ROOK_DIRS);
    static ref BISHOP_MASKS: [u64; 64] = generate_masks(&BISHOP_DIRS);
    static ref ROOK_ATTACKS: Vec<Vec<u64>> = generate_tables(&ROOK_DIRS, &ROOK_MASKS);
    static ref BISHOP_ATTACKS: Vec<Vec<u64>> = generate_tables(&BISHOP_DIRS, &BISHOP_MASKS);
}

fn generate_masks(dirs: &[(i32, i32); 4]) -> [u64; 64] {
    let mut arr = [0u64; 64];
    for s in 0u8..64 {
        let f = (s % 8) as i32;
        let r = (s / 8) as i32;
        for (df, dr) in dirs.iter() {
            let mut nf = f + df;
            let mut nr = r + dr;
            while let Some(t) = sq(nf, nr) {
                if (nf == 0 && *df == -1)
                    || (nf == 7 && *df == 1)
                    || (nr == 0 && *dr == -1)
                    || (nr == 7 && *dr == 1)
                {
                    break;
                }
                arr[s as usize] |= 1u64 << t;
                nf += df;
                nr += dr;
            }
        }
    }
    arr
}

fn generate_tables(dirs: &[(i32, i32); 4], masks: &[u64; 64]) -> Vec<Vec<u64>> {
    let mut tables: Vec<Vec<u64>> = Vec::with_capacity(64);
    for s in 0u8..64 {
        let mask = masks[s as usize];
        let bits = mask.count_ones();
        let mut table = vec![0u64; 1 << bits];
        for idx in 0..(1 << bits) {
            let occ = set_occupancy(idx as usize, bits, mask);
            table[idx as usize] = slider_attacks_from(s, occ, dirs);
        }
        tables.push(table);
    }
    tables
}

fn set_occupancy(index: usize, bits: u32, mask: u64) -> u64 {
    let mut occ = 0u64;
    let mut bitboard = mask;
    for i in 0..bits {
        let sq = bitboard.trailing_zeros() as u8;
        bitboard &= bitboard - 1;
        if (index & (1 << i)) != 0 {
            occ |= 1u64 << sq;
        }
    }
    occ
}

fn slider_attacks_from(s: u8, occ: u64, dirs: &[(i32, i32); 4]) -> u64 {
    let f = (s % 8) as i32;
    let r = (s / 8) as i32;
    let mut attacks = 0u64;
    for (df, dr) in dirs.iter() {
        let mut nf = f + df;
        let mut nr = r + dr;
        while let Some(t) = sq(nf, nr) {
            attacks |= 1u64 << t;
            if (occ & (1u64 << t)) != 0 {
                break;
            }
            nf += df;
            nr += dr;
        }
    }
    attacks
}

fn occupancy_to_index(mut occ: u64, mask: u64) -> usize {
    occ &= mask;
    let mut index = 0usize;
    let mut bits = mask;
    let mut i = 0;
    while bits != 0 {
        let sq = bits.trailing_zeros();
        bits &= bits - 1;
        if occ & (1u64 << sq) != 0 {
            index |= 1 << i;
        }
        i += 1;
    }
    index
}

fn rook_attacks(occ: u64, s: u8) -> u64 {
    let mask = ROOK_MASKS[s as usize];
    let idx = occupancy_to_index(occ, mask);
    ROOK_ATTACKS[s as usize][idx]
}

fn bishop_attacks(occ: u64, s: u8) -> u64 {
    let mask = BISHOP_MASKS[s as usize];
    let idx = occupancy_to_index(occ, mask);
    BISHOP_ATTACKS[s as usize][idx]
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
    // Check cache first
    if let Some(cached) = super::movecache::get_cached_legal_moves(b.key) {
        return cached;
    }

    let mut ms = Vec::with_capacity(64);
    pseudo_legal(b, &mut ms);
    
    // Filter illegal moves (leaves king in check)
    let mut legal = Vec::with_capacity(ms.len());
    let my_king = find_king(b, b.stm);
    
    for m in ms.into_iter() {
        let mut bb = b.clone();
        let u = bb.make_move(m);
        
        // Quick check for direct king moves
        let is_king_move = match b.piece_at(m.from) {
            Some(p) => p.kind == PieceKind::King,
            None => false
        };
        
        // If king move, check if target square is attacked
        if is_king_move {
            if !square_attacked(&bb, m.to, b.stm.flip()) {
                legal.push(m);
            }
        } else {
            // For non-king moves, check if our king is in check
            if let Some(king_sq) = my_king {
                if !square_attacked(&bb, king_sq, b.stm.flip()) {
                    legal.push(m);
                }
            }
        }
        bb.unmake_move(m, u);
    }

    // Cache the results
    super::movecache::store_legal_moves(b.key, legal.clone());
    legal
}

// Helper function to find king square
fn find_king(b: &Board, side: Side) -> Option<u8> {
    for s in 0..64u8 {
        if let Some(p) = b.piece_at(s) {
            if p.side == side && p.kind == PieceKind::King {
                return Some(s);
            }
        }
    }
    None
}

pub fn pseudo_legal(b: &Board, out: &mut Vec<Move>) {
    let side = b.stm;
    let occ = b.occupancy();
    for sqi in 0u8..64 {
        if let Some(p) = b.piece_at(sqi) {
            if p.side != side {
                continue;
            }
            match p.kind {
                PieceKind::Pawn => pawn_moves(b, sqi, p.side, out),
                PieceKind::Knight => knight_moves(b, sqi, p.side, out),
                PieceKind::Bishop => {
                    slider_moves(b, sqi, p.side, out, PieceKind::Bishop, occ)
                }
                PieceKind::Rook => {
                    slider_moves(b, sqi, p.side, out, PieceKind::Rook, occ)
                }
                PieceKind::Queen => slider_moves(b, sqi, p.side, out, PieceKind::Queen, occ),
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

fn slider_moves(b: &Board, s: u8, side: Side, out: &mut Vec<Move>, kind: PieceKind, occ: u64) {
    let mut bb = match kind {
        PieceKind::Bishop => bishop_attacks(occ, s),
        PieceKind::Rook => rook_attacks(occ, s),
        PieceKind::Queen => bishop_attacks(occ, s) | rook_attacks(occ, s),
        _ => 0,
    };
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

pub fn square_attacked(b: &Board, s: u8, by: Side) -> bool {
    let f = (s % 8) as i32;
    let r = (s / 8) as i32;
    let occ = b.occupancy();
    
    // Check sliding piece attacks first (most common attackers)
    let rook_queen_attacks = rook_attacks(occ, s);
    let bishop_queen_attacks = bishop_attacks(occ, s);
    
    // Check all attackers in one pass
    for bb in [rook_queen_attacks, bishop_queen_attacks] {
        let mut attacks = bb;
        while attacks != 0 {
            let t = attacks.trailing_zeros() as u8;
            attacks &= attacks - 1;
            if let Some(p) = b.piece_at(t) {
                if p.side == by {
                    match p.kind {
                        PieceKind::Queen => return true,
                        PieceKind::Rook if bb == rook_queen_attacks => return true,
                        PieceKind::Bishop if bb == bishop_queen_attacks => return true,
                        _ => {}
                    }
                }
            }
        }
    }
    
    // Pawn attacks (fast check using precomputed targets)
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
    
    // Knight attacks (using bitboard)
    let mut knights = KNIGHT_TARGETS[s as usize];
    while knights != 0 {
        let t = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        if let Some(p) = b.piece_at(t) {
            if p.side == by && matches!(p.kind, PieceKind::Knight) {
                return true;
            }
        }
    }
    
    // King attacks (last because least likely)
    let mut kings = KING_TARGETS[s as usize];
    while kings != 0 {
        let t = kings.trailing_zeros() as u8;
        kings &= kings - 1;
        if let Some(p) = b.piece_at(t) {
            if p.side == by && matches!(p.kind, PieceKind::King) {
                return true;
            }
        }
    }
    
    false
}
