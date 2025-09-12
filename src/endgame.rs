use crate::board::Board;
use crate::types::{PieceKind, Side};
use lazy_static::lazy_static;
use std::sync::RwLock;

// Cache endgame classifications
lazy_static! {
    static ref ENDGAME_CACHE: RwLock<Vec<(u64, EndgameType)>> = RwLock::new(Vec::with_capacity(1024));
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum EndgameType {
    KPK,     // King and pawn vs king
    KBNK,    // King, bishop and knight vs king
    KRK,     // King and rook vs king
    KBBK,    // King and two bishops vs king
    KQRK,    // King, queen and rook vs king
    KQPK,    // King, queen and pawn vs king
    KRPK,    // King, rook and pawn vs king
    KPPK,    // King and two pawns vs king
    NORMAL,  // Normal endgame or not endgame
}

impl EndgameType {
    pub fn piece_value_adjustment(&self) -> i16 {
        match self {
            EndgameType::KPK => 50,      // Bonus for pushing pawns
            EndgameType::KBNK => 30,     // Help coordinate pieces
            EndgameType::KRK => 40,      // Drive king to edge
            EndgameType::KBBK => 25,     // Coordinate bishops
            EndgameType::KQRK => -10,    // Simpler to win with just queen
            EndgameType::KQPK => 0,      // Normal evaluation ok
            EndgameType::KRPK => 35,     // Bonus for keeping rook behind pawn
            EndgameType::KPPK => 45,     // Bonus for connected pawns
            EndgameType::NORMAL => 0,     // No adjustment
        }
    }

    pub fn search_extension(&self) -> i32 {
        match self {
            EndgameType::KPK => 2,       // Critical to push pawns correctly
            EndgameType::KBNK => 1,      // Need precise coordination
            EndgameType::KRK => 1,       // Drive king to edge
            EndgameType::KBBK => 1,      // Need bishop coordination
            EndgameType::KQRK => 0,      // Easy win
            EndgameType::KQPK => 0,      // Easy win
            EndgameType::KRPK => 1,      // Precise rook placement needed
            EndgameType::KPPK => 1,      // Careful pawn advancement
            EndgameType::NORMAL => 0,     // No extension
        }
    }
}

pub fn identify_endgame(b: &Board) -> EndgameType {
    // Check cache first
    {
        let cache = ENDGAME_CACHE.read().unwrap();
        if let Some(entry) = cache.iter().find(|(hash, _)| *hash == b.key) {
            return entry.1;
        }
    }

    // Count material
    let mut white_material = [0u8; 6];
    let mut black_material = [0u8; 6];
    
    for i in 0..64u8 {
        if let Some(pc) = b.piece_at(i) {
            let arr = if pc.side == Side::White { &mut white_material } else { &mut black_material };
            arr[pc.kind as usize] += 1;
        }
    }

    // Determine endgame type
    let endgame = match (white_material, black_material) {
        // KPK
        (w, b) if w[PieceKind::King as usize] == 1 
                 && w[PieceKind::Pawn as usize] == 1 
                 && b[PieceKind::King as usize] == 1 
                 && is_lone_king(&b) => EndgameType::KPK,
                 
        // KBNK
        (w, b) if w[PieceKind::King as usize] == 1 
                 && w[PieceKind::Bishop as usize] == 1
                 && w[PieceKind::Knight as usize] == 1 
                 && b[PieceKind::King as usize] == 1
                 && is_lone_king(&b) => EndgameType::KBNK,

        // KRK
        (w, b) if w[PieceKind::King as usize] == 1 
                 && w[PieceKind::Rook as usize] == 1
                 && b[PieceKind::King as usize] == 1 
                 && is_lone_king(&b) => EndgameType::KRK,

        // KBBK                 
        (w, b) if w[PieceKind::King as usize] == 1 
                 && w[PieceKind::Bishop as usize] == 2
                 && b[PieceKind::King as usize] == 1 
                 && is_lone_king(&b) => EndgameType::KBBK,

        // KQRK
        (w, b) if w[PieceKind::King as usize] == 1 
                 && w[PieceKind::Queen as usize] == 1
                 && w[PieceKind::Rook as usize] == 1
                 && b[PieceKind::King as usize] == 1 
                 && is_lone_king(&b) => EndgameType::KQRK,

        // KQPK
        (w, b) if w[PieceKind::King as usize] == 1 
                 && w[PieceKind::Queen as usize] == 1
                 && w[PieceKind::Pawn as usize] == 1
                 && b[PieceKind::King as usize] == 1 
                 && is_lone_king(&b) => EndgameType::KQPK,

        // KRPK
        (w, b) if w[PieceKind::King as usize] == 1 
                 && w[PieceKind::Rook as usize] == 1
                 && w[PieceKind::Pawn as usize] == 1
                 && b[PieceKind::King as usize] == 1 
                 && is_lone_king(&b) => EndgameType::KRPK,

        // KPPK
        (w, b) if w[PieceKind::King as usize] == 1 
                 && w[PieceKind::Pawn as usize] == 2
                 && b[PieceKind::King as usize] == 1 
                 && is_lone_king(&b) => EndgameType::KPPK,

        _ => EndgameType::NORMAL,
    };

    // Cache the result
    {
        let mut cache = ENDGAME_CACHE.write().unwrap();
        if cache.len() >= 1024 {
            cache.clear();
        }
        cache.push((b.key, endgame));
    }

    endgame
}

fn is_lone_king(material: &[u8; 6]) -> bool {
    material[PieceKind::King as usize] == 1 
        && material[PieceKind::Queen as usize] == 0
        && material[PieceKind::Rook as usize] == 0 
        && material[PieceKind::Bishop as usize] == 0
        && material[PieceKind::Knight as usize] == 0 
        && material[PieceKind::Pawn as usize] == 0
}

pub fn get_endgame_score_adjustment(board: &Board) -> i16 {
    let endgame = identify_endgame(board);
    
    // Apply appropriate scoring adjustment based on endgame type
    let base_adjustment = endgame.piece_value_adjustment();
    
    // Additional positional adjustments for specific endgames
    match endgame {
        EndgameType::KPK => {
            // Bonus for pushing pawns and king position
            let mut score = base_adjustment;
            for i in 0..64u8 {
                if let Some(pc) = board.piece_at(i) {
                    if pc.kind == PieceKind::Pawn {
                        // Bonus for pawn advancement
                        let rank = if pc.side == Side::White {
                            i / 8
                        } else {
                            7 - (i / 8)
                        };
                        score += (rank as i16) * 10;
                    } else if pc.kind == PieceKind::King {
                        // King should support pawn
                        let pawn_file = find_pawn_file(board, pc.side);
                        if let Some(pf) = pawn_file {
                            let kf = i % 8;
                            let distance = ((pf as i16) - (kf as i16)).abs();
                            score -= distance * 5;
                        }
                    }
                }
            }
            score
        },
        EndgameType::KBNK => {
            // Bonus for driving enemy king to correct corner
            let mut score = base_adjustment;
            if let Some(enemy_king_sq) = find_king(board, board.stm.flip()) {
                let corner_dist = distance_to_corner(enemy_king_sq);
                score += (8 - corner_dist as i16) * 10;
            }
            score
        },
        EndgameType::KRK => {
            // Bonus for restricting enemy king
            let mut score = base_adjustment;
            if let Some(enemy_king_sq) = find_king(board, board.stm.flip()) {
                let edge_dist = distance_to_edge(enemy_king_sq);
                score += (8 - edge_dist as i16) * 15;
            }
            score
        },
        _ => base_adjustment,
    }
}

fn find_king(board: &Board, side: Side) -> Option<u8> {
    for i in 0..64u8 {
        if let Some(pc) = board.piece_at(i) {
            if pc.side == side && pc.kind == PieceKind::King {
                return Some(i);
            }
        }
    }
    None
}

fn find_pawn_file(board: &Board, side: Side) -> Option<u8> {
    for i in 0..64u8 {
        if let Some(pc) = board.piece_at(i) {
            if pc.side == side && pc.kind == PieceKind::Pawn {
                return Some(i % 8);
            }
        }
    }
    None
}

fn distance_to_corner(sq: u8) -> u8 {
    let file = sq % 8;
    let rank = sq / 8;
    let dist_to_a1 = file.max(rank);
    let dist_to_a8 = file.max(7 - rank);
    let dist_to_h1 = (7 - file).max(rank);
    let dist_to_h8 = (7 - file).max(7 - rank);
    dist_to_a1.min(dist_to_a8).min(dist_to_h1).min(dist_to_h8)
}

fn distance_to_edge(sq: u8) -> u8 {
    let file = sq % 8;
    let rank = sq / 8;
    file.min(7 - file).min(rank).min(7 - rank)
}