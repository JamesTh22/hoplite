use crate::board::{Board, Undo};
use crate::endgame;
use crate::movegen::{legal_moves, square_attacked};
use crate::nnue::{Accumulator, NnueNetwork};
use crate::params::PARAMS;
use crate::types::{Move, PieceKind, Side};
use std::sync::Arc;

pub trait Evaluator: Send + Sync {
    fn init_state(&self, b: &Board) -> EvalState;
    fn eval(&self, b: &Board, ctx: &mut EvalState) -> i16;
    fn push_state(&self, b: &Board, ctx: &mut EvalState, mv: Move, undo: &Undo);
    fn pop_state(&self, ctx: &mut EvalState);
}

#[derive(Clone, Default)]
pub struct EvalState {
    nnue_stack: Option<Vec<Accumulator>>,
}

impl EvalState {
    pub fn new() -> Self {
        Self { nnue_stack: None }
    }

    pub fn nnue_stack(&self) -> Option<&[Accumulator]> {
        self.nnue_stack.as_deref()
    }

    pub fn nnue_stack_mut(&mut self) -> Option<&mut Vec<Accumulator>> {
        self.nnue_stack.as_mut()
    }

    pub fn set_nnue_stack(&mut self, stack: Vec<Accumulator>) {
        self.nnue_stack = Some(stack);
    }
}

pub struct PsqtEvaluator;

impl Evaluator for PsqtEvaluator {
    fn init_state(&self, _b: &Board) -> EvalState {
        EvalState::new()
    }

    fn eval(&self, b: &Board, _ctx: &mut EvalState) -> i16 {
        // PSTs from White's perspective
        const PAWN: [i16; 64] = [
            0, 5, 5, -5, -5, 5, 5, 0, 0, 10, -5, 0, 0, -5, 10, 0, 0, 10, 10, 20, 20, 10, 10, 0, 5,
            10, 20, 35, 35, 20, 10, 5, 10, 20, 30, 40, 40, 30, 20, 10, 15, 25, 35, 45, 45, 35, 25,
            15, 20, 30, 30, 0, 0, 30, 30, 20, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        const KNIGHT: [i16; 64] = [
            -50, -30, -10, -10, -10, -10, -30, -50, -30, -5, 5, 10, 10, 5, -5, -30, -10, 10, 15,
            20, 20, 15, 10, -10, -10, 5, 20, 25, 25, 20, 5, -10, -10, 5, 20, 25, 25, 20, 5, -10,
            -10, 10, 15, 20, 20, 15, 10, -10, -30, -5, 5, 10, 10, 5, -5, -30, -50, -30, -10, -10,
            -10, -10, -30, -50,
        ];
        const BISHOP: [i16; 64] = [
            -20, -10, -10, -10, -10, -10, -10, -20, -10, 5, 0, 5, 5, 0, 5, -10, -10, 10, 10, 15,
            15, 10, 10, -10, -10, 10, 15, 20, 20, 15, 10, -10, -10, 10, 15, 20, 20, 15, 10, -10,
            -10, 10, 10, 15, 15, 10, 10, -10, -10, 5, 0, 5, 5, 0, 5, -10, -20, -10, -10, -10, -10,
            -10, -10, -20,
        ];
        const ROOK: [i16; 64] = [
            0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 5, 10, 10, 5, 0, 0, 5, 10,
            10, 15, 15, 10, 10, 5, 5, 10, 10, 15, 15, 10, 10, 5, 0, 0, 5, 10, 10, 5, 0, 0, 0, 0, 5,
            10, 10, 5, 0, 0, 0, 0, 5, 10, 10, 5, 0, 0,
        ];
        const QUEEN: [i16; 64] = [
            -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5,
            0, -10, -5, 0, 5, 10, 10, 5, 0, -5, -5, 0, 5, 10, 10, 5, 0, -5, -10, 0, 5, 5, 5, 5, 0,
            -10, -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
        ];
        const KING: [i16; 64] = [
            -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30,
            -40, -40, -50, -50, -40, -40, -30, -30, -30, -30, -40, -40, -30, -30, -30, -20, -20,
            -20, -20, -20, -20, -20, -20, -10, -10, -10, -10, -10, -10, -10, -10, 20, 20, 0, 0, 0,
            0, 20, 20, 20, 30, 10, 0, 0, 10, 30, 20,
        ];

        let p = PARAMS.read();
        let pv = p.piece_val;

        // Material + PST
        let mut mg: i32 = 0;
        let mut eg: i32 = 0;
        let mut bishops = [0u8; 2];
        let mut pawns_by_file = [[0u8; 8]; 2];
        let mut king_sq = [None::<u8>; 2];
        let mut passed_pawns: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
        for i in 0..64u8 {
            if let Some(pc) = b.piece_at(i) {
                let side_idx = if matches!(pc.side, Side::White) { 0 } else { 1 };
                if matches!(pc.kind, PieceKind::Bishop) {
                    bishops[side_idx] += 1;
                }
                if matches!(pc.kind, PieceKind::Pawn) {
                    pawns_by_file[side_idx][file_of(i)] += 1;
                }
                if matches!(pc.kind, PieceKind::King) {
                    king_sq[side_idx] = Some(i);
                }

                let idx_white = i as usize;
                let idx_black = (63 - i) as usize;
                let (pst, scale) = match pc.kind {
                    PieceKind::Pawn => (&PAWN, p.pst_scale[0]),
                    PieceKind::Knight => (&KNIGHT, p.pst_scale[1]),
                    PieceKind::Bishop => (&BISHOP, p.pst_scale[2]),
                    PieceKind::Rook => (&ROOK, p.pst_scale[3]),
                    PieceKind::Queen => (&QUEEN, p.pst_scale[4]),
                    PieceKind::King => (&KING, p.pst_scale[5]),
                };
                let pst_score = if pc.side == Side::White {
                    pst[idx_white]
                } else {
                    pst[idx_black]
                };
                let pst_scaled = (pst_score as f32 * scale) as i16;
                let val = match pc.kind {
                    PieceKind::Pawn => pv[0] as i32,
                    PieceKind::Knight => pv[1] as i32,
                    PieceKind::Bishop => pv[2] as i32,
                    PieceKind::Rook => pv[3] as i32,
                    PieceKind::Queen => pv[4] as i32,
                    PieceKind::King => pv[5] as i32,
                };
                let term = (val as i16 + pst_scaled) as i32;
                if pc.side == Side::White {
                    mg += term;
                    eg += term;
                } else {
                    mg -= term;
                    eg -= term;
                }
            }
        }

        // Bishop pair
        if bishops[0] >= 2 {
            mg += p.bishop_pair as i32;
            eg += p.bishop_pair as i32;
        }
        if bishops[1] >= 2 {
            mg -= p.bishop_pair as i32;
            eg -= p.bishop_pair as i32;
        }

        // Rook on open / semi-open files
        for i in 0..64u8 {
            if let Some(pc) = b.piece_at(i) {
                if matches!(pc.kind, PieceKind::Rook) {
                    let side_idx = if pc.side == Side::White { 0 } else { 1 };
                    let f = file_of(i);
                    let friendly_pawns = pawns_by_file[side_idx][f];
                    let enemy_pawns = pawns_by_file[1 - side_idx][f];
                    if friendly_pawns == 0 && enemy_pawns == 0 {
                        let bonus = p.rook_open_file as i32;
                        if side_idx == 0 {
                            mg += bonus;
                            eg += bonus;
                        } else {
                            mg -= bonus;
                            eg -= bonus;
                        }
                    } else if friendly_pawns == 0 && enemy_pawns > 0 {
                        let bonus = p.rook_semi_open_file as i32;
                        if side_idx == 0 {
                            mg += bonus;
                            eg += bonus;
                        } else {
                            mg -= bonus;
                            eg -= bonus;
                        }
                    }
                }
            }
        }

        // Pawn structure: isolated, doubled, passed (rank-based)
        for side_idx in 0..2 {
            let sign = if side_idx == 0 { 1 } else { -1 };
            // doubled
            for f in 0..8 {
                if pawns_by_file[side_idx][f] > 1 {
                    let extra = (pawns_by_file[side_idx][f] - 1) as i32;
                    let penalty = sign * extra * (p.doubled_pawn as i32);
                    mg += penalty;
                    eg += penalty;
                }
            }
            // isolated & passed per pawn
            for i in 0..64u8 {
                if let Some(pc) = b.piece_at(i) {
                    if (side_idx == 0 && pc.side == Side::White)
                        || (side_idx == 1 && pc.side == Side::Black)
                    {
                        if matches!(pc.kind, PieceKind::Pawn) {
                            let f = file_of(i) as i32;
                            let r = rank_of(i) as i32;
                            // isolated
                            let left = if f > 0 {
                                pawns_by_file[side_idx][(f - 1) as usize]
                            } else {
                                0
                            };
                            let right = if f < 7 {
                                pawns_by_file[side_idx][(f + 1) as usize]
                            } else {
                                0
                            };
                            if left == 0 && right == 0 {
                                let pen = sign * (p.isolated_pawn as i32);
                                mg += pen;
                                eg += pen;
                            }

                            // passed: scan enemy pawns
                            let mut enemy_block = false;
                            for sq in 0..64u8 {
                                if let Some(ep) = b.piece_at(sq) {
                                    if ep.side
                                        != (if side_idx == 0 {
                                            Side::Black
                                        } else {
                                            Side::White
                                        })
                                    {
                                        continue;
                                    }
                                    if !matches!(ep.kind, PieceKind::Pawn) {
                                        continue;
                                    }
                                    let ef = file_of(sq) as i32;
                                    let er = rank_of(sq) as i32;
                                    if (ef - f).abs() <= 1 {
                                        if side_idx == 0 {
                                            if er > r {
                                                enemy_block = true;
                                            }
                                        } else if er < r {
                                            enemy_block = true;
                                        }
                                    }
                                }
                            }
                            if !enemy_block {
                                let rel_rank = if side_idx == 0 { r } else { 7 - r };
                                let idx = rel_rank.clamp(0, 7) as usize;
                                let bonus = sign * (p.passed_pawn[idx] as i32);
                                mg += bonus;
                                eg += bonus;
                                passed_pawns[side_idx].push(i);
                            }
                        }
                    }
                }
            }
        }

        // Connected passers & rook behind passer (endgame features)
        for side_idx in 0..2 {
            let sign = if side_idx == 0 { 1 } else { -1 };
            let passers = &passed_pawns[side_idx];
            for (i, &sq1) in passers.iter().enumerate() {
                for &sq2 in passers.iter().skip(i + 1) {
                    if (file_of(sq1) as i32 - file_of(sq2) as i32).abs() == 1
                        && rank_of(sq1) == rank_of(sq2)
                    {
                        eg += sign * (p.connected_passers as i32);
                    }
                }
            }
            for &sq in passers.iter() {
                let f = file_of(sq) as i32;
                let mut r = rank_of(sq) as i32 + if side_idx == 0 { -1 } else { 1 };
                while r >= 0 && r <= 7 {
                    let idx = (r * 8 + f) as u8;
                    if let Some(pc) = b.piece_at(idx) {
                        if pc.side
                            == (if side_idx == 0 {
                                Side::White
                            } else {
                                Side::Black
                            })
                            && matches!(pc.kind, PieceKind::Rook)
                        {
                            eg += sign * (p.rook_behind_passer as i32);
                        }
                        break;
                    }
                    r += if side_idx == 0 { -1 } else { 1 };
                }
            }
        }

        // King pawn shield (front three squares one rank ahead) - middlegame
        for side_idx in 0..2 {
            if let Some(ksq) = king_sq[side_idx] {
                let kf = file_of(ksq) as i32;
                let kr = rank_of(ksq) as i32;
                let target_rank = kr + if side_idx == 0 { 1 } else { -1 };
                if target_rank >= 0 && target_rank <= 7 {
                    let mut missing = 0i32;
                    for df in -1..=1 {
                        let nf = kf + df;
                        if nf < 0 || nf > 7 {
                            continue;
                        }
                        let idx = (target_rank * 8 + nf) as usize;
                        let pawn_here = match b.pieces[idx] {
                            Some(pc) => {
                                matches!(pc.kind, PieceKind::Pawn)
                                    && ((side_idx == 0 && pc.side == Side::White)
                                        || (side_idx == 1 && pc.side == Side::Black))
                            }
                            None => false,
                        };
                        if !pawn_here {
                            missing += 1;
                        }
                    }
                    let pen = (p.king_shield_missing as i32) * missing;
                    if side_idx == 0 {
                        mg += pen;
                    } else {
                        mg -= pen;
                    }
                }
            }
        }

        // King ring penalty (undefended attacked squares around king)
        for side_idx in 0..2 {
            if let Some(ksq) = king_sq[side_idx] {
                let kf = file_of(ksq) as i32;
                let kr = rank_of(ksq) as i32;
                let my_side = if side_idx == 0 {
                    Side::White
                } else {
                    Side::Black
                };
                let enemy = my_side.flip();
                for df in -1..=1 {
                    for dr in -1..=1 {
                        if df == 0 && dr == 0 {
                            continue;
                        }
                        if let Some(t) = crate::types::sq(kf + df, kr + dr) {
                            if square_attacked(b, t, enemy) && !square_attacked(b, t, my_side) {
                                if side_idx == 0 {
                                    mg -= p.king_ring_penalty as i32;
                                } else {
                                    mg += p.king_ring_penalty as i32;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Mobility (legal move counts) - middlegame
        let mut mob = [[0i32; 4]; 2];
        for (side_idx, side) in [Side::White, Side::Black].iter().enumerate() {
            let mut bb = b.clone();
            if bb.stm != *side {
                bb.stm = *side;
                bb.key ^= bb.zob.stm;
            }
            let moves = legal_moves(&bb);
            for mv in moves {
                if let Some(pc) = bb.piece_at(mv.from) {
                    match pc.kind {
                        PieceKind::Knight => mob[side_idx][0] += 1,
                        PieceKind::Bishop => mob[side_idx][1] += 1,
                        PieceKind::Rook => mob[side_idx][2] += 1,
                        PieceKind::Queen => mob[side_idx][3] += 1,
                        _ => {}
                    }
                }
            }
        }
        mg += (mob[0][0] - mob[1][0]) * p.mobility_knight as i32;
        mg += (mob[0][1] - mob[1][1]) * p.mobility_bishop as i32;
        mg += (mob[0][2] - mob[1][2]) * p.mobility_rook as i32;
        mg += (mob[0][3] - mob[1][3]) * p.mobility_queen as i32;

        // Blend middlegame and endgame scores
        let denom = (p.mg_weight as i32 + p.eg_weight as i32).max(1);
        let mut blended = (mg * p.mg_weight as i32 + eg * p.eg_weight as i32) / denom;

        // Apply endgame-specific scoring adjustments
        let endgame_adjustment = endgame::get_endgame_score_adjustment(b);
        blended += endgame_adjustment as i32;

        let out = if b.stm == Side::White {
            blended
        } else {
            -blended
        };
        out as i16
    }

    fn push_state(&self, _b: &Board, _ctx: &mut EvalState, _mv: Move, _undo: &Undo) {}

    fn pop_state(&self, _ctx: &mut EvalState) {}
}

pub fn default_evaluator() -> Arc<dyn Evaluator> {
    Arc::new(PsqtEvaluator)
}

pub fn nnue_evaluator(network: Arc<NnueNetwork>) -> Arc<dyn Evaluator> {
    Arc::new(NnueEvaluator::new(network))
}

pub struct NnueEvaluator {
    network: Arc<NnueNetwork>,
}

impl NnueEvaluator {
    pub fn new(network: Arc<NnueNetwork>) -> Self {
        Self { network }
    }
}

impl Evaluator for NnueEvaluator {
    fn init_state(&self, b: &Board) -> EvalState {
        let mut state = EvalState::new();
        state.set_nnue_stack(vec![Accumulator::from_board(b)]);
        state
    }

    fn eval(&self, b: &Board, ctx: &mut EvalState) -> i16 {
        if ctx.nnue_stack_mut().is_none() {
            ctx.set_nnue_stack(vec![Accumulator::from_board(b)]);
        }

        if let Some(stack) = ctx.nnue_stack_mut() {
            if stack.is_empty() {
                stack.push(Accumulator::from_board(b));
            }
            if let Some(acc) = stack.last_mut() {
                return self
                    .network
                    .evaluate(b, acc)
                    .expect("NNUE evaluation failed");
            }
        }

        panic!("NNUE accumulator stack missing");
    }

    fn push_state(&self, b: &Board, ctx: &mut EvalState, mv: Move, undo: &Undo) {
        if let Some(stack) = ctx.nnue_stack_mut() {
            let mut next = stack
                .last()
                .cloned()
                .unwrap_or_else(|| Accumulator::from_board(b));
            next.update(b, mv, undo);
            stack.push(next);
        } else {
            ctx.set_nnue_stack(vec![Accumulator::from_board(b)]);
        }
    }

    fn pop_state(&self, ctx: &mut EvalState) {
        if let Some(stack) = ctx.nnue_stack_mut() {
            if stack.len() > 1 {
                stack.pop();
            }
        }
    }
}

#[inline]
fn file_of(sq: u8) -> usize {
    (sq as usize) & 7
}

#[inline]
fn rank_of(sq: u8) -> usize {
    (sq as usize) >> 3
}
