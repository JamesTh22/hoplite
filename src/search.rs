use crate::board::Board;
use crate::movegen::legal_moves;
use crate::params::PARAMS;
use crate::tt::{Bound, Entry, TT};
use crate::types::{Move, PieceKind, Side};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_PLY: usize = 128;
#[inline]
#[allow(dead_code)]
fn pack_move16(m: Move) -> u16 {
    ((m.from as u16) << 6)
        | (m.to as u16)
        | ((m.promo.unwrap_or(crate::types::PieceKind::Pawn) as u16) << 12)
}
#[inline]
fn is_draw(history: &[u64], key: u64, halfmove: u32) -> bool {
    if halfmove >= 100 {
        return true;
    } // 50-move rule
    let mut cnt = 0;
    for &k in history.iter().rev() {
        if k == key {
            cnt += 1;
            if cnt >= 3 {
                return true;
            }
        }
    }
    false
}

#[derive(Clone)]
pub struct Search {
    // Learning & caches
    pub exp_table: HashMap<u128, (u32, u32)>, // key: (zobrist<<16)|move16 -> (plays,wins)
    pub exp_enabled: bool,
    pub exp_strength: i32, // 0..100
    pub exp_path: Option<String>,

    pub pawn_tt: HashMap<u64, i32>,
    pub move_overhead_ms: u64,
    pub nodes: u64,
    pub tt: Arc<Mutex<TT>>,
    pub stop: bool,
    pub threads: usize,
    pub deadline: Option<Instant>,

    // Move ordering
    pub killers: [[Move; 2]; MAX_PLY], // two killer moves per ply
    pub history: [[i32; 4096]; 2],     // side, from*64+to
}

impl Search {
    pub fn new() -> Self {
        Self {
            nodes: 0,
            tt: Arc::new(Mutex::new(TT::new(64))),
            stop: false,
            threads: 1,
            deadline: None,
            pawn_tt: HashMap::with_capacity(1 << 16),
            exp_table: HashMap::with_capacity(1 << 14),
            exp_enabled: false,
            exp_strength: 40,
            exp_path: None,
            move_overhead_ms: 15,
            killers: [[Move::default(); 2]; MAX_PLY],
            history: [[0; 4096]; 2],
        }
    }
    pub fn set_hash_mb(&mut self, mb: usize) {
        *self.tt.lock() = TT::new(mb);
    }
    pub fn set_threads(&mut self, n: usize) {
        self.threads = n.max(1);
        // Try to configure rayon's global thread pool. This is a no-op if already set.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .build_global();
    }

    #[inline]
    fn time_up(&self) -> bool {
        if let Some(d) = self.deadline {
            Instant::now() >= d
        } else {
            false
        }
    }

    // Depth-limited
    pub fn bestmove(&mut self, b: &mut Board, depth: i32) -> Move {
        let mut history_keys: Vec<u64> = vec![b.key];
        self.nodes = 0;
        self.stop = false;
        self.deadline = None;
        let mut best = Move::default();
        let mut last_score: i16 = 0;

        for d in 1..=depth {
            let (m, sc) = self.search_root(b, d, &mut history_keys, last_score);
            if self.stop {
                break;
            }
            last_score = sc;
            if m.from != 0 || m.to != 0 {
                best = m;
            }
        }
        best
    }

    // Time-limited
    pub fn bestmove_time(&mut self, b: &mut Board, time_ms: u128) -> Move {
        let mut history_keys: Vec<u64> = vec![b.key];
        self.nodes = 0;
        self.stop = false;
        self.deadline = Some(Instant::now() + Duration::from_millis(time_ms as u64));
        let mut best = Move::default();
        let mut last_score: i16 = 0;

        for d in 1..=64 {
            let (m, sc) = self.search_root(b, d, &mut history_keys, last_score);
            last_score = sc;
            if m.from != 0 || m.to != 0 {
                best = m;
            }
            if self.time_up() {
                self.stop = true;
                break;
            }
        }
        best
    }

    fn search_root(
        &mut self,
        b: &mut Board,
        depth: i32,
        history_keys: &mut Vec<u64>,
        prev_score: i16,
    ) -> (Move, i16) {
        let moves = legal_moves(b);
        if moves.is_empty() {
            return (Move::default(), 0);
        }

        let mut scored = self.score_moves(b, &moves, 0, None);
        scored.sort_by_key(|(s, _)| -*s); // descending

        // Parallel root search when multiple threads are requested
        if self.threads > 1 {
            let base = self.clone();
            let base_board = b.clone();
            let results: Vec<(Move, i16, u64)> = scored
                .par_iter()
                .map(|(_, mv)| {
                    let mut s = base.clone();
                    let mut bb = base_board.clone();
                    let is_capture = bb.piece_at(mv.to).is_some();
                    let _u = bb.make_move(*mv);
                    let gave_check = bb.in_check(bb.stm);
                    let mut new_depth = depth - 1;
                    if gave_check {
                        new_depth += 1;
                    }
                    if depth >= 3 && !is_capture && !gave_check {
                        let d = depth as i32;
                        let mut r = 1 + (d / 3);
                        if r > d - 1 {
                            r = d - 1;
                        }
                        if r > 0 {
                            new_depth -= r;
                        }
                    }
                    let mut hk = history_keys.clone();
                    let sc = -s.alphabeta(
                        &mut bb,
                        new_depth,
                        i16::MIN / 4,
                        i16::MAX / 4,
                        1,
                        false,
                        &mut hk,
                    );
                    (*mv, sc, s.nodes)
                })
                .collect();
            self.nodes += results.iter().map(|(_, _, n)| *n).sum::<u64>();
            if let Some((mv, sc, _)) = results.into_iter().max_by_key(|(_, sc, _)| *sc) {
                if self.time_up() {
                    self.stop = true;
                }
                return (mv, sc);
            } else {
                return (Move::default(), 0);
            }
        }

        let mut best = Move::default();
        let mut alpha = prev_score.saturating_sub(50);
        let mut beta = prev_score.saturating_add(50);
        let (orig_a, orig_b) = (alpha, beta);

        let mut widened_low = false;
        'asp: loop {
            for (_, mv) in scored.clone() {
                if self.time_up() {
                    self.stop = true;
                    break;
                }
                let u = b.make_move(mv);
                let is_capture = b.piece_at(mv.to).is_some();
                let mut new_depth = depth - 1;

                // Check extension
                let gave_check = b.in_check(b.stm);
                if gave_check {
                    new_depth += 1;
                }

                // LMR
                if depth >= 3 && !is_capture && !gave_check {
                    let d = depth as i32;
                    let mut r = 1 + (d / 3);
                    if r > d - 1 {
                        r = d - 1;
                    }
                    if r > 0 {
                        new_depth -= r;
                    }
                }
                let sc = -self.alphabeta(b, new_depth, -beta, -alpha, 1, false, history_keys);
                b.unmake_move(mv, u);
                if sc > alpha {
                    alpha = sc;
                    best = mv;
                }
            }

            if alpha <= orig_a && !widened_low {
                alpha = i16::MIN / 4;
                beta = orig_b;
                widened_low = true;
                continue 'asp;
            }
            if alpha >= orig_b {
                alpha = orig_a;
                beta = i16::MAX / 4;
                continue 'asp;
            }
            break;
        }
        (best, alpha)
    }

    fn alphabeta(
        &mut self,
        b: &mut Board,
        depth: i32,
        mut alpha: i16,
        beta: i16,
        ply: usize,
        _in_null: bool,
        history_keys: &mut Vec<u64>,
    ) -> i16 {
        self.nodes += 1;
        if (self.nodes & 0x1FFF) == 0 && self.time_up() {
            self.stop = true;
            return alpha;
        }
        if is_draw(history_keys, b.key, b.halfmove) {
            return 0;
        }

        if depth <= 0 {
            return self.qsearch(b, alpha, beta, history_keys);
        }

        // Transposition table
        {
            let tt = self.tt.lock();
            if let Some(e) = tt.probe(b.key) {
                match e.bound {
                    Bound::Exact => return e.value,
                    Bound::Lower => {
                        if e.value > alpha {
                            alpha = e.value;
                        }
                    }
                    Bound::Upper => {
                        if e.value < beta {
                            return e.value;
                        }
                    }
                }
            }
        }

        // Null-move pruning (safe)
        if depth >= 3 && has_non_pawn_material(b, b.stm) && !b.in_check(b.stm) {
            let r = 2;
            // Do a null move: switch side to move without making a move
            let saved_stm = b.stm;
            b.stm = if saved_stm == Side::White {
                Side::Black
            } else {
                Side::White
            };
            history_keys.push(b.key);
            let score = -self.alphabeta(
                b,
                depth - 1 - r,
                -beta,
                -beta + 1,
                ply + 1,
                true,
                history_keys,
            );
            history_keys.pop();
            b.stm = saved_stm;
            if score >= beta {
                return beta;
            }
        }

        let moves = legal_moves(b);
        if moves.is_empty() {
            if b.in_check(b.stm) {
                return -30000;
            } else {
                return 0;
            }
        }

        let mut value = i16::MIN / 2;
        // Order moves
        let mut scored = self.score_moves(b, &moves, ply, None);
        scored.sort_by_key(|(s, _)| -*s);

        for (idx, (_, mv)) in scored.into_iter().enumerate() {
            let u = b.make_move(mv);
            let is_capture = b.piece_at(mv.to).is_some();
            let mut new_depth = depth - 1;
            let gave_check = b.in_check(b.stm);
            if gave_check {
                new_depth += 1;
            }
            if depth >= 3 && !is_capture && !gave_check && idx >= 3 {
                let d = depth as i32;
                let m = (idx as i32) + 1;
                let mut r = 1 + ((d / 3) + (m / 6));
                if r > d - 1 {
                    r = d - 1;
                }
                if r > 0 {
                    new_depth -= r;
                }
            }

            history_keys.push(b.key);
            let sc = -self.alphabeta(b, new_depth, -beta, -alpha, ply + 1, false, history_keys);
            history_keys.pop();
            b.unmake_move(mv, u);

            if sc >= beta {
                if !is_capture {
                    let side_idx = if b.stm == Side::White { 0 } else { 1 };
                    let idx_hist = (mv.from as usize) * 64 + (mv.to as usize);
                    let bonus = (depth.max(1) * depth.max(1)) as i32;
                    self.history[side_idx][idx_hist] =
                        self.history[side_idx][idx_hist].saturating_add(bonus);
                    if self.killers[ply][0].from != mv.from || self.killers[ply][0].to != mv.to {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = mv;
                    }
                }
                return beta;
            }
            if sc > value {
                value = sc;
            }
            if sc > alpha {
                alpha = sc;
            }
        }

        // Store in TT
        let bound = if value <= alpha {
            Bound::Upper
        } else if value >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        {
            let mut tt = self.tt.lock();
            tt.store(Entry {
                key: b.key,
                depth: (depth as i8),
                value,
                bound,
                best: Move::default(),
            });
        }
        value
    }

    fn qsearch(
        &mut self,
        b: &mut Board,
        mut alpha: i16,
        beta: i16,
        history_keys: &mut Vec<u64>,
    ) -> i16 {
        self.nodes += 1;
        if (self.nodes & 0x1FFF) == 0 && self.time_up() {
            self.stop = true;
            return alpha;
        }
        if is_draw(history_keys, b.key, b.halfmove) {
            return 0;
        }

        let stand_pat = eval(b);
        if stand_pat >= beta {
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let moves = legal_moves(b);
        let mut captures: Vec<(i32, Move)> = Vec::new();
        let mut checks: Vec<Move> = Vec::new();
        for mv in moves {
            if b.piece_at(mv.to).is_some() {
                captures.push((score_capture(b, mv), mv));
            } else {
                let u = b.make_move(mv);
                if b.in_check(b.stm) {
                    checks.push(mv);
                }
                b.unmake_move(mv, u);
            }
        }
        captures.sort_by_key(|(s, _)| -*s);

        for (_, mv) in captures {
            if see(b, mv.from, mv.to) < 0 {
                continue;
            }
            let u = b.make_move(mv);
            let score = -self.qsearch(b, -beta, -alpha, history_keys);
            b.unmake_move(mv, u);
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        // A few checking moves
        let mut tried = 0;
        for mv in checks {
            if tried >= 4 {
                break;
            }
            tried += 1;
            let u = b.make_move(mv);
            let score = -self.qsearch(b, -beta, -alpha, history_keys);
            b.unmake_move(mv, u);
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        alpha
    }

    fn score_moves(
        &self,
        b: &Board,
        moves: &[Move],
        ply: usize,
        tt_move: Option<Move>,
    ) -> Vec<(i32, Move)> {
        let mut v: Vec<(i32, Move)> = Vec::with_capacity(moves.len());
        for &mv in moves {
            let mut s: i32 = 0;
            if let Some(tmv) = tt_move {
                if tmv.from == mv.from && tmv.to == mv.to && tmv.promo == mv.promo {
                    s += 1_000_000;
                }
            }
            if b.piece_at(mv.to).is_some() {
                if see(b, mv.from, mv.to) < 0 {
                    continue;
                }
                s += score_capture(b, mv);
            } else {
                let side_idx = if b.stm == Side::White { 0 } else { 1 };
                // killers
                if self.killers[ply][0].from == mv.from && self.killers[ply][0].to == mv.to {
                    s += 12_000;
                }
                if self.killers[ply][1].from == mv.from && self.killers[ply][1].to == mv.to {
                    s += 8_000;
                }
                // history
                let idx = (mv.from as usize) * 64 + (mv.to as usize);
                s += self.history[side_idx][idx];
            }
            v.push((s, mv));
        }
        v
    }
}

// Check if side has non-pawn material (for null-move safety)
fn has_non_pawn_material(b: &Board, side: Side) -> bool {
    for i in 0..64u8 {
        if let Some(p) = b.piece_at(i) {
            if p.side == side && !matches!(p.kind, PieceKind::Pawn | PieceKind::King) {
                return true;
            }
        }
    }
    false
}

// Capture scoring helper
fn score_capture(b: &Board, mv: Move) -> i32 {
    // MVV-LVA basic
    let val = |k: PieceKind| -> i32 {
        match k {
            PieceKind::Pawn => 100,
            PieceKind::Knight => 320,
            PieceKind::Bishop => 330,
            PieceKind::Rook => 500,
            PieceKind::Queen => 900,
            PieceKind::King => 10_000,
        }
    };
    let victim = b.piece_at(mv.to).unwrap();
    let attacker = b.piece_at(mv.from).unwrap();
    val(victim.kind) - val(attacker.kind) / 10
}

// Simple SEE and helpers
fn see(b: &Board, from: u8, to: u8) -> i32 {
    use crate::types::PieceKind::*;
    let val = |k: crate::types::PieceKind| -> i32 {
        match k {
            Pawn => 100,
            Knight => 320,
            Bishop => 330,
            Rook => 500,
            Queen => 900,
            King => 10_000,
        }
    };
    let mut occ = b.pieces;
    let mut side = b.stm;
    let mut gain = [0i32; 32];
    let mut depth = 0usize;

    let Some(mut victim) = b.piece_at(to) else {
        return 0;
    };

    let mut from_sq = from as usize;
    gain[depth] = val(victim.kind);

    loop {
        depth += 1;
        let attacker = occ[from_sq].unwrap();
        occ[to as usize] = Some(attacker);
        occ[from_sq] = None;

        side = if side == Side::White {
            Side::Black
        } else {
            Side::White
        };

        let mut best_from: Option<usize> = None;
        let mut best_kind_val = i32::MAX;

        for sq in 0..64usize {
            if let Some(p) = occ[sq] {
                if p.side != side {
                    continue;
                }
                let f = (sq as u8 % 8) as i32;
                let r = (sq as u8 / 8) as i32;
                let tf = (to % 8) as i32;
                let tr = (to / 8) as i32;
                let df = tf - f;
                let dr = tr - r;

                let attacks = match p.kind {
                    Pawn => {
                        let dir = if p.side == Side::White { 1 } else { -1 };
                        (dr == dir) && (df.abs() == 1)
                    }
                    Knight => (df.abs() == 1 && dr.abs() == 2) || (df.abs() == 2 && dr.abs() == 1),
                    Bishop => {
                        if df.abs() == dr.abs() {
                            ray_clear(&occ, sq, to as usize, df.signum(), dr.signum())
                        } else {
                            false
                        }
                    }
                    Rook => {
                        if df == 0 || dr == 0 {
                            let sx = df.signum();
                            let sy = dr.signum();
                            ray_clear(&occ, sq, to as usize, sx, sy)
                        } else {
                            false
                        }
                    }
                    Queen => {
                        if df.abs() == dr.abs() || df == 0 || dr == 0 {
                            let sx = df.signum();
                            let sy = dr.signum();
                            ray_clear(&occ, sq, to as usize, sx, sy)
                        } else {
                            false
                        }
                    }
                    King => df.abs() <= 1 && dr.abs() <= 1,
                };
                if attacks {
                    let k = val(p.kind);
                    if k < best_kind_val {
                        best_kind_val = k;
                        best_from = Some(sq);
                    }
                }
            }
        }

        if best_from.is_none() {
            break;
        }
        from_sq = best_from.unwrap();
        victim = occ[to as usize].unwrap();
        gain[depth] = val(victim.kind) - gain[depth - 1].max(0);
        if gain[depth] < 0 {
            break;
        }
    }

    let mut n = depth as i32;
    while n > 0 {
        gain[(n - 1) as usize] = -gain[(n - 1) as usize].max(-gain[n as usize]);
        n -= 1;
    }
    gain[0]
}

fn ray_clear(
    occ: &[Option<crate::types::Piece>; 64],
    from: usize,
    to: usize,
    sx: i32,
    sy: i32,
) -> bool {
    let f = (from as u8 % 8) as i32;
    let r = (from as u8 / 8) as i32;
    let tf = (to as u8 % 8) as i32;
    let tr = (to as u8 / 8) as i32;
    let mut cf = f + sx;
    let mut cr = r + sy;
    while cf != tf || cr != tr {
        let idx = (cr * 8 + cf) as usize;
        if occ[idx].is_some() {
            return false;
        }
        cf += sx;
        cr += sy;
    }
    true
}

// --- Evaluation (temporary minimal eval; real eval is elsewhere) ---
fn eval(b: &Board) -> i16 {
    let p = PARAMS.read();
    let pv = p.piece_val;
    let mut s: i32 = 0;
    for i in 0..64u8 {
        if let Some(pc) = b.piece_at(i) {
            let val = match pc.kind {
                PieceKind::Pawn => pv[0] as i32,
                PieceKind::Knight => pv[1] as i32,
                PieceKind::Bishop => pv[2] as i32,
                PieceKind::Rook => pv[3] as i32,
                PieceKind::Queen => pv[4] as i32,
                PieceKind::King => pv[5] as i32,
            };
            s += if pc.side == Side::White { val } else { -val };
        }
    }
    if b.stm == Side::White {
        s as i16
    } else {
        (-s) as i16
    }
}
