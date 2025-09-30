use crate::board::Board;
use crate::eval::{default_evaluator, EvalState, Evaluator};
use crate::movegen::{legal_moves, square_attacked};
use crate::tt::{Bound, Entry, TT};
use crate::types::{Move, PieceKind, Side};
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
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
    pub stop: Arc<AtomicBool>,
    pub threads: usize,
    pub deadline: Option<Instant>,

    // Multi-PV
    pub multipv: usize,

    // Move ordering
    pub killers: [[Move; 2]; MAX_PLY], // two killer moves per ply
    pub history: [[i32; 4096]; 2],     // side, from*64+to
    pub evaluator: Arc<dyn Evaluator + Send + Sync>,
}

impl Search {
    // Helper method to check if last move was a capture
    fn last_was_capture(&self, history: &[u64]) -> bool {
        if history.len() < 2 {
            return false;
        }
        // If Zobrist hash changed by more than a move and side-to-move, likely a capture
        let xor = history[history.len() - 1] ^ history[history.len() - 2];
        xor.count_ones() > 2
    }

    pub fn new() -> Self {
        Self {
            nodes: 0,
            tt: Arc::new(Mutex::new(TT::new(64))),
            stop: Arc::new(AtomicBool::new(false)),
            threads: 1,
            deadline: None,
            pawn_tt: HashMap::with_capacity(1 << 16),
            exp_table: HashMap::with_capacity(1 << 14),
            exp_enabled: false,
            exp_strength: 40,
            exp_path: None,
            move_overhead_ms: 15,
            multipv: 1,
            killers: [[Move::default(); 2]; MAX_PLY],
            history: [[0; 4096]; 2],
            evaluator: default_evaluator(),
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

    pub fn set_evaluator(&mut self, evaluator: Arc<dyn Evaluator>) {
        self.evaluator = evaluator;
    }

    #[inline]
    fn time_up(&self) -> bool {
        if let Some(d) = self.deadline {
            Instant::now() >= d
        } else {
            false
        }
    }

    fn get_pv(&self, b: &mut Board, max_depth: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        for _ in 0..max_depth {
            let mv = {
                let tt = self.tt.lock();
                tt.probe(b.key).map(|e| e.best).unwrap_or_default()
            };
            if mv.from == 0 && mv.to == 0 {
                break;
            }
            let legal = legal_moves(b);
            if !legal
                .iter()
                .any(|m| m.from == mv.from && m.to == mv.to && m.promo == mv.promo)
            {
                break;
            }
            let _u = b.make_move(mv);
            pv.push(mv);
        }
        pv
    }

    // Depth-limited
    pub fn bestmove(&mut self, b: &mut Board, depth: i32) -> Move {
        let mut history_keys: Vec<u64> = vec![b.key];
        let mut eval_state = self.evaluator.init_state(b);
        self.nodes = 0;
        self.stop.store(false, Ordering::Relaxed);
        self.deadline = None;
        let mut best = Move::default();
        let mut last_score: i16 = 0;
        let mut stable_count = 0; // Track how many times the best move remains stable
        let start = Instant::now();

        for d in 1..=depth {
            let pvs = self.search_root(
                b,
                d,
                &mut history_keys,
                last_score,
                self.multipv,
                &mut eval_state,
            );
            if pvs.is_empty() {
                break;
            }
            let new_score = pvs[0].1;
            let new_best = pvs[0].0[0];

            // Check if the best move is stable across iterations
            if d > 1 {
                if new_best.from == best.from && new_best.to == best.to {
                    stable_count += 1;
                    // If move is stable for 2 iterations and we're past depth 6, return early
                    if stable_count >= 2 && d >= 6 {
                        break;
                    }
                } else {
                    stable_count = 0;
                }
            }

            last_score = new_score;
            best = new_best;
            let elapsed = start.elapsed().as_millis();
            if self.multipv > 1 {
                for (idx, (pv, sc)) in pvs.iter().enumerate() {
                    let pv_str = pv.iter().map(|m| m.uci()).collect::<Vec<_>>().join(" ");
                    println!(
                        "info depth {} multipv {} score cp {} time {} nodes {} pv {}",
                        d,
                        idx + 1,
                        sc,
                        elapsed,
                        self.nodes,
                        pv_str
                    );
                }
            } else {
                let pv_str = pvs[0]
                    .0
                    .iter()
                    .map(|m| m.uci())
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "info depth {} score cp {} time {} nodes {} pv {}",
                    d, pvs[0].1, elapsed, self.nodes, pv_str
                );
            }
            let _ = io::stdout().flush();
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
        }
        best
    }

    // Time-limited
    // Dynamic time management based on position complexity and game phase
    fn allocate_time(&self, total_time_ms: u128, b: &mut Board) -> u128 {
        // Count material to estimate game phase (0-256)
        let mut phase = 0;
        let mut total_pieces = 0;
        for i in 0..64u8 {
            if let Some(pc) = b.piece_at(i) {
                total_pieces += 1;
                match pc.kind {
                    PieceKind::Pawn => phase += 0,
                    PieceKind::Knight | PieceKind::Bishop => phase += 8,
                    PieceKind::Rook => phase += 16,
                    PieceKind::Queen => phase += 32,
                    PieceKind::King => phase += 0,
                }
            }
        }
        phase = phase.min(256);

        // Estimate position complexity (0-100)
        let mut complexity = 0;

        // More pieces = more complex
        complexity += (total_pieces * 2).min(30);

        // Check for tactical elements
        let moves = legal_moves(b);
        for mv in moves.iter() {
            if b.piece_at(mv.to).is_some() {
                // Captures available
                complexity += 5;
            }
            let u = b.make_move(*mv);
            if b.in_check(b.stm) {
                // Checks available
                complexity += 10;
            }
            b.unmake_move(*mv, u);
        }

        // Kings under attack need more time
        if square_attacked(b, king_square(b, b.stm), b.stm.flip()) {
            complexity += 20;
        }

        complexity = complexity.min(100);

        // Adjust time based on phase and complexity
        let mut allocated = total_time_ms;

        // Use more time in complex middlegame positions
        if phase > 64 && phase < 192 {
            allocated = allocated * (100 + complexity as u128) / 100;
        }

        // Use less time in simple endgames
        if phase <= 64 {
            allocated = allocated * 70 / 100;
        }

        // Ensure reasonable bounds
        allocated = allocated.clamp(total_time_ms / 10, total_time_ms * 2);

        allocated
    }

    pub fn bestmove_time(&mut self, b: &mut Board, time_ms: u128) -> Move {
        let mut history_keys: Vec<u64> = vec![b.key];
        let mut eval_state = self.evaluator.init_state(b);
        self.nodes = 0;
        self.stop.store(false, Ordering::Relaxed);

        // Get dynamically allocated time based on position
        let allocated_time = self.allocate_time(time_ms, b);
        self.deadline = Some(Instant::now() + Duration::from_millis(allocated_time as u64));

        let mut best = Move::default();
        let mut last_score: i16 = 0;
        let mut stable_count = 0; // Track stability
        let start = Instant::now();

        for d in 1..=64 {
            let pvs = self.search_root(
                b,
                d,
                &mut history_keys,
                last_score,
                self.multipv,
                &mut eval_state,
            );
            if pvs.is_empty() {
                break;
            }
            let new_score = pvs[0].1;
            let new_best = pvs[0].0[0];

            // Check if the best move is stable across iterations
            if d > 1 {
                if new_best.from == best.from && new_best.to == best.to {
                    stable_count += 1;
                    // If move is stable for 2 iterations and we're past depth 6, return early
                    if stable_count >= 2 && d >= 6 {
                        break;
                    }
                } else {
                    stable_count = 0;
                }
            }

            last_score = new_score;
            best = new_best;
            let elapsed = start.elapsed().as_millis();
            if self.multipv > 1 {
                for (idx, (pv, sc)) in pvs.iter().enumerate() {
                    let pv_str = pv.iter().map(|m| m.uci()).collect::<Vec<_>>().join(" ");
                    println!(
                        "info depth {} multipv {} score cp {} time {} nodes {} pv {}",
                        d,
                        idx + 1,
                        sc,
                        elapsed,
                        self.nodes,
                        pv_str
                    );
                }
            } else {
                let pv_str = pvs[0]
                    .0
                    .iter()
                    .map(|m| m.uci())
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "info depth {} score cp {} time {} nodes {} pv {}",
                    d, pvs[0].1, elapsed, self.nodes, pv_str
                );
            }
            let _ = io::stdout().flush();
            if self.time_up() {
                self.stop.store(true, Ordering::Relaxed);
                break;
            }
        }
        if let Some(deadline) = self.deadline {
            let now = Instant::now();
            if now < deadline {
                std::thread::sleep(deadline - now);
            }
        }
        best
    }

    // Infinite search until stop flag is set
    pub fn bestmove_infinite(&mut self, b: &mut Board) -> Move {
        let mut history_keys: Vec<u64> = vec![b.key];
        let mut eval_state = self.evaluator.init_state(b);
        self.nodes = 0;
        self.stop.store(false, Ordering::Relaxed);
        self.deadline = None;
        let mut best = Move::default();
        let mut last_score: i16 = 0;
        let mut stable_count = 0; // Track stability
        let start = Instant::now();

        for d in 1..=64 {
            let pvs = self.search_root(
                b,
                d,
                &mut history_keys,
                last_score,
                self.multipv,
                &mut eval_state,
            );
            if pvs.is_empty() {
                break;
            }
            let new_score = pvs[0].1;
            let new_best = pvs[0].0[0];

            // Check if the best move is stable across iterations
            if d > 1 {
                if new_best.from == best.from && new_best.to == best.to {
                    stable_count += 1;
                    // If move is stable for 2 iterations and we're past depth 6, return early
                    if stable_count >= 2 && d >= 6 {
                        break;
                    }
                } else {
                    stable_count = 0;
                }
            }

            last_score = new_score;
            best = new_best;
            let elapsed = start.elapsed().as_millis();
            if self.multipv > 1 {
                for (idx, (pv, sc)) in pvs.iter().enumerate() {
                    let pv_str = pv.iter().map(|m| m.uci()).collect::<Vec<_>>().join(" ");
                    println!(
                        "info depth {} multipv {} score cp {} time {} nodes {} pv {}",
                        d,
                        idx + 1,
                        sc,
                        elapsed,
                        self.nodes,
                        pv_str
                    );
                }
            } else {
                let pv_str = pvs[0]
                    .0
                    .iter()
                    .map(|m| m.uci())
                    .collect::<Vec<_>>()
                    .join(" ");
                println!(
                    "info depth {} score cp {} time {} nodes {} pv {}",
                    d, pvs[0].1, elapsed, self.nodes, pv_str
                );
            }
            let _ = io::stdout().flush();
            if self.stop.load(Ordering::Relaxed) {
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
        multipv: usize,
        eval_state: &mut EvalState,
    ) -> Vec<(Vec<Move>, i16)> {
        const ASPIRATION_WINDOW: i16 = 50; // Initial window size
        const ASPIRATION_DELTA: i16 = 25; // Window growth increment
        let moves = legal_moves(b);
        if moves.is_empty() {
            return Vec::new();
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
                    let mut state = s.evaluator.init_state(&bb);
                    let is_capture = bb.piece_at(mv.to).is_some();
                    let undo = bb.make_move(*mv);
                    s.evaluator.push_state(&bb, &mut state, *mv, &undo);
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
                        &mut state,
                    );
                    s.evaluator.pop_state(&mut state);
                    bb.unmake_move(*mv, undo);
                    (*mv, sc, s.nodes)
                })
                .collect();
            self.nodes += results.iter().map(|(_, _, n)| *n).sum::<u64>();
            let mut results: Vec<(Move, i16)> =
                results.into_iter().map(|(m, s, _)| (m, s)).collect();
            results.sort_by_key(|(_, s)| -*s);
            if self.time_up() {
                self.stop.store(true, Ordering::Relaxed);
            }
            let mut pvs = Vec::new();
            for (mv, sc) in results.iter().take(multipv) {
                let mut bb = b.clone();
                let _ = bb.make_move(*mv);
                let mut pv = vec![*mv];
                pv.extend(self.get_pv(&mut bb, depth.max(0) as usize));
                pvs.push((pv, *sc));
            }
            return pvs;
        }

        let mut alpha = prev_score.saturating_sub(50);
        let mut beta = prev_score.saturating_add(50);
        let (orig_a, orig_b) = (alpha, beta);

        let mut results: Vec<(Move, i16)> = Vec::new();
        let mut widened_low = false;
        'asp: loop {
            results.clear();
            for (_, mv) in scored.clone() {
                if self.time_up() {
                    self.stop.store(true, Ordering::Relaxed);
                    break;
                }
                let u = b.make_move(mv);
                self.evaluator.push_state(b, eval_state, mv, &u);
                let is_capture = b.piece_at(mv.to).is_some();
                let mut new_depth = depth - 1;

                // Enhanced positional extensions
                let gave_check = b.in_check(b.stm);
                let is_pawn_push = if let Some(p) = b.piece_at(mv.from) {
                    matches!(p.kind, PieceKind::Pawn) && (mv.to / 8 == 6 || mv.to / 8 == 1)
                // 7th/2nd rank
                } else {
                    false
                };
                let is_recapture = is_capture && self.last_was_capture(history_keys);

                // Extend search in critical positions
                if gave_check {
                    new_depth += 1;
                }
                if is_pawn_push {
                    new_depth += 1;
                }
                if is_recapture {
                    new_depth += 1;
                }

                // But limit total extension
                new_depth = new_depth.min(depth + 2); // LMR
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
                let sc = -self.alphabeta(
                    b,
                    new_depth,
                    -beta,
                    -alpha,
                    1,
                    false,
                    history_keys,
                    eval_state,
                );
                self.evaluator.pop_state(eval_state);
                b.unmake_move(mv, u);
                results.push((mv, sc));
                if sc > alpha {
                    alpha = sc;
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

        results.sort_by_key(|(_, s)| -*s);
        let mut pvs = Vec::new();
        for (mv, sc) in results.iter().take(multipv) {
            let mut bb = b.clone();
            let _ = bb.make_move(*mv);
            let mut pv = vec![*mv];
            pv.extend(self.get_pv(&mut bb, depth.max(0) as usize));
            pvs.push((pv, *sc));
        }
        pvs
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
        eval_state: &mut EvalState,
    ) -> i16 {
        self.nodes += 1;
        if (self.nodes & 0x1FFF) == 0 && self.time_up() {
            self.stop.store(true, Ordering::Relaxed);
            return alpha;
        }
        if is_draw(history_keys, b.key, b.halfmove) {
            return 0;
        }

        let in_check = b.in_check(b.stm);

        // Static eval pruning
        if depth <= 0 {
            return self.qsearch(b, alpha, beta, history_keys, eval_state);
        }

        // Eval pruning / futility pruning
        if !in_check && depth <= 3 {
            let eval = self.evaluator.eval(b, eval_state);

            // Reverse futility pruning
            if depth <= 3 && eval >= beta + futility_margin(depth) {
                return eval;
            }

            // Futility pruning
            if depth <= 2 {
                let margin = futility_margin(depth);
                if eval + margin <= alpha {
                    let qscore = self.qsearch(b, alpha, beta, history_keys, eval_state);
                    if qscore <= alpha {
                        return qscore;
                    }
                }
            }
        }

        // Transposition table lookup
        let mut tt_entry = {
            let tt = self.tt.lock();
            tt.probe(b.key)
        };

        // Internal Iterative Deepening when we have no TT move
        if depth >= 4 && !tt_entry.is_some() {
            let iid_depth = depth - 2;
            self.alphabeta(
                b,
                iid_depth,
                alpha,
                beta,
                ply,
                false,
                history_keys,
                eval_state,
            );
            let tt = self.tt.lock();
            tt_entry = tt.probe(b.key);
        }

        // Use TT entry if we found one
        if let Some(e) = tt_entry {
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
                eval_state,
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
        let mut best_move = Move::default();
        let orig_alpha = alpha;
        // Order moves
        let mut scored = self.score_moves(b, &moves, ply, None);
        scored.sort_by_key(|(s, _)| -*s);

        for (idx, (_, mv)) in scored.into_iter().enumerate() {
            let u = b.make_move(mv);
            self.evaluator.push_state(b, eval_state, mv, &u);
            let is_capture = b.piece_at(mv.to).is_some();
            let mut new_depth = depth - 1;
            let gave_check = b.in_check(b.stm);
            if gave_check {
                new_depth += 1;
            }
            // Enhanced Late Move Reductions (LMR)
            if depth >= 3 && !is_capture && !gave_check {
                let d = depth as i32;
                let m = (idx as i32) + 1;

                // Dynamic base reduction
                let mut r = if idx >= 3 {
                    // More aggressive reduction for later moves
                    let base = ((d / 2) + (m / 3)) as f32;
                    // Scale reduction based on move history
                    let side_idx = if b.stm == Side::White { 0 } else { 1 };
                    let hist_idx = (mv.from as usize) * 64 + (mv.to as usize);
                    let hist_score = self.history[side_idx][hist_idx] as f32;
                    let scale = 1.0 - (hist_score / 16000.0).min(0.7); // Cap history impact
                    (base * scale).round() as i32 + 1
                } else {
                    // Light reduction for early moves
                    1 + (d / 4)
                };

                // Adjust based on move characteristics
                if let Some(p) = b.piece_at(mv.from) {
                    match p.kind {
                        PieceKind::Knight | PieceKind::Bishop if idx < 8 => {
                            // Less reduction for developing moves early
                            r = r.saturating_sub(1);
                        }
                        PieceKind::Pawn => {
                            // Less reduction for pawn moves to 6th/7th rank
                            let to_rank = mv.to / 8;
                            if (p.side == Side::White && to_rank >= 5)
                                || (p.side == Side::Black && to_rank <= 2)
                            {
                                r = r.saturating_sub(1);
                            }
                        }
                        _ => {}
                    }
                }

                // History-based adjustments
                let side_idx = if b.stm == Side::White { 0 } else { 1 };
                let hist_idx = (mv.from as usize) * 64 + (mv.to as usize);
                if self.history[side_idx][hist_idx] > 8000 {
                    r = r.saturating_sub(1);
                }

                // Clamp reduction
                r = r.clamp(1, d - 1);
                new_depth -= r;

                // Ensure minimum search depth
                new_depth = new_depth.max(1);
            }

            history_keys.push(b.key);
            let sc = -self.alphabeta(
                b,
                new_depth,
                -beta,
                -alpha,
                ply + 1,
                false,
                history_keys,
                eval_state,
            );
            history_keys.pop();
            self.evaluator.pop_state(eval_state);
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
                {
                    let mut tt = self.tt.lock();
                    tt.store(Entry {
                        key: b.key,
                        depth: (depth as i8),
                        value: beta,
                        bound: Bound::Lower,
                        best: mv,
                    });
                }
                return beta;
            }
            if sc > value {
                value = sc;
                best_move = mv;
            }
            if sc > alpha {
                alpha = sc;
            }
        }

        // Store in TT
        let bound = if value <= orig_alpha {
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
                best: best_move,
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
        eval_state: &mut EvalState,
    ) -> i16 {
        self.nodes += 1;
        if (self.nodes & 0x1FFF) == 0 && self.time_up() {
            self.stop.store(true, Ordering::Relaxed);
            return alpha;
        }
        if is_draw(history_keys, b.key, b.halfmove) {
            return 0;
        }

        let stand_pat = self.evaluator.eval(b, eval_state);
        if stand_pat >= beta {
            return beta;
        }

        // Delta pruning - don't look for captures if we're too far behind
        const DELTA_MARGIN: i16 = 900; // Queen value
        if stand_pat + DELTA_MARGIN < alpha {
            return alpha;
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let moves = legal_moves(b);
        let mut captures: Vec<(i32, Move)> = Vec::new();
        let mut checks: Vec<Move> = Vec::new();
        let mut dangerous_moves: Vec<Move> = Vec::new(); // For potential threats

        for mv in moves {
            if b.piece_at(mv.to).is_some() {
                let see_score = see(b, mv.from, mv.to);
                if see_score >= 0 {
                    // Only consider positive or equal captures
                    captures.push((score_capture(b, mv) + see_score as i32, mv));
                }
            } else {
                let u = b.make_move(mv);

                // Check for discovered attacks on king or valuable pieces
                if b.in_check(b.stm) {
                    checks.push(mv);
                } else {
                    // Look for moves that attack valuable pieces
                    for sq in 0..64u8 {
                        if let Some(pc) = b.piece_at(sq) {
                            if pc.side == b.stm
                                && matches!(pc.kind, PieceKind::Queen | PieceKind::Rook)
                            {
                                if square_attacked(b, sq, b.stm.flip()) {
                                    dangerous_moves.push(mv);
                                    break;
                                }
                            }
                        }
                    }
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
            self.evaluator.push_state(b, eval_state, mv, &u);
            let score = -self.qsearch(b, -beta, -alpha, history_keys, eval_state);
            self.evaluator.pop_state(eval_state);
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
            self.evaluator.push_state(b, eval_state, mv, &u);
            let score = -self.qsearch(b, -beta, -alpha, history_keys, eval_state);
            self.evaluator.pop_state(eval_state);
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
        b: &mut Board,
        moves: &[Move],
        ply: usize,
        tt_move: Option<Move>,
    ) -> Vec<(i32, Move)> {
        let mut v: Vec<(i32, Move)> = Vec::with_capacity(moves.len());
        // Pre-calculate king and queen positions for move targeting bonus
        let mut king_sqs = [None::<u8>; 2];
        let mut queen_sqs = [Vec::new(), Vec::new()];
        for i in 0..64u8 {
            if let Some(pc) = b.piece_at(i) {
                let side_idx = if pc.side == Side::White { 0 } else { 1 };
                match pc.kind {
                    PieceKind::King => king_sqs[side_idx] = Some(i),
                    PieceKind::Queen => queen_sqs[side_idx].push(i),
                    _ => {}
                }
            }
        }

        for &mv in moves {
            let mut s: i32 = 0;

            // TT move gets highest priority
            if let Some(tmv) = tt_move {
                if tmv.from == mv.from && tmv.to == mv.to && tmv.promo == mv.promo {
                    s += 2_000_000; // Higher bonus for TT moves
                }
            }

            let side_idx = if b.stm == Side::White { 0 } else { 1 };

            // Captures scored by MVV/LVA
            if let Some(victim) = b.piece_at(mv.to) {
                if see(b, mv.from, mv.to) >= 0 {
                    let see_score = see(b, mv.from, mv.to);
                    s += 1_000_000 + see_score + score_capture(b, mv);

                    // Bonus for capturing with lesser pieces
                    if let Some(attacker) = b.piece_at(mv.from) {
                        if Self::piece_value(attacker.kind) < Self::piece_value(victim.kind) {
                            s += 5000;
                        }
                    }
                } else {
                    s -= 1_000_000; // Heavily penalize losing captures
                }
            } else {
                // Enhanced scoring for non-captures
                let u = b.make_move(mv);

                // Check detection with bonus scaling
                if b.in_check(b.stm) {
                    s += 500_000; // Base bonus for checks

                    // Additional bonus for discovered checks
                    let from_pc = b.piece_at(mv.from);
                    if from_pc.map_or(false, |p| {
                        !matches!(
                            p.kind,
                            PieceKind::Queen | PieceKind::Rook | PieceKind::Bishop
                        )
                    }) {
                        s += 100_000; // Extra bonus for discovered checks
                    }
                }

                // Threat detection
                let enemy_side = b.stm;
                if let Some(enemy_king) = king_sqs[enemy_side as usize] {
                    // Distance to enemy king (closer moves get bonus)
                    let file_dist = (file_of(mv.to) as i32 - file_of(enemy_king) as i32).abs();
                    let rank_dist = (rank_of(mv.to) as i32 - rank_of(enemy_king) as i32).abs();
                    let king_dist = file_dist.max(rank_dist);
                    s += (8 - king_dist) * 2000;
                }

                // Control of critical squares
                let is_central =
                    (mv.to % 8 >= 2 && mv.to % 8 <= 5) && (mv.to / 8 >= 2 && mv.to / 8 <= 5);
                if is_central {
                    s += 3000;
                }

                b.unmake_move(mv, u);

                // Killers get good bonus but below good captures
                if self.killers[ply][0].from == mv.from && self.killers[ply][0].to == mv.to {
                    s += 100_000;
                }
                if self.killers[ply][1].from == mv.from && self.killers[ply][1].to == mv.to {
                    s += 80_000;
                }

                // History score
                let idx = (mv.from as usize) * 64 + (mv.to as usize);
                s += self.history[side_idx][idx];

                // Positional bonuses
                if let Some(p) = b.piece_at(mv.from) {
                    match p.kind {
                        PieceKind::Pawn => {
                            // Bonus for pawn advances
                            let to_rank = mv.to / 8;
                            if p.side == Side::White {
                                s += (to_rank as i32) * 1000;
                            } else {
                                s += (7 - to_rank as i32) * 1000;
                            }
                        }
                        PieceKind::Knight | PieceKind::Bishop => {
                            // Development bonus in opening
                            if ply < 20 {
                                s += 2000;
                            }
                        }
                        _ => {}
                    }
                }
            }
            v.push((s, mv));
        }
        v
    }

    fn piece_value(kind: PieceKind) -> i32 {
        use PieceKind::*;
        match kind {
            Pawn => 100,
            Knight => 320,
            Bishop => 330,
            Rook => 500,
            Queen => 900,
            King => 10000,
        }
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

// Static Exchange Evaluation
fn see(b: &Board, from: u8, to: u8) -> i32 {
    use crate::types::PieceKind::*;
    let val = |k: crate::types::PieceKind| -> i32 {
        // More precise SEE values - slightly lower than actual piece values
        // to avoid overvaluing captures
        match k {
            Pawn => 95,    // Conservative pawn value
            Knight => 310, // Knights slightly undervalued in SEE
            Bishop => 320, // Bishops slightly undervalued in SEE
            Rook => 475,   // Rooks undervalued to prevent bad rook trades
            Queen => 875,  // Queens slightly undervalued to prevent bad queen trades
            King => 9900,  // High but not infinite to detect king captures
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

// Helper function to find king's square for a given side
fn king_square(b: &Board, side: Side) -> u8 {
    for i in 0..64u8 {
        if let Some(pc) = b.piece_at(i) {
            if pc.side == side && matches!(pc.kind, PieceKind::King) {
                return i;
            }
        }
    }
    64
}

#[inline]
fn file_of(sq: u8) -> usize {
    (sq as usize) & 7
}

#[inline]
fn rank_of(sq: u8) -> usize {
    (sq as usize) >> 3
}

#[inline]
fn futility_margin(depth: i32) -> i16 {
    let base = match depth {
        1 => 125, // Slightly higher than pawn value
        2 => 350, // Higher than minor piece
        3 => 550, // Higher than rook value
        _ => 0,
    };

    // Scale margin up in late game positions
    let phase_bonus = if depth > 1 { depth as i16 * 25 } else { 0 };
    base + phase_bonus
}
