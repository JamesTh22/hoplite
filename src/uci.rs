use crate::board::Board;
use crate::eval::{default_evaluator, nnue_evaluator};
use crate::nnue;
use crate::search::Search;
use crate::types::Move;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::{atomic::Ordering, Arc};

pub struct Uci {
    board: Board,
    search: Search,
    search_thread: Option<std::thread::JoinHandle<()>>,
    eval_dir: PathBuf,
    auto_load_attempted: bool,
}

impl Uci {
    pub fn new() -> Self {
        Self {
            board: Board::new_start(),
            search: Search::new(),
            search_thread: None,
            eval_dir: PathBuf::from("NNUE"),
            auto_load_attempted: false,
        }
    }

    fn ensure_eval_dir(&self) -> io::Result<()> {
        if self.eval_dir.as_os_str().is_empty() {
            return Ok(());
        }
        fs::create_dir_all(&self.eval_dir)
    }

    fn nnue_files(&self) -> io::Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        if self.eval_dir.as_os_str().is_empty() {
            return Ok(files);
        }
        let entries = match fs::read_dir(&self.eval_dir) {
            Ok(iter) => iter,
            Err(e) => return Err(e),
        };
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ext.eq_ignore_ascii_case("nnue") {
                    files.push(path);
                }
            }
        }
        files.sort_by(|a, b| {
            let an = a
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            let bn = b
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            an.cmp(&bn)
        });
        Ok(files)
    }

    fn resolve_nnue_path(&self, value: &str) -> PathBuf {
        let candidate = Path::new(value);
        if candidate.is_absolute() || value.contains('/') || value.contains("\\") {
            candidate.to_path_buf()
        } else {
            self.eval_dir.join(candidate)
        }
    }

    fn load_nnue_from_path(&mut self, path: &Path) -> Result<String, String> {
        match nnue::load_nnue(path) {
            Ok(net) => {
                let summary = net.summary();
                let net = Arc::new(net);
                self.search.set_evaluator(nnue_evaluator(net));
                Ok(summary)
            }
            Err(e) => Err(e.to_string()),
        }
    }

    fn auto_load_default_nnue(&mut self) -> Vec<String> {
        if self.auto_load_attempted {
            return Vec::new();
        }
        self.auto_load_attempted = true;
        let mut messages = Vec::new();
        if let Err(e) = self.ensure_eval_dir() {
            messages.push(format!(
                "info string failed to prepare NNUE directory `{}`: {}",
                self.eval_dir.display(),
                e
            ));
            self.auto_load_attempted = false;
            return messages;
        }
        let files = match self.nnue_files() {
            Ok(list) => list,
            Err(e) => {
                messages.push(format!(
                    "info string failed to enumerate NNUE directory `{}`: {}",
                    self.eval_dir.display(),
                    e
                ));
                self.auto_load_attempted = false;
                return messages;
            }
        };
        if files.is_empty() {
            messages.push(format!(
                "info string no NNUE networks found in `{}`",
                self.eval_dir.display()
            ));
            self.auto_load_attempted = false;
            return messages;
        }
        let path = &files[0];
        match self.load_nnue_from_path(path) {
            Ok(summary) => messages.push(format!(
                "info string auto-loaded NNUE `{}` {}",
                path.display(),
                summary
            )),
            Err(e) => {
                messages.push(format!(
                    "info string failed to auto-load NNUE `{}`: {}",
                    path.display(),
                    e
                ));
                self.auto_load_attempted = false;
            }
        }
        messages
    }

    pub fn mainloop(&mut self) {
        let stdin = io::stdin();
        let mut out = io::stdout();
        for line in stdin.lock().lines() {
            let Ok(line) = line else { continue };
            let line = line.trim().to_string();
            if line == "uci" {
                writeln!(out, "id name Hoplite 0.1.0").unwrap();
                writeln!(out, "id author you").unwrap();
                writeln!(
                    out,
                    "option name Hash type spin default 256 min 1 max 16384"
                )
                .unwrap();
                writeln!(out, "option name Threads type spin default 1 min 1 max 64").unwrap();
                writeln!(
                    out,
                    "option name MoveOverhead type spin default 15 min 0 max 5000"
                )
                .unwrap();
                writeln!(out, "option name MultiPV type spin default 1 min 1 max 10").unwrap();
                writeln!(out, "option name LearnEnable type check default false").unwrap();
                writeln!(
                    out,
                    "option name LearnStrength type spin default 40 min 0 max 100"
                )
                .unwrap();
                writeln!(out, "option name LearnFile type string default hoplite.exp").unwrap();
                writeln!(
                    out,
                    "option name ParamsFile type string default params.json"
                )
                .unwrap();
                writeln!(out, "option name EvalFile type string default").unwrap();
                writeln!(out, "option name EvalDirectory type string default NNUE").unwrap();
                writeln!(
                    out,
                    "option name MinDepth type spin default 20 min 1 max 64"
                )
                .unwrap();
                for msg in self.auto_load_default_nnue() {
                    writeln!(out, "{}", msg).unwrap();
                }
                match self.nnue_files() {
                    Ok(files) if !files.is_empty() => {
                        let listing = files
                            .iter()
                            .filter_map(|p| p.file_name().and_then(|s| s.to_str()))
                            .collect::<Vec<_>>()
                            .join(", ");
                        writeln!(
                            out,
                            "info string NNUE files in `{}`: {}",
                            self.eval_dir.display(),
                            listing
                        )
                        .unwrap();
                    }
                    Ok(_) => {
                        writeln!(
                            out,
                            "info string drop NNUE networks into `{}` to enable NNUE",
                            self.eval_dir.display()
                        )
                        .unwrap();
                    }
                    Err(e) => {
                        writeln!(
                            out,
                            "info string failed to inspect NNUE directory `{}`: {}",
                            self.eval_dir.display(),
                            e
                        )
                        .unwrap();
                    }
                }
                writeln!(out, "uciok").unwrap();
            } else if line.starts_with("setoption") {
                // setoption name <Name> value <Val>
                let mut name = String::new();
                let mut val = String::new();
                let mut it = line.split_whitespace();
                it.next(); // setoption
                if it.next() == Some("name") {
                    while let Some(tok) = it.next() {
                        if tok == "value" {
                            break;
                        }
                        if !name.is_empty() {
                            name.push(' ');
                        }
                        name.push_str(tok);
                    }
                    val = it.collect::<Vec<_>>().join(" ");
                }
                if name.eq_ignore_ascii_case("Hash") {
                    if let Ok(mb) = val.trim().parse::<usize>() {
                        self.search.set_hash_mb(mb);
                    }
                } else if name.eq_ignore_ascii_case("Threads") {
                    if let Ok(n) = val.trim().parse::<usize>() {
                        self.search.set_threads(n);
                    }
                } else if name.eq_ignore_ascii_case("MultiPV") {
                    if let Ok(n) = val.trim().parse::<usize>() {
                        self.search.multipv = n.max(1);
                    }
                } else if name.eq_ignore_ascii_case("ParamsFile") {
                    if !val.trim().is_empty() {
                        let _ = crate::params::load_params_from(val.trim());
                    }
                } else if name.eq_ignore_ascii_case("MoveOverhead") {
                    if let Ok(ms) = val.trim().parse::<u64>() {
                        self.search.move_overhead_ms = ms;
                    }
                } else if name.eq_ignore_ascii_case("LearnEnable") {
                    let v = val.trim().eq_ignore_ascii_case("true") || val.trim() == "1";
                    self.search.exp_enabled = v;
                } else if name.eq_ignore_ascii_case("LearnStrength") {
                    if let Ok(s) = val.trim().parse::<i32>() {
                        self.search.exp_strength = s.clamp(0, 100);
                    }
                } else if name.eq_ignore_ascii_case("LearnFile") {
                    if !val.trim().is_empty() {
                        self.search.exp_path = Some(val.trim().to_string());
                        if let Ok(s) = std::fs::read_to_string(val.trim()) {
                            let map: HashMap<u128, (u32, u32)> =
                                serde_json::from_str(&s).unwrap_or_default();
                            self.search.exp_table = map;
                        }
                    }
                } else if name.eq_ignore_ascii_case("EvalFile") {
                    let trimmed = val.trim();
                    if trimmed.is_empty() {
                        self.search.set_evaluator(default_evaluator());
                        writeln!(out, "info string switched to PSQT evaluator").unwrap();
                    } else {
                        let path = self.resolve_nnue_path(trimmed);
                        match self.load_nnue_from_path(&path) {
                            Ok(summary) => {
                                writeln!(
                                    out,
                                    "info string loaded NNUE `{}` {}",
                                    path.display(),
                                    summary
                                )
                                .unwrap();
                                self.auto_load_attempted = true;
                            }
                            Err(e) => {
                                writeln!(
                                    out,
                                    "info string failed to load NNUE `{}`: {}",
                                    path.display(),
                                    e
                                )
                                .unwrap();
                                self.search.set_evaluator(default_evaluator());
                            }
                        }
                    }
                } else if name.eq_ignore_ascii_case("EvalDirectory") {
                    let trimmed = val.trim();
                    self.eval_dir = if trimmed.is_empty() {
                        PathBuf::from("NNUE")
                    } else {
                        PathBuf::from(trimmed)
                    };
                    self.auto_load_attempted = false;
                    for msg in self.auto_load_default_nnue() {
                        writeln!(out, "{}", msg).unwrap();
                    }
                    match self.nnue_files() {
                        Ok(files) if !files.is_empty() => {
                            let listing = files
                                .iter()
                                .filter_map(|p| p.file_name().and_then(|s| s.to_str()))
                                .collect::<Vec<_>>()
                                .join(", ");
                            writeln!(
                                out,
                                "info string NNUE files in `{}`: {}",
                                self.eval_dir.display(),
                                listing
                            )
                            .unwrap();
                        }
                        Ok(_) => {
                            writeln!(
                                out,
                                "info string drop NNUE networks into `{}` to enable NNUE",
                                self.eval_dir.display()
                            )
                            .unwrap();
                        }
                        Err(e) => {
                            writeln!(
                                out,
                                "info string failed to inspect NNUE directory `{}`: {}",
                                self.eval_dir.display(),
                                e
                            )
                            .unwrap();
                        }
                    }
                } else if name.eq_ignore_ascii_case("MinDepth") {
                    if let Ok(d) = val.trim().parse::<i32>() {
                        self.search.set_min_depth(d);
                        writeln!(
                            out,
                            "info string minimum search depth set to {}",
                            self.search.min_depth
                        )
                        .unwrap();
                    }
                }
            } else if line == "saveparams" {
                let _ = crate::params::save_params_to("params.json");

                // setoption name <Name> value <Val>
                let mut name = String::new();
                let mut val = String::new();
                let mut it = line.split_whitespace();
                it.next(); // setoption
                if it.next() == Some("name") {
                    while let Some(tok) = it.next() {
                        if tok == "value" {
                            break;
                        }
                        if !name.is_empty() {
                            name.push(' ');
                        }
                        name.push_str(tok);
                    }
                    val = it.collect::<Vec<_>>().join(" ");
                }
                if name.eq_ignore_ascii_case("Hash") {
                    if let Ok(mb) = val.trim().parse::<usize>() {
                        self.search.set_hash_mb(mb);
                    }
                } else if name.eq_ignore_ascii_case("Threads") {
                    if let Ok(n) = val.trim().parse::<usize>() {
                        self.search.set_threads(n);
                    }
                } else if name.eq_ignore_ascii_case("MultiPV") {
                    if let Ok(n) = val.trim().parse::<usize>() {
                        self.search.multipv = n.max(1);
                    }
                }
            } else if line == "isready" {
                writeln!(out, "readyok").unwrap();
            } else if line.starts_with("ucinewgame") {
                self.board = Board::new_start();
            } else if line.starts_with("position") {
                self.handle_position(&line);
            } else if line == "d" {
                let fen = self.board.to_fen();
                writeln!(out, "info string FEN {}", fen).unwrap();
            } else if line.starts_with("perft") {
                let depth = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(4);
                let mut bclone = self.board.clone();
                let nodes = crate::perft::perft(&mut bclone, depth);
                writeln!(out, "info string perft({}) = {}", depth, nodes).unwrap();
            } else if line.starts_with("go") {
                // Stop any previous search
                if let Some(handle) = self.search_thread.take() {
                    self.search.stop.store(true, Ordering::Relaxed);
                    let _ = handle.join();
                }
                // Parse time controls
                let mut depth: Option<i32> = None;
                let mut movetime: Option<u128> = None;
                let mut wtime: Option<u128> = None;
                let mut btime: Option<u128> = None;
                let mut winc: u128 = 0;
                let mut binc: u128 = 0;
                let mut movestogo: Option<u128> = None;
                let mut multipv: Option<usize> = None;
                let mut infinite = false;

                let toks: Vec<&str> = line.split_whitespace().collect();
                let mut i = 1;
                while i < toks.len() {
                    match toks[i] {
                        "depth" => {
                            if i + 1 < toks.len() {
                                depth = toks[i + 1].parse::<i32>().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        }
                        "movetime" => {
                            if i + 1 < toks.len() {
                                movetime = toks[i + 1].parse::<u128>().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        }
                        "wtime" => {
                            if i + 1 < toks.len() {
                                wtime = toks[i + 1].parse::<u128>().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        }
                        "btime" => {
                            if i + 1 < toks.len() {
                                btime = toks[i + 1].parse::<u128>().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        }
                        "winc" => {
                            if i + 1 < toks.len() {
                                winc = toks[i + 1].parse::<u128>().unwrap_or(0);
                                i += 2;
                            } else {
                                i += 1;
                            }
                        }
                        "binc" => {
                            if i + 1 < toks.len() {
                                binc = toks[i + 1].parse::<u128>().unwrap_or(0);
                                i += 2;
                            } else {
                                i += 1;
                            }
                        }
                        "movestogo" => {
                            if i + 1 < toks.len() {
                                movestogo = toks[i + 1].parse::<u128>().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        }
                        "multipv" => {
                            if i + 1 < toks.len() {
                                multipv = toks[i + 1].parse::<usize>().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        }
                        "infinite" => {
                            infinite = true;
                            i += 1;
                        }
                        _ => {
                            i += 1;
                        }
                    }
                }
                if infinite {
                    self.search.stop.store(false, Ordering::Relaxed);
                    let mut s = self.search.clone();
                    if let Some(mv) = multipv {
                        s.multipv = mv.max(1);
                    }
                    let mut b = self.board.clone();
                    self.search_thread = Some(std::thread::spawn(move || {
                        let best = s.bestmove_infinite(&mut b);
                        let mv = if best.from == 0 && best.to == 0 {
                            "0000".to_string()
                        } else {
                            best.uci()
                        };
                        println!("bestmove {}", mv);
                    }));
                } else if let Some(d) = depth {
                    let prev = self.search.multipv;
                    if let Some(mv) = multipv {
                        self.search.multipv = mv.max(1);
                    }
                    let target_depth = d.max(self.search.min_depth);
                    if target_depth > d {
                        writeln!(
                            out,
                            "info string depth {} raised to minimum {}",
                            d, target_depth
                        )
                        .unwrap();
                    }
                    let best = self.search.bestmove(&mut self.board, target_depth);
                    self.search.multipv = prev;
                    let mv = if best.from == 0 && best.to == 0 {
                        "0000".to_string()
                    } else {
                        best.uci()
                    };
                    writeln!(out, "bestmove {}", mv).unwrap();
                } else {
                    // compute time budget
                    let stm_white = matches!(self.board.stm, crate::types::Side::White);
                    let overhead = self.search.move_overhead_ms as u128;
                    let budget = if let Some(mt) = movetime {
                        mt.saturating_sub(overhead)
                    } else if stm_white {
                        compute_budget(wtime.unwrap_or(1000), winc, movestogo, overhead)
                    } else {
                        compute_budget(btime.unwrap_or(1000), binc, movestogo, overhead)
                    };
                    let prev = self.search.multipv;
                    if let Some(mv) = multipv {
                        self.search.multipv = mv.max(1);
                    }
                    let best = self.search.bestmove_time(&mut self.board, budget);
                    self.search.multipv = prev;
                    let mv = if best.from == 0 && best.to == 0 {
                        "0000".to_string()
                    } else {
                        best.uci()
                    };
                    writeln!(out, "bestmove {}", mv).unwrap();
                }
            } else if line == "quit" {
                self.search.stop.store(true, Ordering::Relaxed);
                if let Some(handle) = self.search_thread.take() {
                    let _ = handle.join();
                }
                break;
            } else if line == "stop" {
                self.search.stop.store(true, Ordering::Relaxed);
                if let Some(handle) = self.search_thread.take() {
                    let _ = handle.join();
                }
            }
        }
    }

    fn handle_position(&mut self, cmd: &str) {
        // position startpos [moves ...]
        // position fen <fen...> [moves ...]
        let mut toks = cmd.split_whitespace();
        toks.next();
        match toks.next() {
            Some("startpos") => {
                self.board = Board::new_start();
                if let Some("moves") = toks.next() {
                    for m in toks {
                        let mv = self.parse_uci_move(m);
                        self.play_if_legal(mv);
                    }
                }
            }
            Some("fen") => {
                let fen_parts: Vec<String> = toks.by_ref().take(6).map(|s| s.to_string()).collect();
                if fen_parts.len() == 6 {
                    if let Ok(b) = Board::from_fen(&fen_parts.join(" ")) {
                        self.board = b;
                    }
                }
                let rest: Vec<String> = toks.map(|s| s.to_string()).collect();
                if rest.first().map(String::as_str) == Some("moves") {
                    for m in &rest[1..] {
                        let mv = self.parse_uci_move(m);
                        self.play_if_legal(mv);
                    }
                }
            }
            _ => {}
        }
    }

    fn parse_uci_move(&self, s: &str) -> Move {
        if s.len() < 4 {
            return Move::default();
        }
        let b = s.as_bytes();
        let ff = (b[0] - b'a') as u8;
        let fr = (b[1] - b'1') as u8;
        let tf = (b[2] - b'a') as u8;
        let tr = (b[3] - b'1') as u8;
        let from = fr * 8 + ff;
        let to = tr * 8 + tf;
        let promo = if s.len() == 5 {
            Some(match b[4] as char {
                'q' | 'Q' => crate::types::PieceKind::Queen,
                'r' | 'R' => crate::types::PieceKind::Rook,
                'b' | 'B' => crate::types::PieceKind::Bishop,
                'n' | 'N' => crate::types::PieceKind::Knight,
                _ => crate::types::PieceKind::Queen,
            })
        } else {
            None
        };
        Move { from, to, promo }
    }

    fn play_if_legal(&mut self, mv: Move) {
        // naive: generate legal and check membership
        let legal = crate::movegen::legal_moves(&self.board);
        if legal
            .iter()
            .any(|m| m.from == mv.from && m.to == mv.to && m.promo == mv.promo)
        {
            let _u = self.board.make_move(mv);
        }
    }
}

fn compute_budget(time_ms: u128, inc_ms: u128, movestogo: Option<u128>, overhead_ms: u128) -> u128 {
    // Simple allocator: keep a safety reserve, use increment, divide by moves to go (default 30)
    let mtg = movestogo.unwrap_or(30);
    let reserve = (time_ms as f64 * 0.05) as u128; // keep 5%
    let usable = time_ms.saturating_sub(reserve) + (inc_ms * 8 / 10); // 80% of inc
    let slice = (usable / mtg.max(1)).max(20); // floor 20ms
    let cap = time_ms / 2; // never spend more than half
    slice.min(cap).max(20).saturating_sub(overhead_ms).max(1)
}
