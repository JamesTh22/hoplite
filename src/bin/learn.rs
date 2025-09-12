use std::{fs, path::PathBuf, io::Read, collections::HashMap};
use hoplite::{board::Board, movegen::legal_moves, types::{Move, Side}};
use serde_json;

fn pack_move16(m: Move) -> u16 {
    ((m.from as u16) << 6) | (m.to as u16) | ((m.promo.unwrap_or(hoplite::types::PieceKind::Pawn) as u16) << 12)
}

fn find_uci(b: &Board, uci: &str) -> Option<Move> {
    let ms = legal_moves(b);
    for m in ms {
        if m.uci() == uci { return Some(m); }
    }
    None
}

fn main() {
    let mut args = std::env::args().skip(1);
    let pgn_path = PathBuf::from(args.next().expect("usage: learn <games.pgn> <exp.json>"));
    let exp_path = PathBuf::from(args.next().unwrap_or_else(|| "hoplite.exp".into()));

    let mut txt = String::new();
    fs::File::open(&pgn_path).expect("open pgn").read_to_string(&mut txt).unwrap();

    let mut exp: HashMap<u128, (u32,u32)> = if let Ok(s) = fs::read_to_string(&exp_path) {
        serde_json::from_str(&s).unwrap_or_default()
    } else { HashMap::new() };

    for game in txt.split("\n\n\n") {
        if game.trim().is_empty() { continue; }
        let result = if let Some(rline) = game.lines().find(|l| l.starts_with("[Result ")) {
            if rline.contains("1-0") { Some(1.0) } else if rline.contains("0-1") { Some(0.0) } else if rline.contains("1/2-1/2") { Some(0.5) } else { None }
        } else { None };
        if result.is_none() { continue; }
        let res = result.unwrap();

        // Extract move section (very loose)
        let movesec = game.lines().filter(|l| !l.starts_with('[')).collect::<Vec<_>>().join(" ");
        let tokens = movesec.split_whitespace().collect::<Vec<_>>();
        let mut b = Board::new_start();
        let mut token_idx = 0usize;
        let mut move_count = 0usize;
        while token_idx < tokens.len() {
            let t = tokens[token_idx];
            // skip move numbers like "12."
            if t.ends_with('.') { token_idx += 1; continue; }
            if t.len()>=4 && t.len()<=5 {
                if let Some(mv) = find_uci(&b, t) {
                    // record ONLY the first root move of the game (side to move)
                    if move_count == 0 {
                        let k: u128 = ((b.key as u128) << 16) | (pack_move16(mv) as u128);
                        let e = exp.entry(k).or_insert((0,0));
                        e.0 += 1;
                        if (b.stm==Side::White && res==1.0) || (b.stm==Side::Black && res==0.0) { e.1 += 1; }
                        // draws add nothing to wins
                    }
                    let _u = b.make_move(mv);
                    move_count += 1;
                }
            }
            token_idx += 1;
        }
    }

    fs::write(&exp_path, serde_json::to_string_pretty(&exp).unwrap()).expect("write exp");
    println!("Updated experience written to {}", exp_path.display());
}
