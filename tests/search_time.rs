use std::sync::Arc;
use std::time::{Duration, Instant};

use hoplite::board::Board;
use hoplite::nnue;
use hoplite::search::Search;

#[test]
fn movetime_with_high_min_depth_returns_quickly() {
    let nnue_path = match std::env::var("HOPLITE_NNUE") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("skipping test: set HOPLITE_NNUE to a valid NNUE file");
            return;
        }
    };
    let network = match nnue::load_nnue(&nnue_path) {
        Ok(net) => Arc::new(net),
        Err(err) => {
            eprintln!(
                "skipping test: failed to load NNUE `{}`: {}",
                nnue_path, err
            );
            return;
        }
    };
    let mut search = Search::new(Some(network)).expect("failed to initialize search");
    search.set_threads(1);
    search.set_min_depth(10);

    let mut board = Board::new_start();
    let start = Instant::now();
    let mv = search.bestmove_time(&mut board, 1);
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(250),
        "search exceeded timeout: {:?}",
        elapsed
    );
    assert!(mv.from != 0 || mv.to != 0, "search failed to return a move");
}
