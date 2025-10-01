use std::time::{Duration, Instant};

use hoplite::board::Board;
use hoplite::search::Search;

#[test]
fn movetime_with_high_min_depth_returns_quickly() {
    let mut search = Search::new();
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
