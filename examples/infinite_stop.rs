use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::Context;
use hoplite::board::Board;
use hoplite::eval::PsqtEvaluator;
use hoplite::nnue;
use hoplite::search::Search;

fn main() -> anyhow::Result<()> {
    let nnue_path = std::env::var("HOPLITE_NNUE")
        .context("set HOPLITE_NNUE to the path of a valid NNUE network file")?;
    let network = nnue::load_nnue(&nnue_path).context("failed to load NNUE network")?;
    let mut search = Search::new(Some(Arc::new(network)))?;
    search.set_threads(1);
    search.set_evaluator(Arc::new(PsqtEvaluator));
    search.set_min_depth(1);

    // A quiet middlegame position that encourages deeper search iterations.
    let mut board = Board::from_fen("4k3/8/8/8/8/8/2Q5/Q3K3 w - - 0 1")?;

    let stop_flag = Arc::clone(&search.stop);
    let stopper_flag = Arc::clone(&stop_flag);
    let stopper = thread::spawn(move || {
        thread::sleep(Duration::from_secs(120));
        println!("Signalling stop flag...");
        stopper_flag.store(true, Ordering::Relaxed);
    });

    let best_move = search.bestmove_infinite(&mut board);
    println!("Search returned {}", best_move.uci());

    stopper.join().expect("stopper thread panicked");
    println!(
        "Stop flag final value: {}",
        stop_flag.load(Ordering::Relaxed)
    );

    Ok(())
}
