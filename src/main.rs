mod types;
mod uci;
mod board;
mod movegen;
mod search;
mod eval;
mod zobrist;
mod tt;
mod params;
mod endgame;
mod perft;

use crate::uci::Uci;

fn main() {
    let mut uci = Uci::new();
    uci.mainloop();
}
