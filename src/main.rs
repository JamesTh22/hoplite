
mod types;
mod uci;
mod board;
mod movegen;
mod search;
mod perft;
mod zobrist;
mod tt;
mod params;

use crate::uci::Uci;

fn main() {
    let mut uci = Uci::new();
    uci.mainloop();
}
