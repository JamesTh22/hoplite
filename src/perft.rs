
use crate::board::Board;
use crate::movegen::legal_moves;

pub fn perft(b: &mut Board, depth: u32) -> u64 {
    if depth == 0 { return 1; }
    let moves = legal_moves(b);
    if depth == 1 { return moves.len() as u64; }
    let mut nodes = 0u64;
    for mv in moves {
        let u = b.make_move(mv);
        nodes += perft(b, depth - 1);
        b.unmake_move(mv, u);
    }
    nodes
}
