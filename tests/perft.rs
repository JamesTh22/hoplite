use hoplite::board::Board;
use hoplite::perft::perft;

#[test]
fn startpos_perft() {
    let b = Board::new_start();
    assert_eq!(perft(&mut b.clone(), 1), 20);
    assert_eq!(perft(&mut b.clone(), 2), 400);
    assert_eq!(perft(&mut b.clone(), 3), 8902);
    assert_eq!(perft(&mut b.clone(), 4), 197281);
}
