use criterion::{criterion_group, criterion_main, Criterion};
use hoplite::board::Board;
use hoplite::perft::perft;

fn perft_bench(c: &mut Criterion) {
    c.bench_function("perft depth 4", |b| {
        b.iter(|| {
            let mut board = Board::new_start();
            perft(&mut board, 4);
        })
    });
}

criterion_group!(benches, perft_bench);
criterion_main!(benches);
