
# Hoplite Tuning (Rust-only, parallel)

This repo includes a **pure Rust** SPSA-style tuner that can run **10k+ games** across your CPU cores.

## Build
```bash
cargo build --release
```

## Run a big batch (≈10,000 games)
Each iteration plays `2 * --games` games (White/Black swap). So `--iters 50 --games 100` ≈ `50 * 200 = 10,000` games.

```bash
./target/release/tune \
  --iters 50 \
  --games 100 \
  --movetime 20 \
  --parallel 8 \
  --save-every 5
```

- `--movetime` is **per move in ms** (smaller = more games/hour; larger = cleaner signal).
- `--parallel` is thread count (defaults to CPU core count).

Progress and checkpoints are printed to stderr; tuned params are saved to `params.json` at the end and at checkpoints.

## Use tuned params in the engine / Lucas Chess
Place `params.json` next to your engine binary or send UCI:
```
setoption name ParamsFile value params.json
isready
```
The engine also tries to auto-load `params.json` on startup if present.

## Tips
- For early runs, prefer `--movetime 15–30`. For refinement, `40–60` ms.
- If you want to push beyond piece values + PST scales, add more fields to `src/params.rs` and use them in `eval()` (passed pawns, bishop pair, mobility, king safety, etc.). The tuner will optimize them automatically.
- You can run **multiple tuners in parallel** in separate folders with different `--seed` values, then keep the best `params.json`.

### Progress bar & logs
The tuner now shows a per-iteration progress bar with ETA and throughput. Use `--quiet` to disable the bar.
