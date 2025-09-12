
use hoplite::board::Board;
use hoplite::search::Search;
use hoplite::params::{PARAMS, Params, save_params_to};
use hoplite::types::Side;
use rand::prelude::*;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle, ParallelProgressIterator};

fn play_game(p_white: &Params, p_black: &Params, seed: u64, movetime_ms: u128, max_plies: usize) -> f32 {
    let mut b = Board::new_start();
    let mut s_white = Search::new();
    let mut s_black = Search::new();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut plies = 0usize;

    loop {
        let side_white = matches!(b.stm, Side::White);
        let use_p = if side_white { p_white } else { p_black }.clone();
        *PARAMS.write() = use_p;

        let mv = if plies < 6 && rng.gen::<f32>() < 0.1 {
            let ms = hoplite::movegen::legal_moves(&b);
            if ms.is_empty() { break; }
            ms[rng.gen::<usize>() % ms.len()]
        } else {
            if side_white { s_white.bestmove_time(&mut b, movetime_ms) } else { s_black.bestmove_time(&mut b, movetime_ms) }
        };

        if mv.from==0 && mv.to==0 { return 0.5; }
        let _u = b.make_move(mv);
        plies += 1;
        if plies >= max_plies { return 0.5; }

        let ms = hoplite::movegen::legal_moves(&b);
        if ms.is_empty() {
            if b.in_check(b.stm) { return if side_white { 1.0 } else { 0.0 }; }
            else { return 0.5; }
        }
    }
    0.5
}

fn elo_from_score(p: f32) -> f32 {
    if p <= 0.0 { return -9999.0; }
    if p >= 1.0 { return  9999.0; }
    400.0 * ((p / (1.0 - p)).log10())
}

fn main() {
    let mut iters: usize = 50;
    let mut games_per_iter: usize = 100;
    let mut movetime_ms: u128 = 20;
    let mut seed: u64 = 42;
    let mut save_every: usize = 5;
    let mut max_plies: usize = 200;
    let mut parallel: usize = num_cpus::get();
    let mut quiet: bool = false;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" if i+1 < args.len() => { iters = args[i+1].parse().unwrap_or(iters); i+=2; }
            "--games" if i+1 < args.len() => { games_per_iter = args[i+1].parse().unwrap_or(games_per_iter); i+=2; }
            "--movetime" if i+1 < args.len() => { movetime_ms = args[i+1].parse().unwrap_or(movetime_ms); i+=2; }
            "--seed" if i+1 < args.len() => { seed = args[i+1].parse().unwrap_or(seed); i+=2; }
            "--save-every" if i+1 < args.len() => { save_every = args[i+1].parse().unwrap_or(save_every); i+=2; }
            "--max-plies" if i+1 < args.len() => { max_plies = args[i+1].parse().unwrap_or(max_plies); i+=2; }
            "--parallel" if i+1 < args.len() => { parallel = args[i+1].parse().unwrap_or(parallel); i+=2; }
            "--quiet" => { quiet = true; i+=1; }
            _ => { i+=1; }
        }
    }
    rayon::ThreadPoolBuilder::new().num_threads(parallel).build_global().ok();

    let mut theta = Params::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let total_games_iter = games_per_iter * 2; eprintln!("tune: iters={iters} games/iter={total_games_iter} movetime={movetime_ms}ms parallel={parallel}");

    for k in 0..iters {
        let mut deltas_val = [0i16;6];
        let mut deltas_scale = [0f32;6];
        for i in 0..6 {
            deltas_val[i] = if rng.gen::<bool>() { 1 } else { -1 };
            deltas_scale[i] = if rng.gen::<bool>() { 1.0 } else { -1.0 };
        }
        let step_val = 5i16.max(1);
        let step_scale = 0.10f32;

        let mut plus = theta.clone();
        let mut minus = theta.clone();
        for i in 0..6 {
            plus.piece_val[i] = (plus.piece_val[i] as i32 + (step_val as i32)* (deltas_val[i] as i32)) as i16;
            minus.piece_val[i] = (minus.piece_val[i] as i32 - (step_val as i32)* (deltas_val[i] as i32)) as i16;
            plus.pst_scale[i] += step_scale * deltas_scale[i];
            minus.pst_scale[i] -= step_scale * deltas_scale[i];
        }

        let seeds: Vec<u64> = (0..games_per_iter).map(|g| (k as u64)*100000 + g as u64).collect();
        let total = (games_per_iter as u64) * 2;
        let pb = if quiet {
            ProgressBar::hidden()
        } else {
            let pb = ProgressBar::new(total);
            let style = ProgressStyle::with_template(
                "[iter {k}/{iters}] {bar:40.cyan/blue} {pos}/{len} games • {percent}% • {elapsed_precise}<{eta_precise} • {per_sec} it/s"
            ).unwrap()
            .progress_chars("##-");
            pb.set_style(style);
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb
        };

        let results: Vec<(f32,f32)> = seeds.par_iter()
            .progress_with(pb.clone())
            .map(|&s| {
                let r1 = play_game(&plus, &minus, s, movetime_ms, max_plies);
                pb.inc(1);
                let r2_white = play_game(&minus, &plus, s+9999, movetime_ms, max_plies);
                pb.inc(1);
                (r1, 1.0 - r2_white)
            }).collect();

        if !quiet { pb.finish_and_clear(); }

        let mut score = 0.0f32;
        for (a,b) in results { score += a + b; }
        let total_games = (games_per_iter as f32) * 2.0;
        let avg = score / total_games;
        let elo = elo_from_score(avg);

        if avg > 0.5 {
            theta = plus;
            eprintln!("[iter {k}] avg={avg:.3} (~{elo:.1} Elo vs θ−)   pick=θ+   piece_val={:?} pst_scale={:?}", theta.piece_val, theta.pst_scale);
        } else {
            theta = minus;
            eprintln!("[iter {k}] avg={avg:.3} (~{elo:.1} Elo vs θ+)   pick=θ−   piece_val={:?} pst_scale={:?}", theta.piece_val, theta.pst_scale);
        }

        if (k+1) % save_every == 0 {
            *PARAMS.write() = theta.clone();
            let _ = save_params_to("params.json");
            eprintln!("checkpoint saved at iter {}", k+1);
        }
    }

    *PARAMS.write() = theta.clone();
    let _ = save_params_to("params.json");
    println!("Saved tuned params to params.json");
}


    // NOTE: The block below was added to perturb extended eval parameters too.
