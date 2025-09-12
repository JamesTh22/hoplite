
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use std::fs;

#[derive(Clone, Serialize, Deserialize)]
pub struct Params {
    // --- Search-independent eval tunables ---

    // Piece values: P, N, B, R, Q, K (K unused)
    pub piece_val: [i16; 6],
    // PST scales per piece (multiplier for baked-in PST tables)
    pub pst_scale: [f32; 6],

    // Tapered eval blending (0..256)
    pub mg_weight: i16, // middlegame weight
    pub eg_weight: i16, // endgame weight

    // Mobility (per move) in centipawns
    pub mobility_knight: i16,
    pub mobility_bishop: i16,
    pub mobility_rook: i16,
    pub mobility_queen: i16,

    // Endgame-ish extras
    pub connected_passers: i16,
    pub rook_behind_passer: i16,

    // King safety ring (penalty per undefended ring square), light
    pub king_ring_penalty: i16,


    // --- Eval pack tunables (centipawns unless noted) ---
    pub bishop_pair: i16,
    pub rook_open_file: i16,
    pub rook_semi_open_file: i16,
    pub isolated_pawn: i16,
    pub doubled_pawn: i16,
    pub passed_pawn: [i16; 8],        // bonus by rank (0..7 from White POV)
    pub king_shield_missing: i16,     // penalty per missing pawn in front of king (3 files, 1 rank ahead)
}
impl Default for Params {
    fn default() -> Self {
        Self {
            piece_val: [100, 320, 330, 500, 900, 0],
            pst_scale: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],

            mg_weight: 192,  // ~75% MG
            eg_weight: 64,   // ~25% EG

            mobility_knight: 2,
            mobility_bishop: 3,
            mobility_rook:   2,
            mobility_queen:  1,

            connected_passers: 20,
            rook_behind_passer: 15,

            king_ring_penalty: 2,


            bishop_pair: 30,
            rook_open_file: 15,
            rook_semi_open_file: 7,
            isolated_pawn: -8,
            doubled_pawn: -12,
            passed_pawn: [0, 0, 10, 20, 35, 60, 90, 0],
            king_shield_missing: -12,
        }
    }
}

lazy_static::lazy_static! {
    pub static ref PARAMS: RwLock<Params> = RwLock::new(Params::default());
}

pub fn load_params_from(path: &str) -> anyhow::Result<()> {
    let data = fs::read_to_string(path)?;
    let p: Params = serde_json::from_str(&data)?;
    *PARAMS.write() = p;
    Ok(())
}

pub fn save_params_to(path: &str) -> anyhow::Result<()> {
    let p = PARAMS.read().clone();
    let s = serde_json::to_string_pretty(&p)?;
    fs::write(path, s)?;
    Ok(())
}
