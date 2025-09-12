use std::collections::HashMap;
use parking_lot::RwLock;
use lazy_static::lazy_static;
use crate::types::Move;

// Cache entry stores moves and the position hash they were generated for
#[derive(Clone)]
struct CacheEntry {
    hash: u64,
    moves: Vec<Move>,
}

lazy_static! {
    // Cache for legal moves
    static ref LEGAL_MOVES_CACHE: RwLock<HashMap<u64, CacheEntry>> = RwLock::new(HashMap::with_capacity(1 << 16));
    
    // Cache for pseudo-legal moves
    static ref PSEUDO_MOVES_CACHE: RwLock<HashMap<u64, CacheEntry>> = RwLock::new(HashMap::with_capacity(1 << 16));
}

pub fn get_cached_legal_moves(hash: u64) -> Option<Vec<Move>> {
    let cache = LEGAL_MOVES_CACHE.read();
    cache.get(&hash).map(|e| e.moves.clone())
}

pub fn store_legal_moves(hash: u64, moves: Vec<Move>) {
    let mut cache = LEGAL_MOVES_CACHE.write();
    if cache.len() >= 1 << 16 {
        // Clear cache if too large
        cache.clear();
    }
    cache.insert(hash, CacheEntry { hash, moves });
}

pub fn get_cached_pseudo_moves(hash: u64) -> Option<Vec<Move>> {
    let cache = PSEUDO_MOVES_CACHE.read();
    cache.get(&hash).map(|e| e.moves.clone())
}

pub fn store_pseudo_moves(hash: u64, moves: Vec<Move>) {
    let mut cache = PSEUDO_MOVES_CACHE.write();
    if cache.len() >= 1 << 16 {
        // Clear cache if too large
        cache.clear();
    }
    cache.insert(hash, CacheEntry { hash, moves });
}