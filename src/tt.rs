#![allow(dead_code)]

use crate::types::Move;

#[derive(Copy, Clone)]
pub enum Bound { Exact, Lower, Upper }

#[derive(Copy, Clone)]
// prefer deeper
pub struct Entry {
    pub key: u64,
    pub depth: i8,
    pub value: i16,
    pub best: Move,
    pub bound: Bound,
}

pub struct TT {
    table: Vec<Entry>,
    mask: usize,
}

impl TT {
    pub fn new(mb: usize) -> Self {
        let bytes = mb * 1024 * 1024;
        let n = bytes / std::mem::size_of::<Entry>().max(1);
        let cap = n.next_power_of_two().max(1024);
        let table = vec![Entry { key:0, depth:-1, value:0, best:Move::default(), bound: Bound::Exact }; cap];
        Self { table, mask: cap-1 }
    }
    #[inline] pub fn idx(&self, key: u64) -> usize { (key as usize) & self.mask }
    pub fn probe(&self, key: u64) -> Option<Entry> {
        let e = self.table[self.idx(key)];
        if e.key == key && e.depth >= 0 { Some(e) } else { None }
    }
    pub fn store(&mut self, e: Entry) {
        let i = self.idx(e.key);
        // very naive replacement; replace shallow with deeper
        if self.table[i].depth <= e.depth || self.table[i].key == 0 {
            self.table[i] = e;
        }
    }
}
