
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Side { White, Black }

impl Side {
    #[inline] pub fn flip(self) -> Side { if self == Side::White { Side::Black } else { Side::White } }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum PieceKind { Pawn, Knight, Bishop, Rook, Queen, King }

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Piece { pub side: Side, pub kind: PieceKind }

#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Move {
    pub from: u8, pub to: u8,
    pub promo: Option<PieceKind>,
}

impl Move {
    pub fn uci(self) -> String {
        let ffile = (self.from % 8) as u8;
        let frank = (self.from / 8) as u8;
        let tfile = (self.to % 8) as u8;
        let trank = (self.to / 8) as u8;
        let mut s = String::new();
        s.push((b'a'+ffile) as char);
        s.push((b'1'+frank) as char);
        s.push((b'a'+tfile) as char);
        s.push((b'1'+trank) as char);
        if let Some(pk) = self.promo {
            s.push(match pk {
                PieceKind::Queen => 'q',
                PieceKind::Rook => 'r',
                PieceKind::Bishop => 'b',
                PieceKind::Knight => 'n',
                _ => 'q'
            });
        }
        s
    }
}

#[inline]
pub fn sq(file: i32, rank: i32) -> Option<u8> {
    if file >= 0 && file < 8 && rank >= 0 && rank < 8 {
        Some((rank * 8 + file) as u8)
    } else { None }
}
