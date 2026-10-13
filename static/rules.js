/* The few chess rules the move follower needs, written to behave like python-chess (which
   the server version used): legal moves, SAN, FEN, and whether a position is valid. Squares
   are 0..63 from a1 to h8; a board is 64 characters, '.' for an empty square. */
const FILES = 'abcdefgh';
export const squareName = s => FILES[s & 7] + ((s >> 3) + 1);
const fileOf = s => s & 7, rankOf = s => s >> 3;
const isWhite = ch => ch >= 'A' && ch <= 'Z';
const colourOf = ch => ch === '.' ? null : (isWhite(ch) ? 'w' : 'b');
const onBoard = (f, r) => f >= 0 && f < 8 && r >= 0 && r < 8;

const KNIGHT = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
const KING = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
const ROOK_DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const BISHOP_DIRS = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
// where each castling right's rook stands
const CASTLE_ROOK = { K: 7, Q: 0, k: 63, q: 56 };

export class Position {
  constructor(board, turn = 'w', castling = '', ep = -1, half = 0, full = 1){
    this.b = board; this.turn = turn; this.castling = castling; this.ep = ep; this.half = half; this.full = full;
  }

  static fromFen(fen){
    const [placement, turn = 'w', castling = '-', ep = '-', half = '0', full = '1'] = fen.trim().split(/\s+/);
    const b = new Array(64).fill('.');
    const rows = placement.split('/');
    for (let r = 0; r < 8; r++){
      let f = 0;
      for (const ch of rows[r] || ''){
        if (ch >= '1' && ch <= '8') f += +ch;
        else { if (f < 8) b[(7 - r) * 8 + f] = ch; f++; }
      }
    }
    const p = new Position(b, turn === 'b' ? 'b' : 'w', '', ep === '-' ? -1 : (+ep[1] - 1) * 8 + FILES.indexOf(ep[0]),
      +half || 0, +full || 1);
    // like python-chess, keep only the rights that still have their king and rook in place
    for (const c of 'KQkq'){
      if (!castling.includes(c)) continue;
      const white = c === 'K' || c === 'Q';
      if (b[white ? 4 : 60] === (white ? 'K' : 'k') && b[CASTLE_ROOK[c]] === (white ? 'R' : 'r')) p.castling += c;
    }
    return p;
  }

  copy(){ return new Position(this.b.slice(), this.turn, this.castling, this.ep, this.half, this.full); }

  placement(){
    let s = '';
    for (let r = 7; r >= 0; r--){
      let e = 0;
      for (let f = 0; f < 8; f++){
        const c = this.b[r * 8 + f];
        if (c === '.') e++;
        else { if (e){ s += e; e = 0; } s += c; }
      }
      if (e) s += e;
      if (r) s += '/';
    }
    return s;
  }

  fen(){
    // the en passant square is only written when a legal en passant capture exists
    const ep = this.ep >= 0 && this.legalMoves().some(m => this.isEnPassant(m)) ? squareName(this.ep) : '-';
    return `${this.placement()} ${this.turn} ${this.castling || '-'} ${ep} ${this.half} ${this.full}`;
  }

  /* python-chess uses the king on the highest square if there are several */
  king(colour){
    const k = colour === 'w' ? 'K' : 'k';
    for (let s = 63; s >= 0; s--) if (this.b[s] === k) return s;
    return -1;
  }

  attackedBy(colour, sq){
    const b = this.b, f0 = fileOf(sq), r0 = rankOf(sq), w = colour === 'w';
    const pr = r0 + (w ? -1 : 1), pawn = w ? 'P' : 'p';
    for (const df of [-1, 1]) if (onBoard(f0 + df, pr) && b[pr * 8 + f0 + df] === pawn) return true;
    const knight = w ? 'N' : 'n', king = w ? 'K' : 'k';
    for (const [df, dr] of KNIGHT) if (onBoard(f0 + df, r0 + dr) && b[(r0 + dr) * 8 + f0 + df] === knight) return true;
    for (const [df, dr] of KING) if (onBoard(f0 + df, r0 + dr) && b[(r0 + dr) * 8 + f0 + df] === king) return true;
    const rq = w ? 'RQ' : 'rq', bq = w ? 'BQ' : 'bq';
    for (const [dirs, who] of [[ROOK_DIRS, rq], [BISHOP_DIRS, bq]]){
      for (const [df, dr] of dirs){
        let f = f0 + df, r = r0 + dr;
        while (onBoard(f, r)){
          const c = b[r * 8 + f];
          if (c !== '.'){ if (who.includes(c)) return true; break; }
          f += df; r += dr;
        }
      }
    }
    return false;
  }

  checkers(){
    const k = this.king(this.turn);
    if (k < 0) return [];
    const them = this.turn === 'w' ? 'b' : 'w', out = [];
    for (let s = 0; s < 64; s++){
      if (colourOf(this.b[s]) !== them) continue;
      // does the piece on s attack k? test it alone on an otherwise identical board
      if (this.pieceAttacks(s, k)) out.push(s);
    }
    return out;
  }

  pieceAttacks(s, t){
    const ch = this.b[s], p = ch.toLowerCase(), df = fileOf(t) - fileOf(s), dr = rankOf(t) - rankOf(s);
    if (p === 'p') return Math.abs(df) === 1 && dr === (isWhite(ch) ? 1 : -1);
    if (p === 'n') return (Math.abs(df) === 1 && Math.abs(dr) === 2) || (Math.abs(df) === 2 && Math.abs(dr) === 1);
    if (p === 'k') return Math.max(Math.abs(df), Math.abs(dr)) === 1;
    const straight = df === 0 || dr === 0, diag = Math.abs(df) === Math.abs(dr);
    if (!(p === 'q' ? straight || diag : p === 'r' ? straight : diag) || (df === 0 && dr === 0)) return false;
    const sf = Math.sign(df), sr = Math.sign(dr);
    for (let f = fileOf(s) + sf, r = rankOf(s) + sr; f !== fileOf(t) || r !== rankOf(t); f += sf, r += sr)
      if (this.b[r * 8 + f] !== '.') return false;
    return true;
  }

  isCheck(){
    const k = this.king(this.turn);
    return k >= 0 && this.attackedBy(this.turn === 'w' ? 'b' : 'w', k);
  }

  pseudoMoves(){
    const b = this.b, us = this.turn, them = us === 'w' ? 'b' : 'w', out = [];
    const slide = (s, dirs, far) => {
      for (const [df, dr] of dirs){
        let f = fileOf(s) + df, r = rankOf(s) + dr;
        while (onBoard(f, r)){
          const c = b[r * 8 + f];
          if (colourOf(c) !== us) out.push({ from: s, to: r * 8 + f });
          if (c !== '.' || !far) break;
          f += df; r += dr;
        }
      }
    };
    for (let s = 63; s >= 0; s--){
      const ch = b[s];
      if (colourOf(ch) !== us) continue;
      const p = ch.toLowerCase();
      if (p === 'n') slide(s, KNIGHT, false);
      else if (p === 'k') slide(s, KING, false);
      else if (p === 'r') slide(s, ROOK_DIRS, true);
      else if (p === 'b') slide(s, BISHOP_DIRS, true);
      else if (p === 'q') slide(s, [...ROOK_DIRS, ...BISHOP_DIRS], true);
    }
    // castling: king and rook home (kept so by the rights), nothing between them, and the
    // king neither in check nor crossing or landing on an attacked square
    for (const c of us === 'w' ? 'KQ' : 'kq'){
      if (!this.castling.includes(c)) continue;
      const k = us === 'w' ? 4 : 60, rook = CASTLE_ROOK[c], side = rook > k ? 1 : -1;
      let clear = true;
      for (let s = k + side; s !== rook; s += side) if (b[s] !== '.') clear = false;
      if (!clear) continue;
      if ([k, k + side, k + 2 * side].some(s => this.attackedBy(them, s))) continue;
      out.push({ from: k, to: k + 2 * side });
    }
    const fwd = us === 'w' ? 8 : -8, last = us === 'w' ? 7 : 0;
    const pawn = us === 'w' ? 'P' : 'p';
    const add = (from, to) => {
      if (rankOf(to) === last) for (const pr of 'qrbn') out.push({ from, to, promotion: pr });
      else out.push({ from, to });
    };
    for (let s = 63; s >= 0; s--){
      if (b[s] !== pawn) continue;
      for (const df of [-1, 1]){
        const f = fileOf(s) + df, r = rankOf(s) + fwd / 8;
        if (onBoard(f, r) && colourOf(b[r * 8 + f]) === them) add(s, r * 8 + f);
      }
    }
    for (let s = 63; s >= 0; s--){
      if (b[s] !== pawn) continue;
      const t = s + fwd;
      if (t < 0 || t > 63 || b[t] !== '.') continue;
      add(s, t);
      // like python-chess: a second step that lands on the 3rd/4th rank (6th/5th for Black)
      const t2 = t + fwd;
      if (t2 >= 0 && t2 < 64 && b[t2] === '.' && Math.abs(rankOf(t2) - (us === 'w' ? 2.5 : 4.5)) < 1) out.push({ from: s, to: t2 });
    }
    if (this.ep >= 0 && b[this.ep] === '.'){
      for (const df of [-1, 1]){
        const f = fileOf(this.ep) + df, r = rankOf(this.ep) - fwd / 8;
        if (onBoard(f, r) && r === (us === 'w' ? 4 : 3) && b[r * 8 + f] === pawn) out.push({ from: r * 8 + f, to: this.ep });
      }
    }
    return out;
  }

  legalMoves(){
    if (this._legal) return this._legal;
    let moves = this.pseudoMoves();
    const k = this.king(this.turn);
    // without a king of the side to move, python-chess allows every pseudo-legal move
    if (k < 0) return (this._legal = moves);
    const them = this.turn === 'w' ? 'b' : 'w', checkers = this.checkers();
    if (checkers.length){
      // as python-chess: in check, only that king moves, or other pieces (no king) capture or
      // block a single checker
      const target = new Set();
      if (checkers.length === 1){
        const c = checkers[0], sf = Math.sign(fileOf(c) - fileOf(k)), sr = Math.sign(rankOf(c) - rankOf(k));
        target.add(c);
        if ('qrbQRB'.includes(this.b[c]))
          for (let f = fileOf(k) + sf, r = rankOf(k) + sr; f !== fileOf(c) || r !== rankOf(c); f += sf, r += sr) target.add(r * 8 + f);
      }
      moves = moves.filter(m => m.from === k ? !this.isCastling(m)
        : checkers.length === 1 && this.b[m.from].toLowerCase() !== 'k' &&
          (target.has(m.to) || (this.isEnPassant(m) && m.to + (this.turn === 'w' ? -8 : 8) === checkers[0])));
    }
    return (this._legal = moves.filter(m => (m.from === k && this.isCastling(m)) ||
      !this.push(m, true).attackedBy(them, m.from === k ? m.to : k)));
  }

  isEnPassant(m){
    return m.to === this.ep && this.b[m.from].toLowerCase() === 'p' && fileOf(m.from) !== fileOf(m.to) && this.b[m.to] === '.';
  }

  isCastling(m){
    return this.b[m.from].toLowerCase() === 'k' && Math.abs(fileOf(m.to) - fileOf(m.from)) === 2;
  }

  /* a new position after m (quick: skips the bookkeeping only a legality test doesn't need) */
  push(m, quick){
    const n = quick ? new Position(this.b.slice(), this.turn) : this.copy();
    const b = n.b, ch = b[m.from], p = ch.toLowerCase(), capture = b[m.to] !== '.' || this.isEnPassant(m);
    if (this.isEnPassant(m)) b[m.to + (this.turn === 'w' ? -8 : 8)] = '.';
    if (this.isCastling(m)){
      const rookFrom = m.to > m.from ? m.from + 3 : m.from - 4, rookTo = m.to > m.from ? m.from + 1 : m.from - 1;
      b[rookTo] = b[rookFrom]; b[rookFrom] = '.';
    }
    b[m.to] = m.promotion ? (this.turn === 'w' ? m.promotion.toUpperCase() : m.promotion) : ch;
    b[m.from] = '.';
    n.turn = this.turn === 'w' ? 'b' : 'w';
    if (quick) return n;
    n.ep = p === 'p' && Math.abs(m.to - m.from) === 16 && rankOf(m.from) === (this.turn === 'w' ? 1 : 6) ? (m.from + m.to) / 2 : -1;
    n.half = p === 'p' || capture ? 0 : this.half + 1;
    if (this.turn === 'b') n.full = this.full + 1;
    let rights = '';
    for (const c of this.castling){
      const rook = CASTLE_ROOK[c], k = c === 'K' || c === 'Q' ? 4 : 60;
      if (m.from === rook || m.to === rook) continue;
      if (p === 'k' && m.from === k && colourOf(ch) === (k === 4 ? 'w' : 'b')) continue;
      rights += c;
    }
    n.castling = rights;
    return n;
  }

  san(m){
    const b = this.b, ch = b[m.from], p = ch.toLowerCase();
    let san;
    if (this.isCastling(m)) san = m.to > m.from ? 'O-O' : 'O-O-O';
    else {
      const capture = b[m.to] !== '.' || this.isEnPassant(m);
      if (p === 'p'){
        san = capture ? FILES[fileOf(m.from)] : '';
      } else {
        san = ch.toUpperCase();
        let others = [];
        for (const o of this.legalMoves()){
          if (o.to === m.to && o.from !== m.from && b[o.from] === ch) others.push(o.from);
        }
        if (others.length){
          let row = false, column = false;
          if (others.some(s => rankOf(s) === rankOf(m.from))) column = true;
          if (others.some(s => fileOf(s) === fileOf(m.from))) row = true;
          else column = true;
          if (column) san += FILES[fileOf(m.from)];
          if (row) san += rankOf(m.from) + 1;
        }
      }
      if (capture) san += 'x';
      san += squareName(m.to);
      if (m.promotion) san += '=' + m.promotion.toUpperCase();
    }
    const n = this.push(m);
    if (n.isCheck()) san += n.legalMoves().length ? '+' : '#';
    return san;
  }

  /* python-chess's Board.is_valid() for the positions the follower can make */
  isValid(){
    const b = this.b, count = ch => b.filter(c => c === ch).length;
    const W = b.filter(c => colourOf(c) === 'w').length, B = b.filter(c => colourOf(c) === 'b').length;
    if (!W && !B) return false;
    if (count('K') < 1 || count('k') < 1 || count('K') + count('k') > 2) return false;
    if (count('P') > 8 || count('p') > 8 || W > 16 || B > 16) return false;
    for (let f = 0; f < 8; f++) if ('Pp'.includes(b[f]) || 'Pp'.includes(b[56 + f])) return false;
    const them = this.turn === 'w' ? 'b' : 'w', theirKing = this.king(them);
    if (theirKing >= 0 && this.attackedBy(this.turn, theirKing)) return false;
    if (this.ep >= 0) return false;  // the follower never asks with one set
    const ch = this.checkers();
    if (ch.length > 2) return false;
    if (ch.length === 2){
      // two checkers on one line through our king can't both have just started giving check
      const k = this.king(this.turn), [a, c] = ch;
      const line = (x, y) => {
        const df = fileOf(y) - fileOf(x), dr = rankOf(y) - rankOf(x);
        return df === 0 || dr === 0 || Math.abs(df) === Math.abs(dr);
      };
      if (line(a, c) && line(a, k) && line(c, k)){
        const df1 = fileOf(c) - fileOf(a), dr1 = rankOf(c) - rankOf(a), df2 = fileOf(k) - fileOf(a), dr2 = rankOf(k) - rankOf(a);
        if (df1 * dr2 === df2 * dr1) return false;
      }
    }
    return true;
  }
}

export const sameMove = (a, b) => a.from === b.from && a.to === b.to && (a.promotion || null) === (b.promotion || null);
