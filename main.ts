// sokoban-solver: a push-optimal Sokoban solver for make.fun's sokoban-solver-01.
//
// How it works
// ------------
// The levels this competition draws are a hallway with one to four small rooms
// off it, each room holding 3 or 4 boxes and as many goals, and a box pushed out
// of a room's door into the hallway can never be pushed back. So every box stays
// in its room, each room is an independent puzzle, and a level's push-optimal
// count is the sum of its rooms' optima. This program exploits that:
//
//   1. It splits a level into its rooms (connected components of floor off the
//      hallway row), finds each room's door, and solves each room on its own.
//   2. Each room is solved push-optimally with A* over pushes: a state is the
//      sorted box squares plus the player's square; the player is canonicalised
//      to the reachable region they stand in, so equivalent player squares
//      collapse into one state.
//   3. The heuristic is the exact minimum-cost matching of boxes to goals over
//      shortest-path distances (admissible: every push moves one box one square),
//      computed by subset DP and cached per box layout.
//   4. Dead states are pruned with static dead squares, 2x2 block deadlocks and
//      a frozen-box fixpoint (a box that can never move again, off its goal).
//   5. Room solutions are joined by plain walks through the hallway (boxes are
//      obstacles while walking, so no box is ever pushed between rooms).
//
// The search is bounded by work, never by the clock: expansion budgets are fixed
// counts, so two runs write the same bytes. No randomness, no clock reads.
//
// Input:  $MAKEFUN_INPUT_DIR/<id>/level.txt (default /input), ids i001..i005.
// Output: $MAKEFUN_OUTPUT_DIR/<id>/solution.txt (default /output), the player's
// moves in LURD (lowercase a step, uppercase a push), one file per instance.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Expansion budgets (counts of expanded states, not time: deterministic).
const ROOM_BUDGET = 2_000_000;
const GLOBAL_BUDGET = 10_000_000;

// Directions: [dx, dy, step letter, push letter].
const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// ---------------------------------------------------------------------------
// Level parsing
// ---------------------------------------------------------------------------

function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// ---------------------------------------------------------------------------
// Small deterministic helpers
// ---------------------------------------------------------------------------

// Breadth-first distances from `sources` over cells where ok[c] is true.
// Unreachable cells stay -1.
function bfsDist(w, ok, sources) {
  const dist = new Int32Array(ok.length).fill(-1);
  const q = [];
  for (const s of sources) {
    if (ok[s] && dist[s] < 0) {
      dist[s] = 0;
      q.push(s);
    }
  }
  for (let head = 0; head < q.length; head++) {
    const c = q[head];
    const d = dist[c] + 1;
    const x = c % w;
    if (x > 0 && ok[c - 1] && dist[c - 1] < 0) { dist[c - 1] = d; q.push(c - 1); }
    if (x < w - 1 && ok[c + 1] && dist[c + 1] < 0) { dist[c + 1] = d; q.push(c + 1); }
    if (c - w >= 0 && ok[c - w] && dist[c - w] < 0) { dist[c - w] = d; q.push(c - w); }
    if (c + w < ok.length && ok[c + w] && dist[c + w] < 0) { dist[c + w] = d; q.push(c + w); }
  }
  return dist;
}

// A walk from `from` to `to` over cells where ok[c] is true, treating the
// cells in `boxes` as obstacles. Returns the step letters, or null when the
// destination cannot be reached. Neighbours are tried in u, d, l, r order, so
// the shortest path found is deterministic.
function walkPath(w, ok, boxes, from, to) {
  if (from === to) return '';
  const blocked = new Uint8Array(ok.length);
  for (const b of boxes) blocked[b] = 1;
  const prev = new Int32Array(ok.length).fill(-1);
  const prevDir = new Int8Array(ok.length).fill(-1);
  const q = [from];
  prev[from] = from;
  for (let head = 0; head < q.length; head++) {
    const c = q[head];
    for (let d = 0; d < 4; d++) {
      const [dx, dy] = DIRS[d];
      const n = c + dy * w + dx;
      if (n < 0 || n >= ok.length) continue;
      if (!ok[n] || blocked[n] || prev[n] >= 0) continue;
      // Do not wrap around the board's left/right edges.
      if (dx !== 0 && Math.floor(n / w) !== Math.floor(c / w)) continue;
      prev[n] = c;
      prevDir[n] = d;
      if (n === to) {
        let moves = '';
        for (let t = to; t !== from; t = prev[t]) moves = DIRS[prevDir[t]][2] + moves;
        return moves;
      }
      q.push(n);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Push-optimal A* for one enclosed puzzle (a room, or a whole level)
// ---------------------------------------------------------------------------

// cfg: w, wall, boxOK (cells a box may stand on), walkOK (cells the player may
// stand on), boxes (sorted), goals, player. Returns { pushes, moves, boxes }
// with the final box layout, or null when the budget runs out or no solution
// exists. `budget` is a shared { left } counter object, so a whole run's work
// is bounded deterministically.
function solvePushes(cfg, budget) {
  const { w, wall, boxOK, walkOK, goals, player } = cfg;
  const boxes0 = cfg.boxes.slice().sort((a, b) => a - b);
  const size = w * Math.ceil(boxOK.length / w);
  const n = boxes0.length;
  if (n === 0) return { pushes: 0, moves: '', boxes: [] };
  if (goals.length !== n) return null;

  const goalIdx = new Int16Array(size).fill(-1);
  goals.forEach((g, i) => { goalIdx[g] = i; });

  // Shortest-path distance from each goal over box-occupiable cells.
  const dist = goals.map((g) => bfsDist(w, boxOK, [g]));

  // Static dead squares: box-occupiable, not a goal, and no goal reachable.
  const dead = new Uint8Array(size);
  for (let c = 0; c < size; c++) {
    if (!boxOK[c] || goalIdx[c] >= 0) continue;
    let reach = false;
    for (let gi = 0; gi < n; gi++) {
      if (dist[gi][c] >= 0) { reach = true; break; }
    }
    if (!reach) dead[c] = 1;
  }

  // 2x2 blocks whose four cells are all wall or box-occupiable: a block full of
  // wall-or-box holding a box off its goal can never be undone.
  const quads = [];
  const h = Math.ceil(size / w);
  for (let y = 0; y + 1 < h; y++) {
    for (let x = 0; x + 1 < w; x++) {
      const a = y * w + x;
      const cells = [a, a + 1, a + w, a + w + 1];
      if (cells.every((c) => wall[c] || boxOK[c])) quads.push(cells);
    }
  }

  // Exact minimum-cost matching of boxes to goals (subset DP). Infinity when
  // no box can reach a distinct goal: the layout is unsolvable.
  const INF = 1e9;
  const popcnt = new Uint8Array(1 << n);
  for (let m = 1; m < 1 << n; m++) popcnt[m] = popcnt[m >> 1] + (m & 1);
  const matchCache = new Map();
  function matchingCost(boxes) {
    const key = boxes.join(',');
    const hit = matchCache.get(key);
    if (hit !== undefined) return hit;
    const dp = new Float64Array(1 << n).fill(INF);
    dp[0] = 0;
    for (let mask = 0; mask < (1 << n); mask++) {
      const k = popcnt[mask];
      if (k >= n || dp[mask] >= INF) continue;
      for (let j = 0; j < n; j++) {
        if (mask & (1 << j)) continue;
        const d = dist[j][boxes[k]];
        if (d < 0) continue;
        const nm = mask | (1 << j);
        const v = dp[mask] + d;
        if (v < dp[nm]) dp[nm] = v;
      }
    }
    const cost = dp[(1 << n) - 1];
    matchCache.set(key, cost);
    return cost;
  }

  // Occupancy stamps, reused so no per-state allocation happens here.
  const occMark = new Uint32Array(size);
  let occStamp = 0;
  const regionMark = new Uint32Array(size);
  let regionStamp = 0;

  // The player's reachable region from `from` with `boxes` as obstacles.
  // Returns the region's smallest cell (its canonical representative).
  function regionRep(from, boxes) {
    occStamp++;
    for (const b of boxes) occMark[b] = occStamp;
    regionStamp++;
    let rep = from;
    const stack = [from];
    regionMark[from] = regionStamp;
    while (stack.length > 0) {
      const c = stack.pop();
      if (c < rep) rep = c;
      const x = c % w;
      if (x > 0) { const q = c - 1; if (walkOK[q] && !wall[q] && occMark[q] !== occStamp && regionMark[q] !== regionStamp) { regionMark[q] = regionStamp; stack.push(q); } }
      if (x < w - 1) { const q = c + 1; if (walkOK[q] && !wall[q] && occMark[q] !== occStamp && regionMark[q] !== regionStamp) { regionMark[q] = regionStamp; stack.push(q); } }
      if (c - w >= 0) { const q = c - w; if (walkOK[q] && !wall[q] && occMark[q] !== occStamp && regionMark[q] !== regionStamp) { regionMark[q] = regionStamp; stack.push(q); } }
      if (c + w < size) { const q = c + w; if (walkOK[q] && !wall[q] && occMark[q] !== occStamp && regionMark[q] !== regionStamp) { regionMark[q] = regionStamp; stack.push(q); } }
    }
    return rep;
  }

  // Whether `boxes` holds a box off its goal inside a full 2x2 wall/box block.
  // (Uses its own occupancy stamps: the expansion loop keeps its own.)
  const blockMark = new Uint32Array(size);
  let blockStamp = 0;
  function blockDead(boxes) {
    blockStamp++;
    for (const b of boxes) blockMark[b] = blockStamp;
    for (const q of quads) {
      let boxCount = 0;
      let offGoal = false;
      let full = true;
      for (const c of q) {
        if (blockMark[c] === blockStamp) {
          boxCount++;
          if (goalIdx[c] < 0) offGoal = true;
        } else if (!wall[c]) {
          full = false;
          break;
        }
      }
      if (full && boxCount > 0 && offGoal) return true;
    }
    return false;
  }

  // Frozen-box deadlock: a box that can never move again. A direction is
  // blocked when the destination or the stance cell is a wall or a box already
  // known frozen; the test iterates to a fixpoint. Sound: walls and frozen
  // boxes never move, so a box blocked in every direction by them is stuck.
  function frozenDead(boxes) {
    const frozen = new Uint8Array(n);
    const at = new Map();
    boxes.forEach((b, i) => at.set(b, i));
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < n; i++) {
        if (frozen[i]) continue;
        const b = boxes[i];
        let stuck = true;
        for (let d = 0; d < 4; d++) {
          const [dx, dy] = DIRS[d];
          const dest = b + dy * w + dx;
          const stance = b - dy * w - dx;
          const blockedDest = wall[dest] || (at.has(dest) && frozen[at.get(dest)]);
          const blockedStance = wall[stance] || (at.has(stance) && frozen[at.get(stance)]);
          if (!blockedDest && !blockedStance) { stuck = false; break; }
        }
        if (stuck) { frozen[i] = 1; changed = true; }
      }
    }
    for (let i = 0; i < n; i++) {
      if (frozen[i] && goalIdx[boxes[i]] < 0) return true;
    }
    return false;
  }

  function layoutDead(boxes) {
    for (const b of boxes) if (dead[b]) return true;
    if (blockDead(boxes)) return true;
    if (frozenDead(boxes)) return true;
    return false;
  }

  // --- A* over pushes ---
  const sBoxes = [boxes0];
  const sPlayer = [player];
  const sG = [0];
  const sParent = [-1];
  const sEdgeBox = [-1]; // box cell before the push that made this state
  const sEdgeDir = [-1]; // direction of that push
  const bestExact = new Map(); // boxes|player -> best g
  const bestRegion = new Map(); // boxes|regionRep -> best g
  const h0 = matchingCost(boxes0);
  if (h0 >= INF || layoutDead(boxes0)) return null;

  // Binary heap on (f, h, seq): lower f first, then lower h, then older first.
  const heap = [];
  let seq = 0;
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  function heapPush(e) {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  }
  function heapPop() {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  }

  bestExact.set(boxes0.join(',') + '|' + player, 0);
  heapPush([h0, h0, seq++, 0]);

  while (heap.length > 0) {
    if (budget.left <= 0) return null;
    const [, , , s] = heapPop();
    const boxes = sBoxes[s];
    const g = sG[s];
    // Canonical duplicate check: same boxes, same player region, no better g.
    const rep = regionRep(sPlayer[s], boxes);
    const rkey = boxes.join(',') + '|' + rep;
    const seenG = bestRegion.get(rkey);
    if (seenG !== undefined && seenG <= g) continue;
    bestRegion.set(rkey, g);
    budget.left--;
    if (boxes.every((b) => goalIdx[b] >= 0)) {
      // Reconstruct the moves: walk to each push stance, then the push.
      let moves = '';
      const edges = [];
      for (let t = s; t > 0; t = sParent[t]) edges.push(t);
      edges.reverse();
      for (const t of edges) {
        const p = sParent[t];
        const [dx, dy] = DIRS[sEdgeDir[t]];
        const stance = sEdgeBox[t] - dy * w - dx;
        const walk = walkPath(w, walkOK, sBoxes[p], sPlayer[p], stance);
        if (walk === null) return null;
        moves += walk + DIRS[sEdgeDir[t]][3];
      }
      return { pushes: g, moves, boxes };
    }
    // Generate pushes: for each box and direction, the player must be able to
    // stand behind the box (in the current region) and the destination must be
    // a free, box-occupiable cell.
    occStamp++;
    for (const b of boxes) occMark[b] = occStamp;
    regionStamp++;
    const regionCells = [sPlayer[s]];
    regionMark[sPlayer[s]] = regionStamp;
    for (let ri = 0; ri < regionCells.length; ri++) {
      const c = regionCells[ri];
      const x = c % w;
      const nbs = [];
      if (x > 0) nbs.push(c - 1);
      if (x < w - 1) nbs.push(c + 1);
      if (c - w >= 0) nbs.push(c - w);
      if (c + w < size) nbs.push(c + w);
      for (const q of nbs) {
        if (walkOK[q] && !wall[q] && occMark[q] !== occStamp && regionMark[q] !== regionStamp) {
          regionMark[q] = regionStamp;
          regionCells.push(q);
        }
      }
    }
    const inRegion = (c) => regionMark[c] === regionStamp;
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi];
      for (let d = 0; d < 4; d++) {
        const [dx, dy] = DIRS[d];
        const stance = b - dy * w - dx;
        const dest = b + dy * w + dx;
        if (dest < 0 || dest >= size || stance < 0 || stance >= size) continue;
        if (dx !== 0 && Math.floor(dest / w) !== Math.floor(b / w)) continue;
        if (dx !== 0 && Math.floor(stance / w) !== Math.floor(b / w)) continue;
        if (!walkOK[stance] || wall[stance] || !inRegion(stance)) continue;
        if (!boxOK[dest] || wall[dest] || occMark[dest] === occStamp) continue;
        const nb = boxes.slice();
        nb[bi] = dest;
        nb.sort((a, b2) => a - b2);
        if (layoutDead(nb)) continue;
        const h = matchingCost(nb);
        if (h >= INF) continue;
        const ng = g + 1;
        const ekey = nb.join(',') + '|' + b;
        const prev = bestExact.get(ekey);
        if (prev !== undefined && prev <= ng) continue;
        bestExact.set(ekey, ng);
        const t = sBoxes.length;
        sBoxes.push(nb);
        sPlayer.push(b);
        sG.push(ng);
        sParent.push(s);
        sEdgeBox.push(b);
        sEdgeDir.push(d);
        heapPush([ng + h, h, seq++, t]);
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Whole-level solver: split into rooms, solve each, join with walks
// ---------------------------------------------------------------------------

function solveLevel(text, budget) {
  const L = parseLevel(text);
  const { w, h, wall, goals, boxes, player } = L;
  const size = w * h;
  if (boxes.length === 0) return '';

  const hallRow = Math.floor(player / w);
  const floor = new Uint8Array(size);
  for (let c = 0; c < size; c++) if (!wall[c]) floor[c] = 1;

  // Connected components of floor off the hallway row: the rooms.
  const comp = new Int32Array(size).fill(-1);
  const rooms = [];
  for (let c = 0; c < size; c++) {
    if (!floor[c] || Math.floor(c / w) === hallRow || comp[c] >= 0) continue;
    const id = rooms.length;
    const cells = [];
    const stack = [c];
    comp[c] = id;
    while (stack.length > 0) {
      const cur = stack.pop();
      cells.push(cur);
      const x = cur % w;
      const nbs = [];
      if (x > 0) nbs.push(cur - 1);
      if (x < w - 1) nbs.push(cur + 1);
      if (cur - w >= 0) nbs.push(cur - w);
      if (cur + w < size) nbs.push(cur + w);
      for (const q of nbs) {
        if (floor[q] && Math.floor(q / w) !== hallRow && comp[q] < 0) {
          comp[q] = id;
          stack.push(q);
        }
      }
    }
    rooms.push({ cells });
  }

  // A box in the hallway row, or a room without a door, breaks the split:
  // fall back to solving the whole level as one puzzle.
  const hallwayBox = boxes.some((b) => Math.floor(b / w) === hallRow);
  for (const room of rooms) {
    room.boxes = boxes.filter((b) => comp[b] === rooms.indexOf(room));
    room.goals = goals.filter((g) => comp[g] === rooms.indexOf(room));
    // Door cells: room cells adjacent to the hallway row; outside cells are
    // the hallway cells beside them.
    room.doors = [];
    for (const c of room.cells) {
      const r = Math.floor(c / w);
      if (r === hallRow - 1) {
        const out = c + w;
        if (floor[out]) room.doors.push([c, out]);
      } else if (r === hallRow + 1) {
        const out = c - w;
        if (floor[out]) room.doors.push([c, out]);
      }
    }
  }
  const splitOk = !hallwayBox && rooms.length > 0 &&
    rooms.every((r) => r.doors.length > 0 && r.boxes.length === r.goals.length &&
      r.boxes.length > 0);
  if (!splitOk) {
    const allFloor = floor;
    const res = solvePushes(
      { w, wall, boxOK: allFloor, walkOK: allFloor, boxes, goals, player },
      budget,
    );
    return res ? res.moves : '';
  }

  // Solve rooms left to right by door column; the order only affects the
  // walk, never the push count.
  rooms.sort((a, b) => a.doors[0][1] - b.doors[0][1]);
  let moves = '';
  let curPlayer = player;
  const curBoxes = boxes.slice();
  for (const room of rooms) {
    const boxOK = new Uint8Array(size);
    for (const c of room.cells) boxOK[c] = 1;
    const walkOK = boxOK.slice();
    for (const [, out] of room.doors) walkOK[out] = 1;
    const doorOut = room.doors[0][1];
    // Walk to the door from wherever the player stands.
    const walk = walkPath(w, floor, curBoxes, curPlayer, doorOut);
    if (walk === null) return '';
    moves += walk;
    curPlayer = doorOut;
    const res = solvePushes(
      { w, wall, boxOK, walkOK, boxes: room.boxes, goals: room.goals, player: doorOut },
      budget,
    );
    if (res === null) {
      // One room resisted: try the whole level as a single puzzle instead.
      const allFloor = floor;
      const whole = solvePushes(
        { w, wall, boxOK: allFloor, walkOK: allFloor, boxes, goals, player },
        budget,
      );
      return whole ? whole.moves : '';
    }
    moves += res.moves;
    // Room boxes are now on their goals; the player ends where the room
    // solution leaves them.
    const finalBoxes = res.boxes;
    for (let i = 0; i < room.boxes.length; i++) {
      const idx = curBoxes.indexOf(room.boxes[i]);
      curBoxes[idx] = finalBoxes[i];
    }
    curBoxes.sort((a, b) => a - b);
    // The room solution's final player cell: replay its last push.
    curPlayer = replayEndPlayer(w, wall, room.boxes, room.goals, doorOut, res.moves, walkOK);
    if (curPlayer < 0) return '';
  }
  return moves;
}

// Replay a room's move string to find where the player ends up. Returns -1
// when the replay is illegal (never expected: the solver builds legal moves).
function replayEndPlayer(w, wall, boxes, goals, player, moves, walkOK) {
  const boxSet = new Set(boxes);
  let p = player;
  for (const ch of moves) {
    const lower = ch.toLowerCase();
    const d = DIRS.find(([,, l]) => l === lower);
    if (!d) return -1;
    const [dx, dy] = d;
    const q = p + dy * w + dx;
    if (q < 0 || q >= wall.length || wall[q] || !walkOK[q]) return -1;
    if (boxSet.has(q)) {
      const beyond = q + dy * w + dx;
      if (wall[beyond] || boxSet.has(beyond)) return -1;
      boxSet.delete(q);
      boxSet.add(beyond);
    }
    p = q;
  }
  return p;
}

// ---------------------------------------------------------------------------
// Entry point: answer every instance in the input directory
// ---------------------------------------------------------------------------

const budget = { left: GLOBAL_BUDGET };
for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    const level = readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8');
    moves = solveLevel(level, budget);
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
