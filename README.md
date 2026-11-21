# sokoban-solver

A push-optimal Sokoban solver, written from scratch for make.fun's
**Build the best Sokoban solver, 21 November 2026** bounty
(`sokoban-solver-01-run-20261121`).

## How it works

The levels this competition draws are a hallway with one to four small rooms
off it, each room holding 3 or 4 boxes and as many goals, and a box pushed out
of a room's door into the hallway can never be pushed back. So every box stays
in its room, each room is an independent puzzle, and a level's push-optimal
count is the sum of its rooms' optima. This program exploits that:

1. **Split** a level into its rooms (connected components of floor off the
   hallway row) and find each room's door.
2. **Solve each room push-optimally** with A\* over pushes: a state is the
   sorted box squares plus the player's square; the player is canonicalised to
   the reachable region they stand in, so equivalent player squares collapse
   into one state.
3. **Heuristic:** the exact minimum-cost matching of boxes to goals over
   shortest-path distances (admissible: every push moves one box one square),
   computed by subset DP and cached per box layout.
4. **Deadlock pruning:** static dead squares, 2x2 block deadlocks, and a
   frozen-box fixpoint (a box that can never move again, off its goal).
5. **Join** the room solutions with plain walks through the hallway (boxes are
   obstacles while walking, so no box is ever pushed between rooms).

The search is bounded by work, never by the clock: expansion budgets are fixed
counts, so two runs write the same bytes. No randomness, no clock reads.

## Usage

Node 22 runs `node main.ts`. It reads `$MAKEFUN_INPUT_DIR/<id>/level.txt`
(default `/input`) and writes `$MAKEFUN_OUTPUT_DIR/<id>/solution.txt`
(default `/output`), the player's moves in LURD (lowercase a step, uppercase a
push), one file per instance.

## Results

Practice set (kit's `practice/`, optima 13 / 16 / 29 / 47 / 60 pushes):

| instance | pushes | points |
|----------|--------|--------|
| i001 | 13 | 1000 |
| i002 | 16 | 1000 |
| i003 | 29 | 1000 |
| i004 | 47 | 1000 |
| i005 | 60 | 1000 |

Score **1,000,000** (verified with the kit's local grader:
`node cli/makefun.mjs grade-local --kit . --bot <repo> --driver process`),
in ~0.5 s for the whole set — far inside the 60 s budget. The house baseline
(plain-astar 1.0.0) scores 200,000 on the same set: it only solves i001.

The solver was also run on three freshly drawn sets from the kit's own
generator (different seeds): 1,000,000 on all three, each in under 0.5 s.

## License

MIT. See [LICENSE](LICENSE).
