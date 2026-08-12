"""Turn games into tokenized shards.

Games (from any source) are filtered, converted to move-token sequences, and written as
a flat ``uint16`` token stream plus a per-game offsets index. We store whole games (not
fixed windows), because reconstructing the board at ply *t* for probing requires the
move history from the game start — so training sequences and probe targets stay aligned.

Output layout (per split) under ``out_dir``:
  * ``{split}.bin``          — flat uint16 token ids of all games, concatenated
  * ``{split}.offsets.npy``  — int64 array, shape (num_games+1,), cumulative token counts
  * ``meta.json``            — vocab size/fingerprint, filters, counts

Sources are decoupled from writing: ``prepare`` consumes any iterable of
``chess.pgn.Game``. Helpers here provide a PGN-text reader and a random-game generator
(handy for offline plumbing tests); ``download.py`` supplies the Lichess stream.
"""

from __future__ import annotations

import io
import json
import random
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

from cwm import moves


def iter_pgn_games(text_stream: io.TextIOBase) -> Iterator[chess.pgn.Game]:
    """Yield games from a PGN text stream until exhausted."""
    while True:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            return
        yield game


def generate_random_games(
    n: int, seed: int = 0, min_plies: int = 10, max_plies: int = 80
) -> Iterator[chess.pgn.Game]:
    """Generate random *legal* games. Nonsense chess, but valid board dynamics — enough
    to exercise the full pipeline and world-model plumbing offline."""
    rng = random.Random(seed)
    for _ in range(n):
        board = chess.Board()
        target = rng.randint(min_plies, max_plies)
        for _ in range(target):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        game = chess.pgn.Game.from_board(board)
        game.headers["WhiteElo"] = "1500"
        game.headers["BlackElo"] = "1500"
        yield game


def _elo(headers: chess.pgn.Headers, key: str) -> Optional[int]:
    try:
        return int(headers.get(key, ""))
    except ValueError:
        return None


def game_to_token_ids(
    game: chess.pgn.Game,
    min_elo: int = 0,
    min_plies: int = 10,
    max_plies: int = 512,
) -> Optional[list[int]]:
    """Convert one game to token ids, or return None if it fails the filters.

    Filters: standard variant only, both players' Elo >= ``min_elo`` (when Elo headers
    are present), and at least ``min_plies`` moves. Games are truncated to ``max_plies``.
    """
    headers = game.headers
    if headers.get("Variant", "Standard") != "Standard":
        return None
    if min_elo > 0:
        we, be = _elo(headers, "WhiteElo"), _elo(headers, "BlackElo")
        if we is None or be is None or we < min_elo or be < min_elo:
            return None

    ucis: list[str] = []
    board = game.board()
    for move in game.mainline_moves():
        ucis.append(move.uci())
        board.push(move)
        if len(ucis) >= max_plies:
            break
    if len(ucis) < min_plies:
        return None
    return moves.encode_game(ucis, add_bos=True)


def prepare(
    games: Iterable[chess.pgn.Game],
    out_dir: str | Path,
    min_elo: int = 0,
    min_plies: int = 10,
    max_plies: int = 512,
    max_games: Optional[int] = None,
    val_frac: float = 0.02,
    seed: int = 0,
    source: str = "unknown",
) -> dict:
    """Tokenize and write train/val shards. Returns the metadata dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    buffers = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    kept = 0
    pbar = tqdm(games, desc="tokenizing", unit="game")
    for game in pbar:
        ids = game_to_token_ids(game, min_elo=min_elo, min_plies=min_plies, max_plies=max_plies)
        if ids is None:
            continue
        split = "val" if rng.random() < val_frac else "train"
        buffers[split].append(np.asarray(ids, dtype=np.uint16))
        counts[split] += 1
        kept += 1
        if max_games is not None and kept >= max_games:
            break
        if kept % 1000 == 0:
            pbar.set_postfix(kept=kept, train=counts["train"], val=counts["val"])

    for split, games_ids in buffers.items():
        if not games_ids:
            # Always leave a (possibly empty) file so downstream loaders don't crash.
            np.asarray([], dtype=np.uint16).tofile(out / f"{split}.bin")
            np.save(out / f"{split}.offsets.npy", np.zeros(1, dtype=np.int64))
            continue
        lengths = np.asarray([len(g) for g in games_ids], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        np.concatenate(games_ids).tofile(out / f"{split}.bin")
        np.save(out / f"{split}.offsets.npy", offsets)

    meta = {
        "vocab_size": moves.VOCAB_SIZE,
        "vocab_fingerprint": moves.vocab_fingerprint(),
        "pad_id": moves.PAD_ID,
        "bos_id": moves.BOS_ID,
        "min_elo": min_elo,
        "min_plies": min_plies,
        "max_plies": max_plies,
        "num_games": counts,
        "source": source,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Tokenize chess games into shards.")
    ap.add_argument("--source", choices=["random", "pgn", "lichess"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pgn-path", help="path to a .pgn file (source=pgn)")
    ap.add_argument("--month", help="Lichess month YYYY-MM (source=lichess)")
    ap.add_argument("--num-random", type=int, default=2000)
    ap.add_argument("--min-elo", type=int, default=0)
    ap.add_argument("--min-plies", type=int, default=10)
    ap.add_argument("--max-plies", type=int, default=512)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.source == "random":
        games: Iterable[chess.pgn.Game] = generate_random_games(
            args.num_random, seed=args.seed, max_plies=args.max_plies
        )
    elif args.source == "pgn":
        assert args.pgn_path, "--pgn-path required for source=pgn"
        stream = open(args.pgn_path, "r", encoding="utf-8", errors="ignore")
        games = iter_pgn_games(stream)
    else:  # lichess
        from cwm.data.download import lichess_pgn_stream

        assert args.month, "--month required for source=lichess"
        games = iter_pgn_games(lichess_pgn_stream(args.month))

    meta = prepare(
        games,
        out_dir=args.out_dir,
        min_elo=args.min_elo,
        min_plies=args.min_plies,
        max_plies=args.max_plies,
        max_games=args.max_games,
        val_frac=args.val_frac,
        seed=args.seed,
        source=args.source,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    _main()
