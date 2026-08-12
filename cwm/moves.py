"""Move-level UCI tokenizer.

One token per ply. The vocabulary is a fixed, position-independent superset of every
UCI move that is legal in *some* chess position:

  * all queen-geometry moves (same rank / file / diagonal) from every square — this
    covers king, queen, rook, bishop, and non-promoting pawn moves,
  * all knight-geometry moves from every square,
  * all pawn promotions (straight + both captures) into {q, r, b, n}.

This yields ~1968 move tokens. Two special tokens are reserved at the front:
``<pad>`` (id 0) and ``<bos>`` (id 1). The vocabulary is deterministic (sorted UCI
strings) so token ids are stable across runs and machines.

The same ``legal_move_mask`` helper used for probing is what a future play module would
use to constrain sampling to legal moves — no retraining or retokenizing needed.
"""

from __future__ import annotations

import functools
from typing import Iterable

import chess
import numpy as np

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
PAD_ID = 0
BOS_ID = 1
_NUM_SPECIAL = 2

_KNIGHT_DELTAS = {(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)}
_PROMO_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


def _generate_move_ucis() -> list[str]:
    """Enumerate every UCI move legal in some position, as a sorted list."""
    ucis: set[str] = set()
    for frm in chess.SQUARES:
        ff, fr = chess.square_file(frm), chess.square_rank(frm)
        for to in chess.SQUARES:
            if frm == to:
                continue
            tf, tr = chess.square_file(to), chess.square_rank(to)
            df, dr = tf - ff, tr - fr
            is_queen = df == 0 or dr == 0 or abs(df) == abs(dr)
            is_knight = (df, dr) in _KNIGHT_DELTAS
            if is_queen or is_knight:
                ucis.add(chess.Move(frm, to).uci())

    # Promotions: white pawn rank 6->7 (0-indexed), black pawn rank 1->0.
    for file in range(8):
        for (from_rank, to_rank) in ((6, 7), (1, 0)):
            for to_file in (file - 1, file, file + 1):
                if not 0 <= to_file < 8:
                    continue
                frm = chess.square(file, from_rank)
                to = chess.square(to_file, to_rank)
                for promo in _PROMO_PIECES:
                    ucis.add(chess.Move(frm, to, promotion=promo).uci())

    return sorted(ucis)


_MOVE_UCIS = _generate_move_ucis()
_ID_TO_MOVE = [PAD_TOKEN, BOS_TOKEN] + _MOVE_UCIS
_MOVE_TO_ID = {uci: i + _NUM_SPECIAL for i, uci in enumerate(_MOVE_UCIS)}

VOCAB_SIZE = len(_ID_TO_MOVE)


def encode_move(uci: str) -> int:
    """UCI string -> token id. Raises KeyError if the move is not in the vocabulary."""
    return _MOVE_TO_ID[uci]


def decode_id(token_id: int) -> str:
    """Token id -> UCI string (or a special token)."""
    return _ID_TO_MOVE[token_id]


def is_move_id(token_id: int) -> bool:
    """True if the id is a real move (not a special token)."""
    return token_id >= _NUM_SPECIAL


def encode_game(ucis: Iterable[str], add_bos: bool = True) -> list[int]:
    """Sequence of UCI moves -> token ids, optionally prefixed with <bos>."""
    ids = [BOS_ID] if add_bos else []
    ids.extend(_MOVE_TO_ID[u] for u in ucis)
    return ids


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """Boolean vector over the vocabulary: True for moves legal in `board`.

    Special tokens are always False. Reused by probing and by a future play module to
    restrict sampling to legal moves.
    """
    mask = np.zeros(VOCAB_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[_MOVE_TO_ID[move.uci()]] = True
    return mask


@functools.lru_cache(maxsize=1)
def vocab_fingerprint() -> str:
    """Stable hash of the vocabulary, stored in dataset metadata to guard against drift."""
    import hashlib

    return hashlib.sha256("\n".join(_ID_TO_MOVE).encode()).hexdigest()[:16]
