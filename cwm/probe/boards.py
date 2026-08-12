"""Ground-truth board labels from a token sequence.

For a game ``[<bos>, m1, m2, ...]``, hidden state at sequence position ``i`` encodes the
first ``i`` tokens, i.e. the board *after* ``i`` moves. We replay the moves with
python-chess and emit one 64-square label per position, aligned to those hidden states.

Each square gets a class in 0..12:
    0 = empty, 1..6 = pawn,knight,bishop,rook,queen,king of one color, 7..12 = the other.

Two encodings:
  * **absolute** — 1..6 white, 7..12 black.
  * **relative** ("mine/theirs", the Karvonen trick) — 1..6 are the side-to-move's pieces,
    7..12 the opponent's. This probes far better because the model tracks the position from
    the mover's perspective.

Boards are never fed to the model — these labels exist only to grade its hidden states.
"""

from __future__ import annotations

import chess
import numpy as np

from cwm import moves

NUM_SQUARES = 64
NUM_CLASSES = 13  # empty + 6 + 6

_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_PIECE_INDEX = {p: i for i, p in enumerate(_PIECE_ORDER)}  # 0..5


def board_to_labels(board: chess.Board, relative: bool = True) -> np.ndarray:
    """64-vector of square classes (0..12) for a board."""
    labels = np.zeros(NUM_SQUARES, dtype=np.int64)
    mover = board.turn  # True=white to move
    for square, piece in board.piece_map().items():
        base = _PIECE_INDEX[piece.piece_type]  # 0..5
        if relative:
            mine = piece.color == mover
        else:
            mine = piece.color == chess.WHITE
        labels[square] = 1 + base + (0 if mine else 6)
    return labels


def game_labels(token_ids: np.ndarray, relative: bool = True, skip_start: bool = True):
    """Replay a tokenized game and return (positions, labels).

    ``positions`` are the sequence indices (into the model's hidden states) and ``labels``
    is an array of shape ``(len(positions), 64)``. Position ``i`` = board after ``i`` moves.
    With ``skip_start`` the trivial initial position (index 0, the same for every game) is
    omitted.
    """
    assert token_ids[0] == moves.BOS_ID
    board = chess.Board()
    positions = []
    labels = []

    def record(i):
        positions.append(i)
        labels.append(board_to_labels(board, relative=relative))

    if not skip_start:
        record(0)
    for i, token in enumerate(token_ids[1:], start=1):
        move = chess.Move.from_uci(moves.decode_id(int(token)))
        board.push(move)
        record(i)
    return np.asarray(positions, dtype=np.int64), np.asarray(labels, dtype=np.int64)
