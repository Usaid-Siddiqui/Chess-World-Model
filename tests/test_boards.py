"""Tests for ground-truth board reconstruction.

The probe's labels must exactly match python-chess, or every downstream accuracy number
is meaningless. We check the starting position, a known opening, and label/position
alignment.
"""

import chess
import numpy as np

from cwm import moves
from cwm.probe import boards


def test_starting_position_absolute():
    board = chess.Board()
    labels = boards.board_to_labels(board, relative=False)
    # 32 pieces, 32 empty squares.
    assert (labels == 0).sum() == 32
    # a1 rook is white rook -> class 1 + rook_index(3) = 4.
    assert labels[chess.A1] == 1 + 3
    # a8 rook is black rook -> class 7 + 3 = 10.
    assert labels[chess.A8] == 7 + 3
    # e1 white king -> 1 + 5 = 6; e8 black king -> 7 + 5 = 12.
    assert labels[chess.E1] == 6
    assert labels[chess.E8] == 12


def test_relative_encoding_flips_with_side_to_move():
    board = chess.Board()  # white to move
    white_view = boards.board_to_labels(board, relative=True)
    # White to move: white pieces are "mine" (1..6). a1 rook -> 4.
    assert white_view[chess.A1] == 4
    board.push_san("e4")  # now black to move
    black_view = boards.board_to_labels(board, relative=True)
    # Black to move: black pieces become "mine" (1..6). a8 rook -> 4.
    assert black_view[chess.A8] == 4
    # And the white rook is now "theirs" (7..12).
    assert black_view[chess.A1] == 10


def test_game_labels_align_with_moves():
    game = ["e2e4", "e7e5", "g1f3", "b8c6"]
    token_ids = np.asarray(moves.encode_game(game, add_bos=True))
    positions, labels = boards.game_labels(token_ids, relative=False, skip_start=True)
    assert positions.tolist() == [1, 2, 3, 4]
    assert labels.shape == (4, 64)
    # After 1.e4 the e2 square is empty and e4 holds a white pawn.
    assert labels[0][chess.E2] == 0
    assert labels[0][chess.E4] == 1  # white pawn -> 1 + pawn_index(0)


def test_game_labels_match_independent_replay():
    game = ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3"]
    token_ids = np.asarray(moves.encode_game(game, add_bos=True))
    positions, labels = boards.game_labels(token_ids, relative=True, skip_start=True)
    board = chess.Board()
    for k, uci in enumerate(game):
        board.push(chess.Move.from_uci(uci))
        expected = boards.board_to_labels(board, relative=True)
        assert np.array_equal(labels[k], expected)
