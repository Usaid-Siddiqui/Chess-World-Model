"""Tests for the play harness: players must always produce legal moves and finish games."""

import chess
import torch

from cwm.model.ar import ARModel
from cwm.model.gpt import GPTConfig
from cwm import moves
from cwm.play import ARPlayer, RandomPlayer, play_game


def test_random_player_legal():
    board = chess.Board()
    p = RandomPlayer(seed=0)
    for _ in range(10):
        if board.is_game_over():
            break
        mv = p.select_move(board)
        assert mv in board.legal_moves
        board.push(mv)


def test_ar_player_returns_legal_moves():
    model = ARModel(GPTConfig(vocab_size=moves.VOCAB_SIZE, ctx=64, n_layer=2, n_head=2, n_embd=32))
    player = ARPlayer(model, torch.device("cpu"))
    board = chess.Board()
    board.push_san("e4")  # exercise a non-empty history
    mv = player.select_move(board)
    assert mv in board.legal_moves


def test_play_game_finishes_legally():
    result, board = play_game(RandomPlayer(1), RandomPlayer(2), max_plies=60)
    assert result in {"1-0", "0-1", "1/2-1/2"}
    # Every move on the stack was legal by construction; the board is a valid position.
    assert board.is_valid()
