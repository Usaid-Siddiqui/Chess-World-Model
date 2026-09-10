"""Tests for the value head and the JEPA latent planner."""

import chess
import torch

from cwm import moves
from cwm.model.gpt import GPTConfig
from cwm.model.jepa import JEPAModel
from cwm.play import JEPAPlanner
from cwm.value import ValueHead, board_value


def test_board_value_start_is_balanced():
    assert board_value(chess.Board()) == 0.0


def test_board_value_material_advantage():
    board = chess.Board()
    board.remove_piece_at(chess.D8)  # remove black's queen; white to move is up 9
    assert board_value(board) == 9.0
    board.push_san("e4")  # now black to move, down 9 from their perspective
    assert board_value(board) == -9.0


def test_jepa_planner_returns_legal_move():
    cfg = GPTConfig(vocab_size=moves.VOCAB_SIZE, ctx=64, n_layer=2, n_head=2, n_embd=32)
    model = JEPAModel(cfg, {"rollout_steps": 2, "predictor_layers": 1, "predictor_mlp_ratio": 2})
    planner = JEPAPlanner(model, ValueHead(cfg.n_embd, hidden=16), torch.device("cpu"))
    board = chess.Board()
    board.push_san("d4")
    mv = planner.select_move(board)
    assert mv in board.legal_moves
