"""Play chess with a trained model — Phase 4, testing whether the world model is *useful*.

The AR model has a native move policy, so it plays by masking its next-move logits to the
legal moves and picking (greedy or temperature-sampled). Players are **stateless**: each
move is chosen by rebuilding the token history from ``board.move_stack``, so the same player
can drive either color and recover from any position.

JEPA has no native policy (it is a representation + dynamics), so a JEPA player needs an
added head — see ``JEPA player`` note at the bottom.
"""

from __future__ import annotations

import chess
import numpy as np
import torch

from cwm import moves


class RandomPlayer:
    """Uniform-random legal moves — the floor baseline."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        return legal[int(self.rng.integers(len(legal)))]


class ARPlayer:
    """Plays by the AR model's next-move policy, masked to legal moves."""

    def __init__(self, model, device, temperature: float = 0.0):
        self.model = model.eval()
        self.device = device
        self.temperature = temperature
        self.ctx = model.cfg.ctx

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        ids = [moves.BOS_ID] + [moves.encode_move(m.uci()) for m in board.move_stack]
        ids = ids[-self.ctx:]  # keep within context window
        x = torch.tensor(ids, device=self.device, dtype=torch.long)[None, :]
        logits = self.model(x)[0, -1].float()  # next-move logits (V,)
        legal = torch.from_numpy(moves.legal_move_mask(board)).to(self.device)
        logits = logits.masked_fill(~legal, float("-inf"))
        if self.temperature <= 0:
            idx = int(logits.argmax())
        else:
            probs = torch.softmax(logits / self.temperature, dim=-1)
            idx = int(torch.multinomial(probs, 1))
        return chess.Move.from_uci(moves.decode_id(idx))


def play_game(white, black, max_plies: int = 300) -> tuple[str, chess.Board]:
    """Play one game; return (result, final board). A ply cap counts as a draw."""
    board = chess.Board()
    players = {chess.WHITE: white, chess.BLACK: black}
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        board.push(players[board.turn].select_move(board))
    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    return result, board
