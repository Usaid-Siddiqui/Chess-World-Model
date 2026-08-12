"""Dataset over tokenized game shards.

Each item is one whole game, right-padded to a fixed context length. Whole games (rather
than fixed windows) keep training sequences aligned with the probe's board
reconstruction, which needs the move history from the game start.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cwm import moves


class GameDataset(Dataset):
    """Serves padded games as ``(input_ids, length)``.

    ``input_ids`` is a LongTensor of shape ``(ctx,)`` padded with ``PAD_ID``; ``length``
    is the number of real tokens (including ``<bos>``). The AR trainer builds shifted
    (x, y) targets from these; the probe harness uses ``length`` to know how many plies
    are real.
    """

    def __init__(self, data_dir: str | Path, split: str, ctx: int):
        self.data_dir = Path(data_dir)
        self.split = split
        self.ctx = ctx
        with open(self.data_dir / "meta.json") as f:
            self.meta = json.load(f)
        if self.meta["vocab_fingerprint"] != moves.vocab_fingerprint():
            raise ValueError(
                "Vocabulary fingerprint mismatch between shards and cwm.moves — "
                "the data was tokenized with a different vocabulary. Re-run prepare."
            )
        self.tokens = np.fromfile(self.data_dir / f"{split}.bin", dtype=np.uint16)
        self.offsets = np.load(self.data_dir / f"{split}.offsets.npy")
        self.num_games = max(len(self.offsets) - 1, 0)

    def __len__(self) -> int:
        return self.num_games

    def game_tokens(self, idx: int) -> np.ndarray:
        """Raw (unpadded) token ids for game ``idx``, truncated to ctx."""
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        return self.tokens[start:end][: self.ctx].astype(np.int64)

    def __getitem__(self, idx: int) -> dict:
        ids = self.game_tokens(idx)
        length = len(ids)
        buf = np.full(self.ctx, moves.PAD_ID, dtype=np.int64)
        buf[:length] = ids
        return {
            "input_ids": torch.from_numpy(buf),
            "length": torch.tensor(length, dtype=torch.long),
        }
