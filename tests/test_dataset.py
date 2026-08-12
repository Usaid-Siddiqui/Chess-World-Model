"""Tests for prepare -> shards -> dataset.

Key integrity property: every token sequence written to a shard must decode back into a
fully legal game. If tokenization or storage corrupted anything, replaying the moves
through python-chess would raise or reject an illegal move.
"""

import chess

from cwm import moves
from cwm.data.dataset import GameDataset
from cwm.data.prepare import generate_random_games, prepare


def _build(tmp_path, n=200):
    games = generate_random_games(n, seed=1, min_plies=10, max_plies=60)
    meta = prepare(games, out_dir=tmp_path, min_plies=10, max_plies=60, val_frac=0.1, source="random")
    return meta


def test_prepare_writes_splits_and_meta(tmp_path):
    meta = _build(tmp_path)
    assert meta["vocab_size"] == moves.VOCAB_SIZE
    assert meta["num_games"]["train"] > 0
    assert meta["num_games"]["val"] > 0
    for split in ("train", "val"):
        assert (tmp_path / f"{split}.bin").exists()
        assert (tmp_path / f"{split}.offsets.npy").exists()


def test_dataset_shapes_and_padding(tmp_path):
    _build(tmp_path)
    ds = GameDataset(tmp_path, "train", ctx=128)
    item = ds[0]
    assert item["input_ids"].shape == (128,)
    length = int(item["length"])
    assert item["input_ids"][0].item() == moves.BOS_ID
    # Everything past the real length is padding.
    assert (item["input_ids"][length:] == moves.PAD_ID).all()


def test_shards_decode_to_legal_games(tmp_path):
    _build(tmp_path)
    ds = GameDataset(tmp_path, "train", ctx=512)
    for idx in range(len(ds)):
        ids = ds.game_tokens(idx)
        assert ids[0] == moves.BOS_ID
        board = chess.Board()
        for token in ids[1:]:
            uci = moves.decode_id(int(token))
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves, f"illegal move {uci} in game {idx}"
            board.push(move)
