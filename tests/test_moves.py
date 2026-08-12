"""Tests for the UCI move tokenizer.

The critical property: the fixed vocabulary must cover *every* legal move that can arise
in real play. We verify this by playing many random games and asserting every legal move
at every step is encodable.
"""

import random

import chess

from cwm import moves


def test_specials_and_vocab_size():
    assert moves.PAD_ID == 0
    assert moves.BOS_ID == 1
    assert moves.decode_id(moves.PAD_ID) == moves.PAD_TOKEN
    assert moves.decode_id(moves.BOS_ID) == moves.BOS_TOKEN
    # ~1968 real moves + 2 specials. Guard against accidental drift.
    assert 1900 <= moves.VOCAB_SIZE <= 2100
    assert not moves.is_move_id(moves.PAD_ID)
    assert not moves.is_move_id(moves.BOS_ID)


def test_roundtrip_all_moves():
    for token_id in range(moves.VOCAB_SIZE):
        uci = moves.decode_id(token_id)
        if moves.is_move_id(token_id):
            assert moves.encode_move(uci) == token_id


def test_encode_game_prefixes_bos():
    game = ["e2e4", "e7e5", "g1f3"]
    ids = moves.encode_game(game, add_bos=True)
    assert ids[0] == moves.BOS_ID
    assert [moves.decode_id(i) for i in ids[1:]] == game
    assert moves.encode_game(game, add_bos=False)[0] != moves.BOS_ID


def test_vocab_covers_all_legal_moves_in_random_games():
    """No legal move in any reachable position should be missing from the vocab."""
    rng = random.Random(0)
    for _ in range(300):
        board = chess.Board()
        for _ in range(200):
            legal = list(board.legal_moves)
            if not legal:
                break
            for move in legal:
                # Must not raise KeyError — every legal move is in the vocabulary.
                assert moves.encode_move(move.uci()) >= moves._NUM_SPECIAL
            board.push(rng.choice(legal))


def test_legal_move_mask_matches_board():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("c5")
    mask = moves.legal_move_mask(board)
    legal_ids = {moves.encode_move(m.uci()) for m in board.legal_moves}
    assert set(mask.nonzero()[0].tolist()) == legal_ids
    assert not mask[moves.PAD_ID] and not mask[moves.BOS_ID]


def test_fingerprint_stable():
    assert moves.vocab_fingerprint() == moves.vocab_fingerprint()
    assert len(moves.vocab_fingerprint()) == 16
