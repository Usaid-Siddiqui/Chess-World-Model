"""Evaluate a player by match play — Phase 4.

Plays a color-balanced match between a model checkpoint (AR) and an opponent (random, or
another checkpoint), and reports win/draw/loss + score from the model's perspective. A
score well above 0.5 vs random is the first sign the policy is real; head-to-head vs another
checkpoint compares two models directly.

Usage:
    python -m cwm.eval --checkpoint checkpoints/ar/model_best.pt --opponent random --n-games 100
    python -m cwm.eval --checkpoint checkpoints/ar/model_best.pt \\
        --opponent checkpoints/jepa_con/model_best.pt --n-games 100
"""

from __future__ import annotations

import argparse

from cwm.play import ARPlayer, RandomPlayer, play_game
from cwm.probe.probe import load_model
from cwm.utils.config import pick_device


def _make_player(spec: str, device, temperature: float):
    if spec == "random":
        return RandomPlayer()
    model, _ = load_model(spec, device)  # a checkpoint path
    if not hasattr(model, "cfg"):
        raise SystemExit(f"{spec}: not an AR checkpoint (no move policy to play with)")
    return ARPlayer(model, device, temperature=temperature)


def match(player, opponent, n_games: int, max_plies: int) -> dict:
    """Play n_games, alternating which side `player` takes. Score from player's view."""
    w = d = l = 0
    for g in range(n_games):
        player_white = g % 2 == 0
        white, black = (player, opponent) if player_white else (opponent, player)
        result, _ = play_game(white, black, max_plies=max_plies)
        if result == "1/2-1/2":
            d += 1
        elif (result == "1-0") == player_white:
            w += 1
        else:
            l += 1
    score = (w + 0.5 * d) / max(n_games, 1)
    return {"games": n_games, "win": w, "draw": d, "loss": l, "score": score}


def run(args):
    device = pick_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    if not hasattr(model, "cfg"):
        raise SystemExit("--checkpoint must be an AR model (JEPA has no native move policy)")
    player = ARPlayer(model, device, temperature=args.temperature)
    opponent = _make_player(args.opponent, device, args.temperature)
    print(f"{args.checkpoint} (arm={ckpt['arm']})  vs  {args.opponent}  "
          f"[{args.n_games} games, temp={args.temperature}]")
    r = match(player, opponent, args.n_games, args.max_plies)
    print(f"W {r['win']}  D {r['draw']}  L {r['loss']}  ->  score {r['score']*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="AR checkpoint to evaluate")
    ap.add_argument("--opponent", default="random", help="'random' or a checkpoint path")
    ap.add_argument("--n-games", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
