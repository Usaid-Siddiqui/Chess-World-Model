"""Tests for the JEPA arm.

Checks the mechanics that are easy to get wrong: the loss is a finite differentiable
scalar, gradients reach the student/predictor but not the frozen teacher, the EMA update
moves the teacher toward the student, and the probe interface matches the AR arm.
"""

import torch

from cwm import moves
from cwm.model.gpt import GPTConfig
from cwm.model.jepa import JEPAModel

CFG = GPTConfig(vocab_size=moves.VOCAB_SIZE, ctx=32, n_layer=2, n_head=2, n_embd=32)
JCFG = {"ema_decay": 0.99, "rollout_steps": 3, "predictor_layers": 2, "predictor_mlp_ratio": 2}


def _batch(B=4, T=20):
    ids = torch.randint(moves.BOS_ID, moves.VOCAB_SIZE, (B, T))
    ids[:, 0] = moves.BOS_ID
    ids[:, -3:] = moves.PAD_ID  # some padding at the end
    return ids


def test_loss_is_finite_scalar_with_grad():
    model = JEPAModel(CFG, JCFG)
    loss = model.compute_loss(_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    # Student and predictor get gradients.
    assert model.backbone.tok_emb.weight.grad is not None
    assert any(p.grad is not None for p in model.predictor.parameters())


def test_teacher_frozen_no_grad():
    model = JEPAModel(CFG, JCFG)
    model.compute_loss(_batch()).backward()
    for p in model.teacher.parameters():
        assert not p.requires_grad
        assert p.grad is None


def test_ema_moves_teacher_toward_student():
    model = JEPAModel(CFG, JCFG)
    # Perturb the student so it differs from the (initially identical) teacher.
    with torch.no_grad():
        for p in model.backbone.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    before = model.teacher.tok_emb.weight.clone()
    student = model.backbone.tok_emb.weight.clone()
    model.update_teacher()
    after = model.teacher.tok_emb.weight
    # Teacher moved a (1-decay) fraction toward the student.
    assert torch.norm(after - student) < torch.norm(before - student)


def test_teacher_stays_in_eval():
    model = JEPAModel(CFG, JCFG)
    model.train()
    assert not model.teacher.training  # dropout off in the teacher even in train mode


def test_hidden_states_shape_matches_probe_interface():
    model = JEPAModel(CFG, JCFG).eval()
    ids = _batch(B=2, T=16)
    h = model.hidden_states(ids, layer=-1)
    assert h.shape == (2, 16, CFG.n_embd)


def test_inverse_dynamics_adds_loss_and_head():
    jcfg = {**JCFG, "inverse_weight": 1.0}
    model = JEPAModel(CFG, jcfg)
    assert hasattr(model, "inverse_head")  # built only when enabled
    loss = model.compute_loss(_batch())
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in model.inverse_head.parameters())
    assert 0.0 <= model.last_inverse_acc <= 1.0  # accuracy logged for leakage watch


def test_inverse_off_builds_no_head():
    model = JEPAModel(CFG, {**JCFG, "inverse_weight": 0.0})
    assert not hasattr(model, "inverse_head")  # older checkpoints load unchanged


def test_contrastive_loss_is_finite_and_leak_free_signal():
    jcfg = {**JCFG, "loss": "contrastive", "contrastive_negs": 8, "contrastive_max": 64}
    model = JEPAModel(CFG, jcfg)
    loss = model.compute_loss(_batch(B=6, T=24))
    assert torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    # Gradients reach the encoder and predictor (the board pressure path).
    assert model.backbone.tok_emb.weight.grad is not None
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert 0.0 <= model.last_contrastive_acc <= 1.0
