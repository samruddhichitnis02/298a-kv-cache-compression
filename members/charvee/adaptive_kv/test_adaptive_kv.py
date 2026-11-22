"""Self-tests for adaptive_kv.py on a tiny random Llama (CPU, no downloads, ~10 s).

    python members/charvee/adaptive_kv/test_adaptive_kv.py
"""

import math
import sys
import tempfile
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_kv import (AdaptiveKVEngine, analytic_ratio, frequency_scores,  # noqa: E402
                         golden_section_search, select_indices)

torch.manual_seed(0)
CFG = LlamaConfig(vocab_size=257, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                  bos_token_id=0, eos_token_id=1)
MODEL = LlamaForCausalLM(CFG).float().eval()
BOS = [0]


def toks(n, lo=2):
    return torch.randint(lo, CFG.vocab_size, (n,)).tolist()


def close(a, b, tol=1e-4):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def test_frequency_scores_shapes_and_low_pass():
    keys = [torch.randn(37, 2, 8) for _ in range(3)]
    vals = [torch.randn(37, 2, 8) for _ in range(3)]
    s = frequency_scores(keys, vals, alpha=0.5)
    assert s.shape == (37,) and math.isclose(float(s.sum()), 1.0, rel_tol=1e-4)
    # alpha = 1 keeps every frequency, so the score equals the plain L2 ranking
    assert close(frequency_scores(keys, vals, alpha=1.0), frequency_scores(keys, vals, mode="l2"))
    # a constant (DC-only) signal survives any low-pass filter unchanged
    const = [torch.ones(16, 1, 4)]
    assert close(frequency_scores(const, const, alpha=0.01), frequency_scores(const, const, mode="l2"))


def test_select_indices():
    ranking = torch.tensor([5, 2, 9, 0, 1, 3, 4, 6, 7, 8])
    assert select_indices(ranking, 0.3).tolist() == [2, 5, 9]
    assert len(select_indices(ranking, 0.0)) == 0 and len(select_indices(ranking, 1.0)) == 10


def run_case(tier, tmp):
    eng = AdaptiveKVEngine(MODEL, tier=tier, pool_dir=tmp)
    prefix, c1, c2, query = BOS + toks(3), toks(40), toks(25), toks(6)
    k1 = eng.encode_chunk(c1, prefix_ids=BOS)
    k2 = eng.encode_chunk(c2, prefix_ids=BOS)
    assert eng.encode_chunk(c1, prefix_ids=BOS) is k1           # memoised
    full_ids = prefix + c1 + c2 + query
    full = eng.full_prefill(full_ids)

    # r = 1: every chunk token is recomputed -> identical to full recompute
    r1 = eng.prefill([prefix, k1, k2, query], ratio=1.0)
    assert close(r1.logits, full.logits), "r=1 must equal full recompute"
    assert r1.loaded_bytes == 0 and r1.n_recomputed == 65

    # r = 0, chunk in the same context it was encoded in ([BOS] + chunk) -> exact reuse;
    # checks the pre-RoPE storage, deferred RoPE and scatter-fusion
    r0 = eng.prefill([BOS, k1, query], ratio=0.0)
    ref = eng.full_prefill(BOS + c1 + query)
    assert close(r0.logits, ref.logits), "exact-context reuse must equal full recompute"

    # non-prefix reuse: quality should improve monotonically-ish with r
    errs = []
    for r in (0.0, 0.15, 0.5, 1.0):
        res = eng.prefill([prefix, k1, k2, query], ratio=r)
        errs.append(float((res.logits - full.logits).abs().max()))
        assert res.loaded_bytes == (65 - res.n_recomputed) * k1.store.nbytes_per_row * CFG.num_hidden_layers
    assert errs[0] > errs[-1] and errs[-1] < 1e-4, errs

    # decoding from the reused cache matches decoding from the full cache when r = 1
    assert eng.generate(r1, 8) == eng.generate(eng.full_prefill(full_ids), 8)

    # a chunk at the very end (no uncached suffix) still yields logits
    tail = eng.prefill([prefix, k1], ratio=0.0)
    assert tail.logits.shape == (CFG.vocab_size,)
    eng.clear_pool()
    return errs


def test_tiers():
    with tempfile.TemporaryDirectory() as tmp:
        for tier in ("gpu", "cpu", "disk"):   # "gpu" just means "on the model's device"
            errs = run_case(tier, Path(tmp))
            print(f"  tier={tier:4s} max|logit err| at r=0/.15/.5/1: " + " ".join(f"{e:.4f}" for e in errs))
        assert not list(Path(tmp).glob("*.bin")), "disk pool not cleaned up"


def test_selection_modes_run():
    for mode in ("high", "l2", "random"):
        eng = AdaptiveKVEngine(MODEL, tier="cpu", selection=mode)
        k = eng.encode_chunk(toks(30), prefix_ids=BOS)
        assert sorted(k.ranking.tolist()) == list(range(30))
        eng.prefill([BOS, k, toks(4)], ratio=0.2)


def test_golden_section():
    target = 0.37
    r_star, evals = golden_section_search(lambda r: (r - target) ** 2 + 1, 0.15, 1.0, r0=0.3, eps=1e-3)
    assert abs(r_star - target) < 2e-3, r_star
    assert len(evals) < 25
    # minimum below r_min is clipped to the quality lower bound
    r_star, _ = golden_section_search(lambda r: r, 0.15, 1.0, r0=0.15, eps=1e-3)
    assert abs(r_star - 0.15) < 2e-3
    assert analytic_ratio(t_c=1.0, t_i=1.0) == 0.5
    assert analytic_ratio(t_c=1.0, t_i=0.01) == 0.15            # cheap loads -> lower bound


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("all tests passed")
