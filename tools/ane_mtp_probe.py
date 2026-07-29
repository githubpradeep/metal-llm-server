"""Measure ANE vs CPU latency for one Gemma4 MTP draft step (E2B shapes).

The whole draft trunk (pre_proj -> 4 layers incl. attention over the target KV
-> output_norm -> post_proj) is one model, so a draft step would cost exactly
one CoreML prediction. lm_head (262144x256) stays out: it belongs on Metal.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

HID = 256          # hidden_head
NH = 4             # draft query heads
HD_SWA = 256
HD_FULL = 512
FFN = 2048
BACKBONE = 1536
SWA_WINDOW = 512


class Layer(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.q = nn.Linear(HID, NH * head_dim, bias=False)
        self.o = nn.Linear(NH * head_dim, HID, bias=False)
        self.gate = nn.Linear(HID, FFN, bias=False)
        self.up = nn.Linear(HID, FFN, bias=False)
        self.down = nn.Linear(FFN, HID, bias=False)
        self.attn_norm = nn.Parameter(torch.ones(HID))
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.post_attn_norm = nn.Parameter(torch.ones(HID))
        self.ffn_norm = nn.Parameter(torch.ones(HID))
        self.post_ffw_norm = nn.Parameter(torch.ones(HID))

    @staticmethod
    def _rms(x, w, eps=1e-6):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def forward(self, h, k, v):
        # k, v: [kv_len, head_dim] (single base KV head, GQA fan-out to NH)
        a = self._rms(h, self.attn_norm)
        q = self.q(a).view(NH, self.head_dim)
        q = self._rms(q, self.q_norm)
        scores = torch.matmul(q, k.transpose(0, 1))          # [NH, kv]
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).reshape(1, NH * self.head_dim)
        h2 = self._rms(self.o(ctx), self.post_attn_norm) + h
        f = self._rms(h2, self.ffn_norm)
        inter = nn.functional.gelu(self.gate(f), approximate="tanh") * self.up(f)
        return self._rms(self.down(inter), self.post_ffw_norm) + h2


class DraftTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(2 * BACKBONE, HID, bias=False)
        self.layers = nn.ModuleList(
            [Layer(HD_SWA), Layer(HD_SWA), Layer(HD_SWA), Layer(HD_FULL)]
        )
        self.out_norm = nn.Parameter(torch.ones(HID))
        self.post = nn.Linear(HID, BACKBONE, bias=False)

    def forward(self, xh, k_swa, v_swa, k_full, v_full):
        h = self.pre(xh)
        for i, layer in enumerate(self.layers):
            if i < 3:
                h = layer(h, k_swa, v_swa)
            else:
                h = layer(h, k_full, v_full)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * self.out_norm
        return self.post(h)


def bench(kv_len, reps=60):
    torch.manual_seed(0)
    model = DraftTrunk().eval()
    swa = min(kv_len, SWA_WINDOW)
    ex = (
        torch.randn(1, 2 * BACKBONE),
        torch.randn(swa, HD_SWA),
        torch.randn(swa, HD_SWA),
        torch.randn(kv_len, HD_FULL),
        torch.randn(kv_len, HD_FULL),
    )
    with torch.no_grad():
        traced = torch.jit.trace(model, ex)

    inputs = [
        ct.TensorType(name="xh", shape=ex[0].shape, dtype=np.float16),
        ct.TensorType(name="k_swa", shape=ex[1].shape, dtype=np.float16),
        ct.TensorType(name="v_swa", shape=ex[2].shape, dtype=np.float16),
        ct.TensorType(name="k_full", shape=ex[3].shape, dtype=np.float16),
        ct.TensorType(name="v_full", shape=ex[4].shape, dtype=np.float16),
    ]

    feed = {
        "xh": ex[0].numpy().astype(np.float16),
        "k_swa": ex[1].numpy().astype(np.float16),
        "v_swa": ex[2].numpy().astype(np.float16),
        "k_full": ex[3].numpy().astype(np.float16),
        "v_full": ex[4].numpy().astype(np.float16),
    }

    results = {}
    for label, units in [
        ("ANE+CPU", ct.ComputeUnit.CPU_AND_NE),
        ("CPU only", ct.ComputeUnit.CPU_ONLY),
        ("ALL(GPU+ANE)", ct.ComputeUnit.ALL),
    ]:
        m = ct.convert(
            traced,
            inputs=inputs,
            minimum_deployment_target=ct.target.macOS14,
            compute_precision=ct.precision.FLOAT16,
            compute_units=units,
convert_to="mlprogram",
        )
        for _ in range(10):
            m.predict(feed)
        t0 = time.perf_counter()
        for _ in range(reps):
            m.predict(feed)
        results[label] = (time.perf_counter() - t0) / reps * 1e3
    return results


if __name__ == "__main__":
    for kv in (192, 512, 2048):
        r = bench(kv)
        line = "  ".join(f"{k}: {v:.2f} ms" for k, v in r.items())
        print(f"kv_len={kv:5d}  {line}", flush=True)
