#!/usr/bin/env python3
"""Minimal standalone GGUF inspector (no external deps).

Dumps header, metadata KVs, and the full tensor table (name/type/shape/offset).
Large arrays (tokenizer vocab/merges) are summarized, not dumped.
"""
import struct
import sys
from collections import Counter

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16",
}

# GGUF metadata value types
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

_FMT = {
    U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        return self.f.read(n)

    def scalar(self, t):
        fmt, n = _FMT[t]
        return struct.unpack(fmt, self.f.read(n))[0]

    def u32(self):
        return self.scalar(U32)

    def u64(self):
        return self.scalar(U64)

    def gstr(self):
        n = self.u64()
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, t):
        if t == STRING:
            return self.gstr()
        if t == ARRAY:
            at = self.u32()
            n = self.u64()
            # Summarize big arrays instead of materializing them.
            if at == STRING:
                # Must consume the bytes to stay aligned.
                sample = []
                for i in range(n):
                    s = self.gstr()
                    if i < 5:
                        sample.append(s)
                return {"array_type": "STRING", "len": n, "sample": sample}
            else:
                fmt, sz = _FMT[at]
                sample = []
                for i in range(n):
                    v = self.scalar(at)
                    if i < 5:
                        sample.append(v)
                return {"array_type": _typename(at), "len": n, "sample": sample}
        return self.scalar(t)


def _typename(t):
    return {U8: "U8", I8: "I8", U16: "U16", I16: "I16", U32: "U32", I32: "I32",
            F32: "F32", BOOL: "BOOL", STRING: "STRING", ARRAY: "ARRAY",
            U64: "U64", I64: "I64", F64: "F64"}.get(t, f"?{t}")


def main(path):
    with open(path, "rb") as f:
        r = Reader(f)
        magic = f.read(4)
        if magic != b"GGUF":
            print("NOT a GGUF file, magic =", magic)
            return
        version = r.u32()
        tensor_count = r.u64()
        kv_count = r.u64()
        print(f"=== GGUF header ===")
        print(f"version={version} tensor_count={tensor_count} metadata_kv_count={kv_count}")

        print(f"\n=== metadata ({kv_count} keys) ===")
        meta = {}
        for _ in range(kv_count):
            key = r.gstr()
            vt = r.u32()
            val = r.value(vt)
            meta[key] = val
            if isinstance(val, dict):  # array summary
                print(f"  {key}: array<{val['array_type']}> len={val['len']} sample={val['sample']}")
            else:
                sval = repr(val)
                if len(sval) > 200:
                    sval = sval[:200] + "..."
                print(f"  {key}: {sval}")

        print(f"\n=== tensors ({tensor_count}) ===")
        type_counter = Counter()
        tensors = []
        for _ in range(tensor_count):
            name = r.gstr()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()
            tname = GGML_TYPES.get(ttype, f"?{ttype}")
            type_counter[tname] += 1
            tensors.append((name, tname, dims, offset))

        # Print all tensors sorted by name
        for name, tname, dims, offset in sorted(tensors):
            print(f"  {name:55s} {tname:6s} dims={dims} off={offset}")

        print(f"\n=== tensor type histogram ===")
        for t, c in type_counter.most_common():
            print(f"  {t:8s} {c}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
