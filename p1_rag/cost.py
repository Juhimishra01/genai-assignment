"""python -m eval.cost  -> markdown table. ALL PRICES ARE ASSUMPTIONS: verify on the vendor pricing pages
and edit the constants below before you submit. Query/LLM cost is excluded (identical for both options)."""
import math
DIM = 384                 # all-MiniLM-L6-v2
BYTES_PER_VEC = DIM * 4 * 1.5 + 600   # float32 * ~1.5 HNSW overhead + ~600B text/metadata
RAM_HEADROOM = 1.5
# Self-hosted: smallest VM whose RAM fits the index (assumed generic cloud VM list prices, $/mo)
VM_TIERS = [(2, 12), (4, 24), (8, 48), (16, 96), (32, 192), (64, 384), (128, 768)]
# Managed (pod-based, always-on): assumed ~$70/pod/mo; assumed capacity ~1M vectors per pod at 768-d (scales ~1/dim)
POD_PRICE, POD_CAP_768 = 70.0, 1_000_000
REPLICAS = 1

def self_hosted(n):
    need = n * BYTES_PER_VEC * RAM_HEADROOM / 1e9
    for gb, price in VM_TIERS:
        if gb >= need: return price, need
    return VM_TIERS[-1][1] * math.ceil(need / VM_TIERS[-1][0]), need

def managed(n):
    cap = POD_CAP_768 * 768 / DIM
    return math.ceil(n / cap) * POD_PRICE * REPLICAS

print("| Vectors | Self-hosted ($/mo) | Managed pods ($/mo) | Savings | Index RAM needed (GB) |\n|---|---|---|---|---|")
for n in (100_000, 1_000_000, 10_000_000):
    s, need = self_hosted(n); m = managed(n)
    print(f"| {n:,} | {s:.0f} | {m:.0f} | {100*(1-s/m):.0f}% | {need:.2f} |")
