"""Summarize / plot a decode sweep CSV (vanilla.csv or sweep.csv) per experiments/demand.md.

    python experiments/plot_results.py [results/sweep.csv]

Prints markdown tables AND an explicit list of OOM/skipped/errored configs (demand.md:
never silently drop one). If matplotlib is installed, writes the 4 demand.md figures next
to the CSV:
  fig1_throughput_vs_batch.png  (line per K, panel per prefix)
  fig2_latency_vs_batch.png     (line per K, panel per prefix)
  fig3_throughput_vs_K.png      (line per prefix, fixed batch — where K>1 overtakes K=1)
  fig4_equal_eff_K_vs_S.png     (line per eff=K/S — wider-K vs more-S, fixed batch+prefix)

With only K=1 data present (vanilla baseline), figs 3/4 are near-trivial; they fill in
once the K>1 engine path produces data. (install plotting: `uv pip install matplotlib`.)
"""
import csv
import os
import sys
from collections import defaultdict


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ok_points(rows):
    pts = []
    for r in rows:
        if r.get("status", "ok") != "ok":
            continue
        thr = fnum(r.get("throughput_tok_s"))
        if thr is None:
            continue
        pts.append({
            "K": int(r["K"]), "S": int(r["S"]), "batch": int(r["batch"]),
            "prefix": int(r["prefix_len"]), "thr": thr,
            # per-request latency = seconds to generate all output_len tokens for a request
            # (static lockstep batch -> equals decode_time_s).
            "lat": fnum(r.get("decode_time_s")), "eff": fnum(r.get("eff_tok_per_step")),
            "peak": r.get("peak_mem_gb", ""),
        })
    return pts


def tables(rows):
    pts = ok_points(rows)
    by_prefix = defaultdict(list)
    for p in pts:
        by_prefix[p["prefix"]].append(p)
    for prefix in sorted(by_prefix):
        print(f"\n### prefix = {prefix}\n")
        print("| K | S | K/S | batch | throughput tok/s | latency s/req (256 tok) | peak GB |")
        print("|--:|--:|----:|------:|-----------------:|------------------------:|--------:|")
        for p in sorted(by_prefix[prefix], key=lambda r: (r["K"], r["S"], r["batch"])):
            print(f"| {p['K']} | {p['S']} | {p['eff']:.2g} | {p['batch']} | "
                  f"{p['thr']:.1f} | {p['lat']:.3f} | {p['peak']} |")
    bad = [r for r in rows if r.get("status", "ok") != "ok"]
    if bad:
        print("\n### OOM / skipped / errored — reported, not dropped (demand.md)\n")
        print("| K | S | batch | prefix_len | status |")
        print("|--:|--:|------:|-----------:|:-------|")
        for r in bad:
            print(f"| {r['K']} | {r['S']} | {r['batch']} | {r['prefix_len']} | {r['status']} |")


def plots(rows, csv_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n(matplotlib unavailable: {e}; skipping PNGs)")
        return
    pts = ok_points(rows)
    if not pts:
        print("(no ok rows to plot)")
        return
    outdir = os.path.dirname(os.path.abspath(csv_path)) or "."
    prefixes = sorted({p["prefix"] for p in pts})
    Ks = sorted({p["K"] for p in pts})
    batches = sorted({p["batch"] for p in pts})
    sup = "Qwen3-32B TP=4 (PCIe, no NVLink)"

    def save(fig, name):
        path = os.path.join(outdir, name)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

    # fig1/fig2: metric vs batch — line per K, panel per prefix
    for metric, ylabel, fname in [("thr", "throughput tok/s", "fig1_throughput_vs_batch.png"),
                                  ("lat", "per-request latency (s / 256 tokens)", "fig2_latency_vs_batch.png")]:
        fig, axes = plt.subplots(1, len(prefixes), figsize=(6 * len(prefixes), 4), squeeze=False)
        for ax, prefix in zip(axes[0], prefixes):
            for K in Ks:
                line = sorted([p for p in pts if p["prefix"] == prefix and p["K"] == K and p["S"] == 1],
                              key=lambda r: r["batch"])
                if line:
                    ax.plot([p["batch"] for p in line], [p[metric] for p in line], marker="o", label=f"K={K}")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("batch B"); ax.set_ylabel(ylabel); ax.set_title(f"prefix={prefix}")
            ax.grid(True, alpha=0.3); ax.legend()
        fig.suptitle(f"{sup} — {ylabel} vs batch")
        save(fig, fname)

    # fig3: throughput vs K at fixed (largest) batch, S=1, line per prefix
    fixed_b = max(batches)
    fig, ax = plt.subplots(figsize=(6, 4))
    for prefix in prefixes:
        line = sorted([p for p in pts if p["prefix"] == prefix and p["batch"] == fixed_b and p["S"] == 1],
                      key=lambda r: r["K"])
        if line:
            ax.plot([p["K"] for p in line], [p["thr"] for p in line], marker="o", label=f"prefix={prefix}")
    ax.set_xlabel("K (S=1)"); ax.set_ylabel("throughput tok/s")
    ax.set_title(f"{sup} — throughput vs K @ batch={fixed_b}")
    ax.grid(True, alpha=0.3); ax.legend()
    save(fig, "fig3_throughput_vs_K.png")

    # fig4: equal K/S — wider-K vs more-S, fixed batch + (largest) prefix, line per eff
    prefix0 = prefixes[-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    effs = sorted({p["eff"] for p in pts if p["batch"] == fixed_b and p["prefix"] == prefix0})
    for eff in effs:
        line = sorted([p for p in pts if p["batch"] == fixed_b and p["prefix"] == prefix0
                       and p["eff"] is not None and abs(p["eff"] - eff) < 1e-6], key=lambda r: r["K"])
        if line:
            ax.plot([p["K"] for p in line], [p["thr"] for p in line], marker="o", label=f"K/S={eff:.2g}")
    ax.set_xlabel("K"); ax.set_ylabel("throughput tok/s")
    ax.set_title(f"{sup} — equal K/S: wider-K vs more-S @ batch={fixed_b}, prefix={prefix0}")
    ax.grid(True, alpha=0.3); ax.legend()
    save(fig, "fig4_equal_eff_K_vs_S.png")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/vanilla.csv"
    if not os.path.exists(path):
        sys.exit(f"no results at {path} — run the sweep first")
    rows = load(path)
    tables(rows)
    plots(rows, path)


if __name__ == "__main__":
    main()
