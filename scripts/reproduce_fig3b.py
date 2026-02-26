from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.datasets import gaussian_ring_clusters
from src.directions import random_unit_vectors, dad_direction_uniform
from src.plot_utils import plot_origin_hyperplane

def main():
    ap = argparse.ArgumentParser(description="Reproduce Fig. 3(b): overlay of random vs DAD hyperplanes.")
    ap.add_argument("--dataset-seed", type=int, default=0)
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--n-per", type=int, default=100)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--noise-std", type=float, default=0.15)

    ap.add_argument("--n-seeds", type=int, default=100, help="Number of independent RNG seeds (0..n_seeds-1).")
    ap.add_argument("--base-seed", type=int, default=0, help="Start seed for the sweep.")
    ap.add_argument("--k", type=int, default=64, help="Number of DAD direction candidates (n_dir_candidates).")
    ap.add_argument("--out", type=str, default="", help="Optional output path (png/pdf).")
    args = ap.parse_args()

    # dataset (matches your notebook default)
    X, _ = gaussian_ring_clusters(
        seed=args.dataset_seed,
        n_clusters=args.n_clusters,
        n_per_cluster=args.n_per,
        radius=args.radius,
        noise_std=args.noise_std,
    )

    # axis limits (match your notebook logic)
    pad = 1.0
    xmin, xmax = (min(X[:, 0].min(), -0.5) - pad, max(X[:, 0].max(), 0.5) + pad)
    ymin, ymax = (min(X[:, 1].min(), -0.5) - pad, max(X[:, 1].max(), 0.5) + pad)
    lim_min = min(xmin, ymin)
    lim_max = max(xmax, ymax)
    xlim = ylim = (lim_min, lim_max)

    # figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.9, color="black")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.axhline(0, color="lightgray", lw=1, alpha=0.5)
    ax.axvline(0, color="lightgray", lw=1, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")

    # overlay: for each seed, create its own RNG (this is key to matching)
    for s in range(args.base_seed, args.base_seed + args.n_seeds):
        rng = np.random.default_rng(s)
        v_rand = random_unit_vectors(rng, 2, 1)[0]
        v_dad = dad_direction_uniform(X, rng, n_dir_candidates=args.k)
        plot_origin_hyperplane(ax, v_rand, xlim, ylim, color="tab:blue", linestyle="--", alpha=0.25, lw=1.8)
        plot_origin_hyperplane(ax, v_dad,  xlim, ylim, color="tab:red",  linestyle="-.", alpha=0.35, lw=2.0)

    proxies = [
        Line2D([0], [0], color="tab:blue", linestyle="--", linewidth=2.0, alpha=0.8, label="random"),
        Line2D([0], [0], color="tab:red",  linestyle="-.", linewidth=2.0, alpha=0.8, label="density-aware"),
    ]
    ax.legend(handles=proxies, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    if args.out:
        plt.savefig(args.out, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    main()