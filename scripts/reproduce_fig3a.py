from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.datasets import gaussian_ring_clusters
from src.directions import random_unit_vectors, dad_direction_uniform
from src.counting_map import counting_map_on_grid
from src.plot_utils import plot_origin_hyperplane

def main():
    ap = argparse.ArgumentParser(description="Reproduce Fig. 3(a): random vs DAD hyperplane and counting maps.")
    ap.add_argument("--dataset-seed", type=int, default=0)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--n-per", type=int, default=100)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--noise-std", type=float, default=0.15)
    ap.add_argument("--k", type=int, default=512, help="Number of DAD direction candidates.")
    ap.add_argument("--grid", type=int, default=500, help="Grid size for counting map plotting.")
    ap.add_argument("--out", type=str, default="", help="Optional output path (png/pdf).")
    args = ap.parse_args()

    X, _centers = gaussian_ring_clusters(
        seed=args.dataset_seed,
        n_clusters=args.n_clusters,
        n_per_cluster=args.n_per,
        radius=args.radius,
        noise_std=args.noise_std,
    )
    rng = np.random.default_rng(args.rng_seed)

   # Random vector: keep sampling until slope condition matches the paper demo
    while True:
        v_rand = random_unit_vectors(rng, 2, 1)[0]
        slope = abs(v_rand[1] / (v_rand[0] + 1e-12))
        if slope > 3:
            break
    
    
    
    # DAD direction
    v_dad = dad_direction_uniform(X, rng, n_dir_candidates=args.k)

    # projections + counting maps
    y_rand = X @ v_rand
    y_dad = X @ v_dad
    x_rand, d_rand = counting_map_on_grid(y_rand, n_grid=args.grid)
    x_dad, d_dad = counting_map_on_grid(y_dad, n_grid=args.grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left: scatter + hyperplanes
    axL = axes[0]
    pad = 1.0
    xmin, xmax = (X[:,0].min()-pad, X[:,0].max()+pad)
    ymin, ymax = (X[:,1].min()-pad, X[:,1].max()+pad)
    lim_min, lim_max = min(xmin, ymin), max(xmax, ymax)
    xlim = ylim = (lim_min, lim_max)

    axL.scatter(X[:,0], X[:,1], s=10, color="black", alpha=0.9)
    plot_origin_hyperplane(axL, v_rand, xlim, ylim, color="tab:blue", linestyle="--", alpha=0.35, lw=1.8, label="random hyperplane")
    plot_origin_hyperplane(axL, v_dad,  xlim, ylim, color="tab:red",  linestyle="-.", alpha=0.45, lw=2.0, label="DAD hyperplane")
    axL.set_xlim(xlim); axL.set_ylim(ylim)
    axL.set_aspect("equal", adjustable="box")
    axL.set_xlabel("x1"); axL.set_ylabel("x2")
    axL.legend(loc="upper left", frameon=False)

    # right: counting maps
    axR = axes[1]
    axR.plot(x_rand, d_rand, color="tab:blue", lw=2.2, label=r"$d_Y(p)$ (random)")
    axR.plot(x_dad,  d_dad,  color="tab:red",  lw=2.2, label=r"$d_Y(p)$ (DAD)")
    axR.axhline(0, lw=1, alpha=0.4)
    axR.scatter(y_rand, np.zeros_like(y_rand), s=8, color="tab:blue", alpha=0.4)
    axR.scatter(y_dad,  np.zeros_like(y_dad),  s=8, color="tab:red",  alpha=0.4)
    axR.set_xlabel(r"$p$ (projection coordinate)")
    axR.set_ylabel(r"$d_Y(p)$")
    axR.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    main()