import argparse
import numpy as np

from src.datasets import make_lattice_set, make_grid_square_set
from src.models.iforest import IsolationForestAD
from src.models.eif import ObliqueIForest
from src.models.scif import SCiForest
from src.plot_utils import plot_row


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use-lattice", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--jitter", type=float, default=0.03)
    p.add_argument("--high-res", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--global-scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--show-points", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--n-trees", type=int, default=1000)
    p.add_argument("--sample-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-dir-candidates", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1.0)

    p.add_argument("--outdir", type=str, default=".")
    return p.parse_args()


def main():
    args = parse_args()

    from pathlib import Path

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # train set
    if args.use_lattice:
        X_train = make_lattice_set()
    else:
        X_train = make_grid_square_set(jitter=args.jitter, seed=args.seed)

    # grid
    if args.high_res:
        grid_x = np.linspace(-1.0, 13.0, 600)
        grid_y = np.linspace(-1.0, 13.0, 600)
    else:
        grid_x = np.linspace(-1.0, 13.0, 60)
        grid_y = np.linspace(-1.0, 13.0, 60)

    GX, GY = np.meshgrid(grid_x, grid_y)
    G = np.c_[GX.ravel(), GY.ravel()]

    common = dict(
        n_trees=args.n_trees,
        sample_size=args.sample_size,
        random_seed=args.seed,
        n_dir_candidates=args.n_dir_candidates,
        alpha=args.alpha,
    )

    models = [
        ("IF", IsolationForestAD(mode="IF", **common)),
        ("IF + DAS", IsolationForestAD(mode="IF+DAS", **common)),
        ("EIF", ObliqueIForest(mode="EIF", **common)),
        ("EIF + DAS", ObliqueIForest(mode="EIF+DAS", **common)),
        ("EIF + DAD", ObliqueIForest(mode="EIF+DAD", **common)),
        ("EIF + DAS + DAD", ObliqueIForest(mode="EIF+DAS+DAD", **common)),
        ("SCiF", SCiForest(mode="SCIF", **common)),
        ("SCiF + DAD", SCiForest(mode="SCIF+DAD", **common)),
    ]

    Z_list = []
    all_scores = []

    for _, mdl in models:
        mdl.fit(X_train)
        Z = mdl.anomaly_score(G).reshape(GX.shape)
        Z_list.append(Z)
        all_scores.append(Z.ravel())

    if args.global_scale:
        all_scores = np.concatenate(all_scores)
        vmin = all_scores.min()
        vmax = all_scores.max()
    else:
        vmin = vmax = None

    out_if = f"{args.outdir}/IF_row.png"
    out_eif = f"{args.outdir}/EIF_row.png"
    out_scif = f"{args.outdir}/SCiF_row.png"

    plot_row(
        Z_list=Z_list[:2],
        grid_x=grid_x,
        grid_y=grid_y,
        X_train=X_train,
        filename=out_if,
        titles=["IF", "IF + DAS"],
        show_points=args.show_points,
        global_scale=args.global_scale,
        global_vmin=vmin,
        global_vmax=vmax,
    )

    plot_row(
        Z_list=Z_list[2:6],
        grid_x=grid_x,
        grid_y=grid_y,
        X_train=X_train,
        filename=out_eif,
        titles=["EIF", "EIF + DAS", "EIF + DAD", "EIF + DAS + DAD"],
        show_points=args.show_points,
        global_scale=args.global_scale,
        global_vmin=vmin,
        global_vmax=vmax,
    )

    plot_row(
        Z_list=Z_list[6:8],
        grid_x=grid_x,
        grid_y=grid_y,
        X_train=X_train,
        filename=out_scif,
        titles=["SCiF", "SCiF + DAD"],
        show_points=args.show_points,
        global_scale=args.global_scale,
        global_vmin=vmin,
        global_vmax=vmax,
    )


if __name__ == "__main__":
    main()
