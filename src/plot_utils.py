from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def plot_origin_hyperplane(ax, v: np.ndarray, xlim, ylim, *, color, linestyle, alpha=0.35, lw=2.0, label=None):
    """
    Plot hyperplane through origin with normal v: {x: v^T x = 0}.
    In 2D this is a line through the origin.
    """
    vx, vy = float(v[0]), float(v[1])
    if abs(vy) < 1e-12:
        ax.plot([0.0, 0.0], ylim, color=color, linestyle=linestyle, linewidth=lw, alpha=alpha, label=label)
    else:
        xs = np.linspace(xlim[0], xlim[1], 200)
        ys = -(vx / vy) * xs
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=lw, alpha=alpha, label=label)
        
    

def plot_row(*, Z_list, grid_x, grid_y, X_train, filename, titles,
             show_points=True, global_scale=False, global_vmin=None, global_vmax=None):
    n_cols = len(Z_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    for j in range(n_cols):
        Z = Z_list[j]

        if global_scale:
            vmin, vmax = global_vmin, global_vmax
            img = Z
        else:
            vmin, vmax = float(Z.min()), float(Z.max())
            img = (Z - vmin) / (vmax - vmin + 1e-12)

        ax = axes[j]
        im = ax.imshow(
            img,
            origin="lower",
            extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
            cmap="inferno",
            interpolation="nearest",
            aspect="equal",
            vmin=None if not global_scale else vmin,
            vmax=None if not global_scale else vmax,
        )

        if show_points:
            ax.scatter(
                X_train[:, 0], X_train[:, 1],
                s=10, c="white", edgecolors="black", linewidths=0.3, alpha=0.9
            )

        ax.set_title(titles[j], fontsize=11)
        ax.set_xlabel("$x_1$")
        if j == 0:
            ax.set_ylabel("$x_2$")
        else:
            ax.set_yticklabels([])

    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("Anomaly score")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()