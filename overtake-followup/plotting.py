"""Plot the reconstructed and FIA-recorded gaps on one scale."""

import matplotlib.pyplot as plt
import numpy as np


def gap_figure(laps, source, path):
    ink = "#1d2635"
    observed = "#304b7a"
    estimated = "#a65d3c"
    matched = "#007e70"
    neutral = "#5d6676"
    fig, (ax, order) = plt.subplots(
        2, 1, figsize=(12, 5.65), gridspec_kw={"height_ratios": [10, 1]}, sharex=True
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.21, hspace=0.1)
    fig.text(0.10, 0.932, "Verstappen and Russell before the Safety Car", fontsize=21,
             weight="bold", color=ink)
    fig.text(0.10, 0.885, "2026 Las Vegas Grand Prix   ·   RUS crossing time − VER crossing time",
             fontsize=11.5, color=neutral)

    ax.axhspan(-1.0, 0, color="#ebf5f2", zorder=0)
    ax.axhline(0, color=ink, lw=0.9)
    ax.axhline(-1.0, color=matched, linestyle=":", lw=1)
    ax.grid(axis="y", color="#e4e6ec", linewidth=0.7, zorder=0)
    for row in laps.itertuples():
        color = matched if row.pattern == "yes" else estimated
        detection = row.detection_rus_minus_ver_s
        finish = row.finish_rus_minus_ver_s
        if np.isfinite(detection):
            ax.plot([row.lap, row.lap], [detection, finish], color="#c8ced8", lw=1.0, zorder=1)
            ax.vlines(row.lap, row.detection_gap_low_s, row.detection_gap_high_s,
                      color=color, lw=1.35, alpha=0.85, zorder=2)
            ax.scatter(row.lap, detection, s=50, color=color, edgecolor="white", linewidth=0.7, zorder=4)
        if np.isfinite(finish):
            ax.scatter(row.lap, finish, s=43, marker="s", facecolor="white", edgecolor=observed,
                       linewidth=1.5, zorder=5)
        leader = row.detection_leader
        suffix = "*" if row.detection_leader_uncertain else ""
        order.text(row.lap, 0, f"{leader}{suffix}", ha="center", va="center",
                   color=matched if leader == "RUS" else neutral, fontsize=8.5, weight="bold")

    ax.scatter([], [], color=estimated, label="Detection estimate")
    ax.scatter([], [], color=matched, label="RUS ahead, VER qualifies")
    ax.scatter([], [], marker="s", facecolors="white", edgecolors=observed,
               linewidth=1.5, label="FIA finish timing")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.005, 1.01), ncols=3, frameon=False,
              fontsize=9.4, columnspacing=2.25)
    worst_high = laps.detection_gap_high_s.replace(np.inf, np.nan).max()
    ax.set_ylim(min(-1.13, laps.detection_gap_low_s.replace(-np.inf, np.nan).min() - 0.09),
                max(1.68, worst_high + 0.14))
    ax.set_xlim(0.48, 18.52)
    ax.set_ylabel("Crossing gap (s)", fontsize=10, color=ink)
    ax.tick_params(axis="x", bottom=False)
    ax.tick_params(axis="y", length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(18.43, -0.93, "VER within 1 s of RUS", ha="right", color=matched, fontsize=8.5)

    order.set_ylim(-0.78, 0.75)
    order.set_xticks(laps.lap)
    order.tick_params(axis="x", length=0, labelsize=9, pad=3)
    order.set_yticks([])
    order.text(-0.015, 0.5, "Detection\nleader", transform=order.transAxes, va="center",
               ha="right", fontsize=8.5, color=neutral)
    for spine in order.spines.values():
        spine.set_visible(False)
    order.set_xlabel("Lap", fontsize=10, color=ink, labelpad=7)

    fig.text(0.10, 0.123, "Above zero, Verstappen crossed first. At the finish he led on all 18 laps.",
             fontsize=9, color=ink)
    fig.text(0.10, 0.086, "Whiskers show sensitivity to line placement and timing. * Hadjar and Verstappen overlap on lap 8.",
             fontsize=9, color=neutral)
    fig.text(0.10, 0.049, f"Sources   FIA Race History Chart and Sector Analysis  ·  {source} race position data",
             fontsize=8.7, color=neutral)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
