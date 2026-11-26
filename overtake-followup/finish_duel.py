"""Plot the late-lap VER–RUS crossing-gap estimates for laps 36 and 37."""

import os

import numpy as np
import pandas as pd

from collect import CACHE, HERE, RESTART_CACHE, official_timing
from crossings import DETECTION_M, FIA_KNOTS, align_to_s3, build_course, cross_gate

os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAPS = (36, 37)
END_M = FIA_KNOTS[-1]
S2_M = FIA_KNOTS[2]
# Rounded 95th-percentile absolute pair-clock error from the restart checks.
RESTART_TIMING_MARGIN_S = 0.144
COLORS = {"OpenF1": "#365980", "Steele replay": "#ab581c"}


def trace_at_gates(lap, samples, course, fia_distances):
    """Use the restart analysis's forward XY crossing and S2-to-finish pin."""
    distance = course.from_fia(fia_distances)
    expected = lap.finish_s - course.time[-1] + np.interp(distance, course.distance, course.time)
    raw_s2 = cross_gate(course, course.knots[2], samples, lap.s2_crossing_s, 2.2)[0]
    raw_line = cross_gate(course, course.knots[3], samples, lap.finish_s, 2.2)[0]
    raw = np.array([
        cross_gate(course, d, samples, time, 6)[0] for d, time in zip(distance, expected, strict=True)
    ])
    pinned = np.array([align_to_s3(time, raw_s2, raw_line, lap) for time in raw])
    return pinned, raw


def sign_brackets(gap):
    """Take the nearest opposite-signed samples, including across rounded zeros."""
    points = gap.dropna()
    points = points.loc[points.ne(0)]
    changes = points.iloc[1:].to_numpy() * points.iloc[:-1].to_numpy() < 0
    return [
        (points.index[i], points.index[i + 1])
        for i in np.flatnonzero(changes)
    ]


def finish_duel():
    timing, _ = official_timing(max_lap=max(LAPS))
    early = pd.read_csv(CACHE / "18_location.csv.gz")
    restart = pd.read_csv(CACHE / f"{RESTART_CACHE}_location.csv.gz")
    drivers = ("VER", "RUS")
    pilot = timing.loc[timing.lap.eq(2)].set_index("driver")
    courses = {
        driver: build_course(pilot.loc[driver],
                             early.loc[early.car.eq(pilot.loc[driver, "driver_number"])])
        for driver in drivers
    }
    grid = np.arange(DETECTION_M - 60, END_M, 5)
    replay = pd.read_csv(HERE / "finish_duel_steele.csv")
    replay["fia_m"] = S2_M + (replay.app_distance_m - replay.app_s2_m) * (
        END_M - S2_M
    ) / (replay.app_finish_m - replay.app_s2_m)
    panels = {}
    for number in LAPS:
        lap = timing.loc[timing.lap.eq(number)].set_index("driver")
        results = {}
        for name, course in courses.items():
            ver, rus = (
                trace_at_gates(lap.loc[driver], restart.loc[restart.car.eq(lap.loc[driver, "driver_number"])],
                               course, grid)
                for driver in drivers
            )
            results[f"{name} reference"] = rus[0] - ver[0]
            if name == "VER":
                results["Raw UTC"] = rus[1] - ver[1]
        openf1 = pd.DataFrame(results, index=grid)
        start_gap = round(lap.loc["RUS", "start_s"] - lap.loc["VER", "start_s"], 3)
        finish_gap = round(lap.loc["RUS", "finish_s"] - lap.loc["VER", "finish_s"], 3)
        native = replay.loc[replay.lap.eq(number)].sort_values("fia_m").set_index("fia_m")
        app_gap = (native.lap_relative_rus_minus_ver_s + start_gap).round(3)
        np.testing.assert_allclose(app_gap.iloc[-1], finish_gap, atol=0.001)
        near_turn = native.replay_landmark.eq("approaching T17")
        turn_bounds = (
            native.index[near_turn].max(),
            native.index[(native.index > native.index[near_turn].max())
                         & native.replay_landmark.eq("approaching T1")].min(),
        )
        brackets = {name: sign_brackets(openf1[name]) for name in openf1}
        brackets["Steele replay"] = sign_brackets(app_gap)
        print(f"Lap {number} sign-change brackets, FIA metres")
        for name, bounds in brackets.items():
            print(" ", name, ", ".join(f"{a:.1f}–{b:.1f}" for a, b in bounds) or "none")
        print(f"  Turn 17 replay landmark {turn_bounds[0]:.1f}–{turn_bounds[1]:.1f} m")
        print(f"  FIA finish {finish_gap:+.3f} s")
        if number == 36:
            latest_consensus = openf1.index[openf1.gt(RESTART_TIMING_MARGIN_S).all(axis=1)].max()
            print(f"  Clock-margin bracket {latest_consensus:.0f}–{END_M:.0f} m "
                  "(OpenF1 variants clear +0.144 s, then the FIA finish confirms the reversal)")
        panels[number] = (openf1["VER reference"], app_gap, finish_gap, np.mean(turn_bounds))

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.102, right=0.968, top=0.773, bottom=0.174, hspace=0.28)
    fig.text(0.102, 0.947, "Russell vs. Verstappen from detection to finish", fontsize=19, weight="bold")
    fig.text(0.102, 0.895, "2026 Las Vegas GP   ·   RUS crossing time − VER crossing time",
             fontsize=11, color="#465365")
    for ax, (number, (openf1, app, finish, turn)) in zip(axes, panels.items(), strict=True):
        ax.axhline(0, color="#333c46", linewidth=1)
        for distance, color, style in [(DETECTION_M, "#758190", "--"), (turn, "#7e6d94", ":"),
                                       (END_M, "#333c46", "-")]:
            ax.axvline(distance / 1000, color=color, linestyle=style, linewidth=1)
        ax.plot(openf1.index / 1000, openf1, color=COLORS["OpenF1"], linewidth=1.45,
                label="OpenF1 estimate")
        ax.plot(app.index / 1000, app, color=COLORS["Steele replay"], linewidth=1.65,
                marker=".", markersize=3.2, label="Steele replay estimate")
        ax.scatter([END_M / 1000], [finish], color="#20262e", zorder=5, s=24, label="FIA finish")
        for name, series in [("OpenF1", openf1), ("Steele replay", app)]:
            for left, right in sign_brackets(series):
                ax.hlines(0, left / 1000, right / 1000, color=COLORS[name], linewidth=3, zorder=6)
        ax.set_title(f"Lap {number}", loc="left", fontsize=11, weight="bold", pad=7)
        ax.set_xlim(5.22, 6.225)
        ax.set_ylim(-0.72, 0.66)
        ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6])
        ax.grid(axis="y", linewidth=0.55, color="#dee3e9")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for x, label, align in [(DETECTION_M + 5, "Detection", "left"),
                             (panels[36][3], "Turn 17 (approx.)", "center"),
                             (END_M, "Finish", "center")]:
        axes[0].text(x / 1000, 1.018, label, transform=axes[0].get_xaxis_transform(),
                     fontsize=9, ha=align, va="bottom", color="#465365")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.16, 1.18), frameon=False, ncols=3, fontsize=9)
    axes[1].set_xlabel("FIA-mapped lap distance (km)", labelpad=9)
    fig.supylabel("Signed gap (s)", x=0.034)
    fig.text(0.102, 0.086, "Positive gaps favour Verstappen. Coloured bars mark the samples bracketing each zero crossing.",
             fontsize=9.2, color="#465365")
    fig.text(0.102, 0.049, "Turn 17 is mapped from the replay landmark and checked against the circuit map and XY bend.",
             fontsize=9.2, color="#465365")
    fig.savefig(HERE / "laps_36_37_gap.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    finish_duel()
