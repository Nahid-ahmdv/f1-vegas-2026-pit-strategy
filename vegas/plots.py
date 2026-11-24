"""Article figures with separate labels for measurements, history and scenarios."""

from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np

from .replay import MEDIUM, RACE_LAPS, SOFT, TARGETS, Strategy, break_even, run_strategy

PAPER = "#fbfaf7"
INK = "#172838"
MUTED = "#475766"
GRID = "#dbdedf"
TEAL = "#147d78"
COPPER = "#a94530"


def _panel(ax):
    ax.set_facecolor(PAPER)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ba8af")
    ax.tick_params(colors=MUTED, length=0, labelsize=10, pad=7)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _heading(fig, label, title, subtitle):
    fig.set_facecolor(PAPER)
    fig.text(0.075, 0.955, label, color=TEAL, weight="bold", fontsize=10)
    fig.text(0.075, 0.882, title, color=INK, family="DejaVu Serif", weight="bold", fontsize=23)
    fig.text(0.075, 0.823, subtitle, color=MUTED, fontsize=11)


def _foot(fig, line_1, line_2):
    fig.text(0.075, 0.076, line_1, color=MUTED, fontsize=9)
    fig.text(0.075, 0.040, line_2, color=MUTED, fontsize=9)


def rejoin_figure(rejoins, observed, bounds):
    names = {"VER": "Max Verstappen", "RUS": "George Russell", "ANT": "Kimi Antonelli"}
    colors = {"VER": "#275b85", "RUS": TEAL, "ANT": COPPER}
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.6), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.982, top=0.683, bottom=0.263, wspace=0.17)
    _heading(
        fig, "LAS VEGAS 2026  /  GREEN-FLAG REJOIN",
        "The first green stop sends the leaders to the midfield",
        "Projected timing-line position when one car pits and the rest stay out.",
    )
    best = int(rejoins.position.min())
    worst = int(rejoins.position.max())
    for driver, ax in zip(TARGETS, axes, strict=True):
        _panel(ax)
        color = colors[driver]
        sample = rejoins.loc[rejoins.driver.eq(driver)]
        envelope = sample.groupby("pit_lap").position.agg(["min", "max"])
        nominal = sample.loc[sample.historical_loss_case.eq("median")].pivot(
            index="pit_lap", columns="spread", values="position",
        )
        ax.axvspan(21.5, 28.5, color="#e9edef", zorder=0)
        ax.fill_between(
            envelope.index, envelope["min"], envelope["max"],
            step="mid", color=color, alpha=0.15, linewidth=0.0, zorder=1,
        )
        ax.step(nominal.index, nominal[0.0], where="mid", color=INK, linestyle="--", linewidth=1.2)
        ax.scatter(
            nominal.index, nominal[0.5], color=color, edgecolor=PAPER, linewidth=0.85,
            s=48, zorder=4,
        )
        ax.set_xlim(18.6, 28.45)
        ax.set_xticks([19, 22, 24, 26, 28])
        ax.set_xlabel("Pit-in lap", fontsize=10, color=MUTED, labelpad=8)
        ax.set_ylim(worst + 0.55, best - 0.55)
        ax.set_yticks(range(best, worst + 1), labels=[f"P{p}" for p in range(best, worst + 1)])
        ax.text(0, 1.17, names[driver], transform=ax.transAxes, color=INK, weight="bold", fontsize=13)
        rank = int(observed.index.get_loc(driver)) + 1
        ax.text(
            0, 1.06, f"Observed lap 18   P{rank}  ·  +{observed.at[driver]:.3f} s",
            transform=ax.transAxes, color=MUTED, fontsize=9.5,
        )
    axes[0].set_ylabel("Simulated rejoin position", color=MUTED, fontsize=10, labelpad=9)
    handles = [
        Line2D([], [], color=INK, linestyle="--", label="Median loss, frozen gaps"),
        Line2D([], [], color=TEAL, marker="o", linestyle="", label="Median loss, half of recent pace drift"),
        Patch(facecolor=TEAL, alpha=0.17, label="Loss range across three pace cases"),
        Patch(facecolor="#e9edef", label="Pirelli's expected window"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.066, 0.122), ncols=4, frameon=False, fontsize=9)
    _foot(
        fig,
        f"Observed seed  ·  FIA lap 18     Historical estimate  ·  2025 green stops, {bounds['low']:.3f}–{bounds['high']:.3f} s effective loss.",
        "Simulation  ·  Add the loss to each car’s projected clock. Pace cases carry forward 0%, 50% and 100% of recent drift.",
    )
    return fig


def break_even_figure(field, settings):
    early = run_strategy(field, settings, Strategy(SOFT, 22))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.982, top=0.704, bottom=0.267, wspace=0.19)
    _heading(
        fig, "LAS VEGAS 2026  /  MATCHED-CAR STRATEGIES",
        "What does a later stop onto Softs need to gain?",
        "Same reference car. Medium → Soft compared with Soft → Medium on lap 22.",
    )
    advances = np.linspace(-0.10, 0.45, 221)
    low, high = -4, 6
    higher_deg = replace(settings, soft_deg=0.05)
    for lap, ax in zip((26, 28), axes, strict=True):
        _panel(ax)
        late = run_strategy(field, settings, Strategy(MEDIUM, lap))
        extra_traffic = late["net_traffic_s"] - early["net_traffic_s"]
        threshold = break_even(field, settings, lap, 0.0)
        threshold_high_deg = break_even(field, higher_deg, lap, 0.0)
        usable_laps = RACE_LAPS - lap
        equal_time = usable_laps * (advances - threshold)
        equal_time_high_deg = usable_laps * (advances - threshold_high_deg)

        ax.fill_between(advances, low, np.clip(equal_time, low, high), color="#d8ebe3")
        ax.fill_between(advances, np.clip(equal_time, low, high), high, color="#f2e5de")
        ax.axhline(0, color="#81919b", linewidth=0.85)
        ax.plot(advances, equal_time, color=INK, linewidth=2.1)
        ax.plot(advances, equal_time_high_deg, color=COPPER, linestyle="--", linewidth=1.5)
        ax.scatter(
            [settings.late_soft_adv_s], [extra_traffic], s=65, color=TEAL,
            edgecolor="white", linewidth=1.25, zorder=4,
        )
        ax.text(-0.082, 4.85, "Early Medium quicker", color="#7d412f", fontsize=10, weight="bold")
        ax.text(0.255, -3.3, "Later Soft quicker", color="#16675e", fontsize=10, weight="bold")
        ax.text(
            0, 1.08, f"Later stop  ·  Lap {lap}", transform=ax.transAxes, color=INK, weight="bold", fontsize=13,
        )
        ax.set_ylim(low, high)
        ax.set_xlim(advances[0], advances[-1])
        ax.set_xticks([-0.10, 0, 0.10, 0.20, 0.30, 0.40])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_locator(FixedLocator([-4, -2, 0, 2, 4, 6]))
        ax.set_xlabel("Warmed Soft advantage at equal age (s/lap)", color=MUTED, fontsize=10, labelpad=8)
    axes[0].set_ylabel("Extra net traffic cost with later Soft (s)", color=MUTED, fontsize=10, labelpad=10)
    handles = [
        Line2D([], [], color=INK, linewidth=2, label="Break-even at central degradation"),
        Line2D([], [], color=COPPER, linestyle="--", label="Break-even with faster Soft degradation"),
        Line2D([], [], color=TEAL, marker="o", linestyle="", label="Central simulation"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.066, 0.122), ncols=3, frameon=False, fontsize=9)
    _foot(
        fig,
        f"Observed seed  ·  Russell’s FIA lap-18 gap, +{field.at['RUS', 'lap18_clock_s']:.3f} s     Historical estimate  ·  {settings.pit_loss_s:.3f} s common two-lap loss, 2025.",
        "Simulation  ·  Full-race time balance. Faster wear doubles the Soft slope to 0.05 s per lap of age. Tow counts as a traffic credit.",
    )
    return fig
