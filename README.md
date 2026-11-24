# What If the Lap-19 Safety Car Had Never Come Out in Las Vegas?

The notebook verifies Mercedes’ 2026 tyre split and compares pitting first under green with saving the Soft for later. The replay starts from the pre-Safety-Car field and uses Russell’s lap-18 clock for both strategy orders.

## Run

Run these commands from the project directory with uv installed. [uv documentation](https://docs.astral.sh/uv/)

```sh
uv sync --locked --python 3.11
uv run --locked python -m vegas.sources
uv run --locked jupyter-execute --inplace --timeout=180 analysis.ipynb
```

The fetch step reuses the downloaded files in `data/raw/`. A fresh checkout downloads public timing files and records their SHA-256 hashes in [data/sources.json](data/sources.json). Use `--refresh` on the fetch command to re-download them. Once the cache and environment are ready, the notebook runs offline. The Buscombe PDF informed the assumptions and is needed only to review her figures.

Open [analysis.ipynb](analysis.ipynb) for the audit, results, and both embedded charts. Running the notebook writes the [2026 driver-by-lap table](outputs/2026_driver_laps.csv), [2025 stop estimates](outputs/2025_green_pit_loss.csv), [lap-18 traffic check](outputs/lap18_rejoin.csv), [strategy comparisons](outputs/strategy_scenarios.csv), and supporting CSVs. It also saves the rejoin-position and later-Soft break-even charts as PNGs in your local `figures/` directory.

## First results

The FIA and OpenF1 agree on **964 lap times after lap 1** and all **23 stop laps and lane durations** in 2026. Russell and Antonelli both entered on lap 19 and began their second stints on lap 20. The official 2025 lap timing independently reproduces the **11-stop median of 21.652 s and range of 19.727–24.918 s**. This is a two-lap excess over green baselines, with out-lap warm-up already inside it.

| Driver | Observed lap 18 | Simulated rejoin at median loss |
| --- | --- | --- |
| Verstappen | P1 | P14, 4.603 s behind Hülkenberg |
| Russell | P2 | P14, 4.610 s behind Hülkenberg |
| Antonelli | P4 | P15, 0.993 s behind Bearman |

Each ranges from P14 to P15 over the 2025 sample. Russell has just **0.206 s** to Bearman at the median loss. The field-spread scenarios change that margin even when the rank holds.

On the matched car, Medium → Soft on lap 28 beats Soft → Medium on lap 22 by **0.10 s** in the central replay. A two-second extra traffic cost pushes the required late Soft advantage to **0.185 s/lap**. Soft wear, quicker Medium warm-up, graining on the opening Medium and tighter traffic can reverse the result.

**Article angle.** The Safety Car spared Mercedes the green-flag choice between keeping track position and learning how the Medium warmed up. Sending Russell in first would have bought that information at the cost of emerging around P14. Saving the Soft gave Antonelli a second option whose value depended on the next car and the tow.

## Cover stress test

Russell pits on lap 22 and Antonelli on lap 28. Cross Verstappen’s same-lap, one-lap and two-lap covers with a seven-car pit wave, using the same historical pit charge for all cars before their compound adjustments. [Central scenarios](outputs/cover_stress_test.csv) · [Pit-loss and field-spread grid](outputs/cover_stress_sensitivity.csv)

Russell rejoins P12–P15 across the grid. A same-lap cover gives zero gain from pit timing. A one-lap delay gives about 0.45 s at the centre and can give virtually zero. A two-lap delay gives 0.70–1.26 s across the grid. The pit wave moves Antonelli’s central rejoin from P14 to P8. With a two-lap cover he rejoins 0.23 s behind Hadjar and waiting costs 0.36 s against his own lap-23 Soft alternative. The matched-car late-Soft gain remains 0.10 s across the central cases and changes sign as pit loss and field spread vary.

## Sources

- FIA History Chart, Lap Analysis, Pit Stop Summary and Race Control Messages for the race, plus FP3 Lap Times. [FIA 2026 event timing](https://www.fia.com/events/fia-formula-one-world-championship/season-2026/las-vegas-grand-prix/eventtiming-information)
- OpenF1 timing, stints and race control for the 2026 race (11420), 2026 FP3 (11415) and 2025 race (9858). [OpenF1 documentation](https://openf1.org/docs/)
- Official 2025 stops, FIA lap timing and VSC records. [Formula 1 pit-stop summary](https://www.formula1.com/en/results/2025/races/1274/las-vegas/pit-stop-summary) · [FIA 2025 event timing](https://www.fia.com/events/fia-formula-one-world-championship/season-2025/las-vegas-grand-prix/eventtiming-information)
- Pirelli’s pre-race window of laps 22–28 and post-race tyre assessment. [Qualifying and strategy report](https://press.pirelli.com/antonelli-goes-all-in-sets-the-fastest-time-on-soft/) · [Race report](https://press.pirelli.com/verstappen-wins-overtaking-spectacle-on-the-strip/)
- Ruth Buscombe’s attached *All on Red* briefing and figures on tyre response, pit timing and the tow.
