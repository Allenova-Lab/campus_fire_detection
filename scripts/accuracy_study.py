"""
How accurate is FirePath?  Drop fires at random spots on campus, run the whole
chain with the live AI, and compare where the system says the fire is with
where we really put it.  Writes docs/accuracy_study.md for the report.

    python -m scripts.accuracy_study            # 20 fires
    python -m scripts.accuracy_study 50 --seed 7
"""

import argparse
import random
import statistics
from pathlib import Path

from firepath import detect as det
from firepath.geometry import point_in_polygon
from firepath.pipeline import Settings, run_all
from firepath.simulate import DEFAULT_SPRITES

OUT = Path(__file__).resolve().parent.parent / "docs" / "accuracy_study.md"
CAMPUS_X, CAMPUS_Y = (80, 850), (-140, 350)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--scenario", default="campus_watch")
    a = ap.parse_args()

    model = det.load_model()
    if model is None:
        raise SystemExit("the accuracy study needs the live AI (ultralytics + weights)")
    rng = random.Random(a.seed)
    rows = []
    for k in range(a.n):
        fire = (round(rng.uniform(*CAMPUS_X), 1), round(rng.uniform(*CAMPUS_Y), 1))
        sprite = DEFAULT_SPRITES[k % len(DEFAULT_SPRITES)]
        res = run_all(fire, Settings(scenario=a.scenario, sprite=sprite, mode="live"), model=model)
        inside = bool(res.region) and point_in_polygon(fire, res.region)
        rows.append((k + 1, fire, sprite, res, inside))
        print(f"{k + 1:3d}  fire ({fire[0]:6.1f}, {fire[1]:6.1f})  "
              + (f"error {res.error_m:5.1f} m  cams {len(res.used)}  inside {inside}"
                 if res.located else f"NOT LOCATED: {res.message}"))

    found = [r for r in rows if r[3].located]
    errs = sorted(r[3].error_m for r in found)
    lines = [
        "# FirePath accuracy study", "",
        f"{a.n} fires dropped at random on campus (x {CAMPUS_X[0]}–{CAMPUS_X[1]} m, "
        f"y {CAMPUS_Y[0]}–{CAMPUS_Y[1]} m, seed {a.seed}), scenario `{a.scenario}`, "
        f"fire width {rows[0][3].fire_width:.0f} m.  Every photo went through the live "
        "YOLO26 fire model; the geometry only saw the AI's boxes.", "",
        "The camera images are **simulated**: a drawn campus with a real flame photograph "
        "pasted in where the geometry says it belongs.  That is what makes the error "
        "measurable — we know exactly where each fire really was.", "",
        "## Summary", "",
        f"- Located: **{len(found)} / {a.n}** fires "
        f"({100 * len(found) / a.n:.0f}%)",
    ]
    if errs:
        p90 = errs[min(len(errs) - 1, int(round(0.9 * (len(errs) - 1))))]
        lines += [
            f"- Error: median **{statistics.median(errs):.1f} m**, mean {statistics.mean(errs):.1f} m, "
            f"90th percentile {p90:.1f} m, worst {errs[-1]:.1f} m",
            f"- True fire inside the reported region: "
            f"{sum(r[4] for r in found)} / {len(found)}",
            f"- Median region size: {statistics.median(r[3].region_ha for r in found):.2f} ha",
        ]
    tested = [r[3] for r in found if r[3].dof]
    if tested:
        ok = sum(r.chi2 < r.chi2_limit for r in tested)
        lines.append(f"- Cameras-agree check (χ², 95%, box edges assumed to wobble "
                     f"{rows[0][3].settings.sigma_u:g} px): passed {ok} / {len(tested)} "
                     f"runs with 3+ cameras (about 95% expected if the wobble is right)")
    lines += ["", "## Every fire", "",
              "| # | placed at (m) | flame photo | cameras used | found at (m) | error (m) "
              "| region (ha) | inside region | note |",
              "|---|---|---|---|---|---|---|---|---|"]
    for i, fire, sprite, res, inside in rows:
        if res.located:
            lines.append(f"| {i} | ({fire[0]:.1f}, {fire[1]:.1f}) | {sprite} | {len(res.used)} "
                         f"| ({res.estimate[0]:.1f}, {res.estimate[1]:.1f}) | {res.error_m:.1f} "
                         f"| {res.region_ha:.3f} | {'yes' if inside else 'no'} | |")
        else:
            lines.append(f"| {i} | ({fire[0]:.1f}, {fire[1]:.1f}) | {sprite} | {len(res.used)} "
                         f"| — | — | — | — | {res.message} |")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
