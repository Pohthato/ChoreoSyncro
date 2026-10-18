"""Command-line entry point: `syncscore analyze <video> --out results/` and `syncscore validate`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config


def _analyze(args: argparse.Namespace) -> int:
    import json

    import numpy as np

    from .outputs import write_outputs
    from .pipeline import analyze_keypoints
    from .pose.extract import extract
    from .pose.tracks import Detections, assemble, assign_members, build_tracklets, count_members, suspected_swaps
    from .pose.video import probe, resample_uniform
    from .viz.smoothing_plot import render_smoothing_plot

    config = load_config(args.config)
    video, out = Path(args.video), Path(args.out)
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    print(f"{video.name}: {info.n_frames} frames, {info.fps:.2f} fps{' (variable)' if info.variable else ''}, {info.size[0]}x{info.size[1]}")

    cached = out / "detections.npz"
    if cached.exists() and not args.redo_pose:
        det = Detections.load(cached)
        print("  using cached detections (pass --redo-pose to run the model again)")
    else:
        det = extract(video, config, info.n_frames, info.fps, info.size)
        det.save(cached)
    min_conf = config.pose.min_keypoint_conf
    members = config.tracking.members or count_members(det, min_conf)
    if members < 2:
        print(f"Found {members} clearly visible people; need at least 2 (set tracking.members in the config).", file=sys.stderr)
        return 3
    tracklets = build_tracklets(det, min_frames=max(3, int(0.3 * info.fps)))
    tracks = assemble(det, assign_members(det, tracklets, members, config.tracking.stitch_max_gap_s))
    swaps = suspected_swaps(tracks.hist, tracks.box, info.fps, config.tracking.crossing_iou)
    names = [f"M{i + 1}" for i in range(len(tracks.xy))]
    np.savez_compressed(out / "keypoints_raw.npz", xy=tracks.xy.astype(np.float32), conf=tracks.conf.astype(np.float32),
                        box=tracks.box.astype(np.float32), fps=info.fps, names=np.array(names))

    xy, conf, fps = tracks.xy, tracks.conf, info.fps
    if info.variable:
        xy, conf, _ = resample_uniform(xy, conf, info.times_ms, fps)
    result, xy_s, conf_s = analyze_keypoints(xy, conf, fps, config, names=names)
    visible = (conf >= min_conf).sum(axis=2) >= 8
    report = {
        "members": len(names), "tracker_ids": int(len(np.unique(det.track_id[det.track_id >= 0]))),
        "fragments": {n: f for n, f in zip(names, tracks.fragments)},
        "rejoins": int(sum(len(f) - 1 for f in tracks.fragments)),
        "visible_share": {n: round(float(v.mean()), 3) for n, v in zip(names, visible)},
        "suspected_swaps": [{**s, "members": [names[k] for k in s["members"]]} for s in swaps],
        "suspected_swaps_per_minute": round(len(swaps) / (info.n_frames / info.fps / 60), 3),
        "note": "Suspected swaps come from clothing colour at crossings and miss swaps between members in matching outfits. "
                "The true rate comes from `syncscore audit` (hand-labelled frames).",
        "variable_frame_rate_resampled": info.variable,
    }
    (out / "tracking_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_outputs(result, xy_s, conf_s, out, config, info.size, gif_seconds=8.0)
    best = int(np.argmax(visible.sum(axis=1)))
    raw = np.where((conf[best, :, 9] >= min_conf)[:, None], xy[best, :, 9], np.nan)
    render_smoothing_plot(raw, xy_s[best, :, 9], fps, out / "smoothing_example.png", f"{names[best]} left wrist: raw vs smoothed")
    print(f"  {len(names)} members, {report['rejoins']} broken tracks rejoined, {len(swaps)} suspected ID swaps")
    print(f"results in {out}")
    return 0


def _audit(args: argparse.Namespace) -> int:
    """Write sampled frames with member labels drawn, plus a CSV to fill in by hand."""
    import csv

    import cv2
    import numpy as np

    data = np.load(Path(args.results) / "keypoints_raw.npz")
    box, names = data["box"], [str(n) for n in data["names"]]
    seen = np.isfinite(box[..., 0]).any(axis=0)
    frames = np.flatnonzero(seen)
    picks = frames[np.linspace(0, len(frames) - 1, min(args.frames, len(frames))).astype(int)]
    out = Path(args.results) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    rows = []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            continue
        for i, name in enumerate(names):
            if np.isfinite(box[i, f]).all():
                x1, y1, x2, y2 = box[i, f].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(frame, name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
                rows.append({"frame": int(f), "shown_label": name, "true_name": ""})
        cv2.imwrite(str(out / f"frame_{int(f):06d}.jpg"), frame)
    with open(out / "ids.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "shown_label", "true_name"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(picks)} frames in {out}. Fill in true_name in ids.csv, then run: syncscore audit-score {out / 'ids.csv'}")
    print("These frames contain the source footage: they stay in results/ (git-ignored) and must not be committed.")
    return 0


def _audit_score(args: argparse.Namespace) -> int:
    import json

    import numpy as np

    from .pose.tracks import score_audit

    csv_path = Path(args.csv)
    fps = float(np.load(csv_path.parent.parent / "keypoints_raw.npz")["fps"])
    result = score_audit(csv_path, fps)
    (csv_path.parent / "audit_score.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


def _shift_test(args: argparse.Namespace) -> int:
    """Plant known delays in real members' tracks and check they come back (no labels needed)."""
    import json

    import numpy as np

    from .synth.real_shift import real_shift_test

    config = load_config(args.config)
    data = np.load(Path(args.results) / "keypoints_raw.npz")
    xy, conf, fps = data["xy"].astype(float), data["conf"].astype(float), float(data["fps"])
    members = [m - 1 for m in args.members]
    print(f"planting {args.shifts} ms on members {args.members}: {len(members) * (len(args.shifts) + 1)} full analyses ...")
    result = real_shift_test(xy, conf, fps, config, members=members, shifts=args.shifts, jobs=args.jobs)
    (Path(args.results) / "shift_test.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    def pct(v):
        return "-" if v is None else f"{v * 100:.0f}%"

    def ms(v):
        return "-" if v is None else f"{v:.1f} ms"

    print(f"sections: measured {pct(result['section_coverage'])}, error {ms(result['section_mae_ms'])}, "
          f"more than 30 ms wrong {pct(result['section_wrong'])}, truth inside the 90% interval {pct(result['section_within_interval'])}")
    print(f"single windows (strict gates): measured {pct(result['window_coverage'])}, error {ms(result['window_mae_ms'])}, "
          f"more than 30 ms wrong {pct(result['window_wrong'])}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    from .synth.benchmark import run_benchmark

    summary = run_benchmark(args.out, seeds=args.seeds, jobs=args.jobs, config_path=args.config, quick=args.quick)
    print(f"report: {Path(args.out) / 'report.md'}")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    """A synthetic 5-member group with known errors, through every output."""
    import dataclasses

    import numpy as np

    from .outputs import write_outputs
    from .pipeline import analyze_keypoints
    from .synth.generate import GroupSpec, LimbError, make_group

    config = load_config(args.config)
    # Synthetic timing spreads far less than real dance, so short sections are measurable here
    # (5 independent blocks each). Real footage needs about 30 s (docs/walkthrough.md, Stage A).
    config = dataclasses.replace(config, sections=dataclasses.replace(config.sections, chunk_s=10.0))
    spec = GroupSpec(
        n_members=5, duration_s=32, fps=30, seed=args.seed, noise=0.01, dropout=0.03, style_deg=3.0,
        offsets_ms=[0, [(0, 0), (16, 0), (32, 120)], -33, 67, 0],  # M2 drifts late in the second half
        limb_errors=[LimbError(member=4, joint="l_elbow", degrees=25.0, t0=6.0, t1=14.0)],
        spins=[(20.0, 0.8)],
    )
    group = make_group(spec)
    result, xy, conf = analyze_keypoints(group.keypoints, group.conf, group.fps, config, names=group.names)
    write_outputs(result, xy, conf, Path(args.out), config, spec.image_size, gif_seconds=8.0)
    from .viz.smoothing_plot import render_smoothing_plot

    raw = np.where((group.conf[0, :, 9] >= config.pose.min_keypoint_conf)[:, None], group.keypoints[0, :, 9], np.nan)
    render_smoothing_plot(raw, xy[0, :, 9], group.fps, Path(args.out) / "smoothing_example.png", "M1 left wrist (synthetic, 1% noise): raw vs smoothed")
    print(f"demo written to {args.out} (truth: M2 0 -> +120 ms after 16 s, M3 -33 ms, M4 +67 ms, M5 left elbow +25 deg 6-14 s)")
    return 0


def _safe_console() -> None:
    """Never crash on printing: Windows consoles are often cp1252 and video names are often not
    (e.g. Korean titles). Unprintable characters become '?' instead of raising."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _safe_console()
    parser = argparse.ArgumentParser(prog="syncscore", description="Measure dance-group synchronization from one video.")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Analyze a video")
    analyze.add_argument("video")
    analyze.add_argument("--out", default="results")
    analyze.add_argument("--redo-pose", action="store_true", help="ignore cached detections")
    analyze.set_defaults(run=_analyze)

    validate = commands.add_parser("validate", help="Run the synthetic recovery benchmark")
    validate.add_argument("--out", default="results/validate")
    validate.add_argument("--seeds", type=int, default=3, help="random choreographies per case")
    validate.add_argument("--jobs", type=int, default=None, help="parallel processes (default: cores - 1)")
    validate.add_argument("--quick", action="store_true", help="one seed for the failure sweeps")
    validate.set_defaults(run=_validate)

    audit = commands.add_parser("audit", help="Sample frames with member labels for a hand-labelled ID check")
    audit.add_argument("video")
    audit.add_argument("--results", default="results")
    audit.add_argument("--frames", type=int, default=30)
    audit.set_defaults(run=_audit)

    audit_score = commands.add_parser("audit-score", help="Score a filled-in audit CSV")
    audit_score.add_argument("csv")
    audit_score.set_defaults(run=_audit_score)

    shift = commands.add_parser("shift-test", help="Plant known delays in real tracks from `analyze` and check they are recovered")
    shift.add_argument("--results", default="results")
    shift.add_argument("--members", type=int, nargs="+", default=[2, 3, 4], help="member numbers (1-based)")
    shift.add_argument("--shifts", type=float, nargs="+", default=[33.0, 67.0, 133.0])
    shift.add_argument("--jobs", type=int, default=8)
    shift.set_defaults(run=_shift_test)

    demo = commands.add_parser("demo", help="Run a synthetic group with known errors through every output")
    demo.add_argument("--out", default="results/demo")
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(run=_demo)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
