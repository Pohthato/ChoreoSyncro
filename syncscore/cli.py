"""Command-line entry point: `syncscore analyze <video> --out results/`, `syncscore validate`, and
`syncscore serve` (the same analysis started from a page in the browser)."""
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
    from .pose.recover import fill_occlusions, redetect
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
    if args.pose_only:
        from .pose.prepare import prepare

        prepare(video, out, det, config, info=info)
        suggested = json.loads((out / "pick_frames.json").read_text(encoding="utf-8"))["suggested"]
        print(f"ready to name the members (suggested frames: {', '.join(map(str, suggested))})")
        return 0
    tracklets = build_tracklets(det, min_frames=max(3, int(0.3 * info.fps)))
    following = fingerprinter = allow = None
    if args.members:
        from .pose.fingerprint import Fingerprinter
        from .pose.identity import allow_for, follow_members
        from .pose.picks import PickError, load_picks, resolve_picks
        from .pose.prepare import prepare

        try:
            picks = load_picks(args.members, det.n_frames)
            picked_rows = resolve_picks(picks, det)
        except PickError as error:
            print(error, file=sys.stderr)
            return 2
        camera, det.vectors = prepare(video, out, det, config, info=info)
        following = follow_members(det, tracklets, camera, picked_rows, config)
        tracks = assemble(det, following.slots, keep_order=True)
        allow = allow_for(following.recognizer)
        fingerprinter = Fingerprinter.load(config.identity.model) if config.tracking.recover else None
        names = [pick.name for pick in picks.members]
    else:
        members = config.tracking.members or count_members(det, min_conf)
        if members < 2:
            print(f"Found {members} clearly visible people; need at least 2 (set tracking.members in the config).", file=sys.stderr)
            return 3
        tracks = assemble(det, assign_members(det, tracklets, members, config.tracking.stitch_max_gap_s))
        names = [f"M{i + 1}" for i in range(len(tracks.xy))]
    # Occlusion: fill members' short gaps with real detections, first ones the assignment left
    # unused, then from a more sensitive second look at just the frames still missing someone.
    cache = out / "recovered.npz"
    if args.redo_pose:
        cache.unlink(missing_ok=True)
    second_look = (lambda frames: redetect(video, frames, config, cache, info.size, info.fps, det.n_frames,
                                           fingerprinter)) if config.tracking.recover else None
    tracks, filling = fill_occlusions(det, tracks, config.tracking.fill_max_gap_s, min_conf, second_look, allow)
    before, after = filling["before"], filling["after"]
    from_leftovers, from_redetection = filling["filled_from_leftovers"], filling["filled_from_redetection"]
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
    }
    if following is None:
        swaps = suspected_swaps(tracks.hist, tracks.box, info.fps, config.tracking.crossing_iou)
        report |= {
            "suspected_swaps": [{**s, "members": [names[k] for k in s["members"]]} for s in swaps],
            "suspected_swaps_per_minute": round(len(swaps) / (info.n_frames / info.fps / 60), 3),
            "note": "Suspected swaps come from clothing colour at crossings and miss swaps between members in matching outfits. "
                    "The true rate comes from `syncscore audit` (hand-labelled frames).",
        }
    else:
        found = following.report
        report["identity"] = {
            "picked_frame": found["picked_frame"], "model": config.identity.model,
            "tracked_share": {n: round(float((tracks.rows[i] >= 0).mean()), 3) for i, n in enumerate(names)},
            "splits": found["splits"],
            "close_calls": [{**c, "member": None if c["member"] is None else names[c["member"]]} for c in found["close_calls"]],
            "position_rejections": [{**r, "member": names[r["member"]]} for r in found["position_rejections"]],
            "taken_under_margin": [{**t, "member": names[t["member"]]} for t in found["taken_under_margin"]],
        }
    report |= {
        "variable_frame_rate_resampled": info.variable,
        "coverage": {"before": before, "after": after, "filled_from_leftovers": from_leftovers, "filled_from_redetection": from_redetection},
    }
    (out / "tracking_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_outputs(result, xy_s, conf_s, out, config, info.size, gif_seconds=8.0)
    best = int(np.argmax(visible.sum(axis=1)))
    raw = np.where((conf[best, :, 9] >= min_conf)[:, None], xy[best, :, 9], np.nan)
    render_smoothing_plot(raw, xy_s[best, :, 9], fps, out / "smoothing_example.png", f"{names[best]} left wrist: raw vs smoothed")
    if following is None:
        print(f"  {len(names)} members, {report['rejoins']} broken tracks rejoined, {len(swaps)} suspected ID swaps")
    else:
        print(f"  {len(names)} members followed by name from frame {following.report['picked_frame']}, "
              f"{len(following.report['splits'])} tracks split where they changed hands")
    print(f"  everyone tracked in {after['all_members_share']:.1%} of frames (was {before['all_members_share']:.1%}); "
          f"filled {from_leftovers} member-frames from unused detections, {from_redetection} from re-detection")
    print(f"results in {out}")
    return 0


def _audit(args: argparse.Namespace) -> int:
    """Write sampled frames with every person drawn (members by label, everyone else ?1, ?2...),
    plus a CSV to fill in by hand."""
    import csv

    import cv2
    import numpy as np

    from .pose.tracks import Detections, _iou, audit_frames, parse_window
    from .pose.video import read_frames

    results = Path(args.results)
    data = np.load(results / "keypoints_raw.npz")
    box, names, fps = data["box"], [str(n) for n in data["names"]], float(data["fps"])
    try:
        windows = [parse_window(w) for w in args.times]
    except ValueError as error:
        print(f"--times: {error}", file=sys.stderr)
        return 2
    seen = np.isfinite(box[..., 0]).any(axis=0)
    picks = audit_frames(seen, fps, args.frames, windows, args.step)
    det = Detections.load(results / "detections.npz") if (results / "detections.npz").exists() else None
    out = results / "audit"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    rows = []
    for f, frame in read_frames(cap, picks):
        candidates = np.flatnonzero(det.frame == f) if det is not None else np.zeros(0, int)
        used = set()
        for i, name in enumerate(names):
            if np.isfinite(box[i, f]).all():
                overlaps = _iou(box[i, f][None], det.box[candidates]) if len(candidates) else np.zeros(0)
                match = int(candidates[np.argmax(overlaps)]) if len(overlaps) and np.nanmax(overlaps) >= 0.9 else -1
                used.add(match)
                x1, y1, x2, y2 = box[i, f].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(frame, name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
                rows.append({"frame": int(f), "row": match, "shown_label": name, "true_name": ""})
        others = sorted((r for r in candidates if int(r) not in used), key=lambda r: det.box[r, 0] + det.box[r, 2])
        for n, r in enumerate(others, start=1):
            x1, y1, x2, y2 = det.box[r].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
            cv2.putText(frame, f"?{n}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 2)
            rows.append({"frame": int(f), "row": int(r), "shown_label": f"?{n}", "true_name": ""})
        cv2.imwrite(str(out / f"frame_{int(f):06d}.jpg"), frame)
    cap.release()
    with open(out / "ids.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "row", "shown_label", "true_name"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(picks)} frames in {out}. Fill in true_name in ids.csv (dancer for anyone who is not a member), "
          f"then run: syncscore audit-score {out / 'ids.csv'}")
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


def _id_gate(args: argparse.Namespace) -> int:
    """Measure how well each re-ID model names hand-labelled people (see syncscore/pose/gate.py)."""
    import csv
    import json
    import time

    import numpy as np

    from .pose.fingerprint import Fingerprinter, fingerprint_video, load_fingerprints, save_fingerprints
    from .pose.gate import margin_grid
    from .pose.prepare import camera_path
    from .pose.tracks import Detections, detections_key

    config = load_config(args.config)
    results = Path(args.results)
    det = Detections.load(results / "detections.npz")
    with open(args.labels, newline="", encoding="utf-8") as handle:
        labels = {int(r["row"]): r["true_name"].strip() for r in csv.DictReader(handle)
                  if r["true_name"].strip() and int(r["row"]) >= 0}
    camera = camera_path(args.video, results, det)
    report_path = results / "gate.json"
    previous = json.loads(report_path.read_text(encoding="utf-8")).get("models", {}) if report_path.exists() else {}
    report = {"picked_frame": args.frame, "labelled_rows": len(labels), "models": {}}
    for name in args.model:
        path = results / f"gate_fingerprints_{name}.npz"
        vectors = load_fingerprints(path, name, detections_key(det))
        timing = {k: previous.get(name, {}).get(k) for k in ("ms_per_crop", "fingerprint_s")}
        if vectors is None:
            start = time.perf_counter()
            vectors = fingerprint_video(args.video, det, Fingerprinter.load(name))
            seconds = time.perf_counter() - start
            save_fingerprints(path, vectors, name, detections_key(det))
            crops = max(int(np.any(vectors != 0, axis=1).sum()), 1)
            timing = {"ms_per_crop": round(1000 * seconds / crops, 2), "fingerprint_s": round(seconds, 1)}
        grid = margin_grid(det, vectors, labels, args.frame, config, camera)
        report["models"][name] = {**timing, "margins": grid}
        print(f"{name}: {timing['ms_per_crop']} ms per crop, {timing['fingerprint_s']} s for the clip")
        for margin, r in grid.items():
            print(f"  margin {margin}: members right {r['member_correct']:.1%}, unassigned {r['member_unassigned']:.1%}, "
                  f"dancers named {r['dancer_as_member']:.1%}  per member "
                  + ", ".join(f"{k} {v:.0%}" for k, v in r["per_member"].items()))
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written to {report_path}")
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

    for kind, label in (("section", "sections from windows"), ("hit_section", "sections from hits"), ("combined_section", "sections, either")):
        print(f"{label}: measured {pct(result[f'{kind}_coverage'])}, error {ms(result[f'{kind}_mae_ms'])}, "
              f"more than 30 ms wrong {pct(result[f'{kind}_wrong'])}, truth inside the 90% interval {pct(result[f'{kind}_within_interval'])}")
    print(f"single windows (strict gates): measured {pct(result['window_coverage'])}, error {ms(result['window_mae_ms'])}, "
          f"more than 30 ms wrong {pct(result['window_wrong'])}")
    print(f"single hits: measured {pct(result['hit_coverage'])}, error {ms(result['hit_mae_ms'])} (median {ms(result['hit_median_abs_ms'])}), "
          f"more than 30 ms wrong {pct(result['hit_wrong'])}")
    return 0


def _view(args: argparse.Namespace) -> int:
    """Write the viewer for an analysed video: open index.html in any browser, no server needed."""
    from .pose.video import probe
    from .viz.viewer_data import write_viewer

    config = load_config(args.config)
    video, results = Path(args.video), Path(args.results)
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 2
    if not (results / "sync_result.npz").is_file():
        print(f"No analysis in {results}: run `syncscore analyze` first.", file=sys.stderr)
        return 2
    index = write_viewer(results, Path(args.out) if args.out else results / "viewer", config, probe(video).size, video, args.video_url)
    print(f"open {index}")
    print("The viewer plays the video from where it is now and never copies it. It stays out of the repository.")
    return 0


def _serve(args: argparse.Namespace) -> int:
    """A page on this computer to upload a video, follow its analysis and open its viewer."""
    from .serve.app import serve

    return serve(Path(args.root), args.config, args.port, open_browser=not args.no_browser)


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
    from .viz.viewer_data import write_viewer

    index = write_viewer(Path(args.out), Path(args.out) / "viewer", config, spec.image_size, None)
    print(f"demo written to {args.out} (truth: M2 0 -> +120 ms after 16 s, M3 -33 ms, M4 +67 ms, M5 left elbow +25 deg 6-14 s)")
    print(f"open {index} to see it (skeletons only: a synthetic group has no video)")
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
    naming = analyze.add_mutually_exclusive_group()
    naming.add_argument("--pose-only", action="store_true",
                        help="find people, follow the camera and fingerprint them, then stop: ready to pick the members")
    naming.add_argument("--members", default=None, metavar="PICKS",
                        help="a picks file (frame and a named box per member): follow those members by name")
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
    audit.add_argument("--times", nargs="*", default=[], help="windows to sample densely, in seconds, e.g. 148-150")
    audit.add_argument("--step", type=float, default=0.2, help="seconds between frames inside a window")
    audit.set_defaults(run=_audit)

    id_gate = commands.add_parser("id-gate", help="Measure how well re-ID fingerprints name hand-labelled people")
    id_gate.add_argument("video")
    id_gate.add_argument("--results", default="results")
    id_gate.add_argument("--labels", required=True, help="a filled-in audit ids.csv")
    id_gate.add_argument("--frame", type=int, required=True, help="the labelled frame that stands in for the picks")
    id_gate.add_argument("--model", action="append", required=True, help="a key of syncscore.pose.fingerprint.MODELS")
    id_gate.set_defaults(run=_id_gate)

    audit_score = commands.add_parser("audit-score", help="Score a filled-in audit CSV")
    audit_score.add_argument("csv")
    audit_score.set_defaults(run=_audit_score)

    view = commands.add_parser("view", help="Write a viewer (open index.html in a browser): skeletons and ghosts on the video")
    view.add_argument("video")
    view.add_argument("--results", default="results", help="the folder `analyze` wrote")
    view.add_argument("--out", default=None, help="where to write the viewer (default: <results>/viewer)")
    view.add_argument("--video-url", default=None, help="the URL the page plays the video from (default: the video's file:// path)")
    view.set_defaults(run=_view)

    serve = commands.add_parser("serve", help="Upload videos and open their viewers in the browser (runs on this computer only)")
    serve.add_argument("--root", default="results/uploads", help="where uploads and their analyses are kept")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-browser", action="store_true", help="don't open the page in the browser")
    serve.set_defaults(run=_serve)

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
