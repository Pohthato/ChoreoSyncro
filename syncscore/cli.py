"""Command-line entry point: `syncscore analyze <video> --out results/` and `syncscore validate`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config


def _analyze(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    video = Path(args.video)
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 2
    print(f"analyze: {video} -> {args.out} (pose model {config.pose.model}) - not implemented yet (stage 6)")
    return 1


def _validate(args: argparse.Namespace) -> int:
    from .synth.benchmark import run_benchmark

    summary = run_benchmark(args.out, seeds=args.seeds, jobs=args.jobs, config_path=args.config, quick=args.quick)
    print(f"report: {Path(args.out) / 'report.md'}")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    """A synthetic 5-member group with known errors, through every output."""
    from .outputs import write_outputs
    from .pipeline import analyze_keypoints
    from .synth.generate import GroupSpec, LimbError, make_group

    config = load_config(args.config)
    spec = GroupSpec(
        n_members=5, duration_s=32, fps=30, seed=args.seed, noise=0.01, dropout=0.03, style_deg=3.0,
        offsets_ms=[0, [(0, 0), (16, 0), (32, 120)], -33, 67, 0],  # M2 drifts late in the second half
        limb_errors=[LimbError(member=4, joint="l_elbow", degrees=25.0, t0=6.0, t1=14.0)],
        spins=[(20.0, 0.8)],
    )
    group = make_group(spec)
    result, xy, conf = analyze_keypoints(group.keypoints, group.conf, group.fps, config, names=group.names)
    write_outputs(result, xy, conf, Path(args.out), config, spec.image_size, gif_seconds=8.0)
    print(f"demo written to {args.out} (truth: M2 0 -> +120 ms after 16 s, M3 -33 ms, M4 +67 ms, M5 left elbow +25 deg 6-14 s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="syncscore", description="Measure dance-group synchronization from one video.")
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Analyze a video")
    analyze.add_argument("video")
    analyze.add_argument("--out", default="results")
    analyze.set_defaults(run=_analyze)

    validate = commands.add_parser("validate", help="Run the synthetic recovery benchmark")
    validate.add_argument("--out", default="results/validate")
    validate.add_argument("--seeds", type=int, default=3, help="random choreographies per case")
    validate.add_argument("--jobs", type=int, default=None, help="parallel processes (default: cores - 1)")
    validate.add_argument("--quick", action="store_true", help="one seed for the failure sweeps")
    validate.set_defaults(run=_validate)

    demo = commands.add_parser("demo", help="Run a synthetic group with known errors through every output")
    demo.add_argument("--out", default="results/demo")
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(run=_demo)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
