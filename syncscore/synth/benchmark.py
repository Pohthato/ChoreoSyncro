"""Validation benchmark: does the method recover offsets and pose errors that we planted?

Everything after pose extraction runs exactly as on video: synthetic keypoints -> smoothing ->
bone directions -> consensus -> timing -> shape. Pose model and tracker errors are *not*
covered here. They are simulated only roughly (noise, dropouts, occlusion, left/right swaps).

Metrics, all against the truth relative to the median member (the method's reference):
- window MAE: mean |estimated - true offset| over windows the method marked reliable.
- coverage: share of windows marked reliable. The method is allowed to say "don't know".
- confidently wrong: share of *reliable* windows that are off by more than 30 ms. This is the
  dangerous failure. Not knowing is fine, being wrong and sure is not.
"""
from __future__ import annotations

import dataclasses
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config, load_config
from ..features.bones import BONE_NAMES, PART_OF_BONE, PARTS
from ..pipeline import analyze_keypoints
from .generate import GroupSpec, Interval, LimbError, make_group

WRONG_MS = 30.0
REALISTIC = dict(noise=0.01, dropout=0.03, style_deg=3.0)


@dataclass
class Case:
    sweep: str
    label: str
    value: float
    spec: GroupSpec
    affected: list = field(default_factory=list)  # (member or None for all, t0, t1): windows to score separately
    config_overrides: dict = field(default_factory=dict)  # {"sync.lag_penalty": 0.0}


def _with_overrides(config: Config, overrides: dict) -> Config:
    for key, value in overrides.items():
        section, name = key.split(".")
        config = dataclasses.replace(config, **{section: dataclasses.replace(getattr(config, section), **{name: value})})
    return config


def _window_truth(group, result) -> np.ndarray:
    frames = np.arange(group.keypoints.shape[1])
    truth = np.stack([np.interp(result.window_centers_s * group.fps, frames, group.offsets_ms[i]) for i in range(len(group.names))])
    return truth - np.median(truth, axis=0)


def _scores(err: np.ndarray, reliable: np.ndarray, mask: np.ndarray) -> dict:
    sel = reliable & mask
    e = np.abs(err[sel])
    return {
        "windows": int(mask.sum()),
        "coverage": float(sel.sum() / max(mask.sum(), 1)),
        "mae_ms": float(e.mean()) if e.size else None,
        "p95_ms": float(np.percentile(e, 95)) if e.size else None,
        "confidently_wrong": float((e > WRONG_MS).mean()) if e.size else None,
    }


def evaluate(case: Case, config: Config) -> dict:
    group = make_group(case.spec)
    result, _, _ = analyze_keypoints(group.keypoints, group.conf, group.fps, _with_overrides(config, case.config_overrides))
    truth = _window_truth(group, result)
    est = np.nan_to_num(result.offset_ms)
    err = est - truth
    reliable = result.reliable
    out = {"sweep": case.sweep, "label": case.label, "value": case.value, "seed": case.spec.seed}
    everything = np.ones_like(reliable)
    if case.affected:
        hit = np.zeros_like(reliable)
        half = result.window_s / 2
        for member, t0, t1 in case.affected:
            rows = slice(None) if member is None else member
            overlap = (result.window_centers_s + half > t0) & (result.window_centers_s - half < t1)
            hit[rows, overlap] = True
        out["affected"] = _scores(err, reliable, hit)
        out["unaffected"] = _scores(err, reliable, ~hit)
    out["all"] = _scores(err, reliable, everything)
    with np.errstate(all="ignore"):
        member_est = np.array([np.nanmedian(np.where(reliable[i], result.offset_ms[i], np.nan)) for i in range(len(group.names))])
    member_truth = np.median(truth, axis=1)
    out["members"] = [{"true_ms": float(t), "estimated_ms": None if not np.isfinite(e) else float(e)} for t, e in zip(member_truth, member_est)]
    limbs = []
    for error, bone in zip(case.spec.limb_errors, group.limb_error_bones):
        frames = slice(int((error.t0 + 0.5) * group.fps), int((error.t1 - 0.5) * group.fps))
        bone_err = np.nanmean(result.bone_shape_deg[error.member, frames], axis=0)
        others = [i for i in range(len(group.names)) if i != error.member]
        part_err = np.nanmean(result.part_shape_deg[error.member, frames], axis=0)
        limbs.append({
            "member": error.member, "true_bone": bone, "true_part": PARTS[PART_OF_BONE[BONE_NAMES.index(bone)]], "true_deg": error.degrees,
            "measured_deg": float(bone_err[BONE_NAMES.index(bone)]),
            "worst_bone": BONE_NAMES[int(np.nanargmax(bone_err))],
            "worst_part": PARTS[int(np.nanargmax(part_err))],
            "baseline_deg": float(np.nanmean(result.bone_shape_deg[others][:, frames, BONE_NAMES.index(bone)])),
        })
    out["limbs"] = limbs
    return out


def headline_cases(seeds: int) -> list[Case]:
    """The requested test: one dancer's motion copied to 7 bodies, shifted by known amounts,
    with a known pose error on one limb (plus a second member with both timing and shape errors)."""
    cases = []
    offsets = [0, 0, 0, -33, 33, 67, 133]
    limbs = [LimbError(member=1, joint="l_elbow", degrees=20.0, t0=10.0, t1=20.0), LimbError(member=5, joint="r_knee", degrees=15.0, t0=24.0, t1=34.0)]
    for condition, extra in (("clean", {}), ("realistic", REALISTIC)):
        for seed in range(seeds):
            spec = GroupSpec(n_members=7, duration_s=40, fps=30, offsets_ms=offsets, limb_errors=limbs, seed=100 + seed, **extra)
            cases.append(Case("headline", condition, 0, spec))
    return cases


def sweep_cases(seeds: int) -> list[Case]:
    base = dict(n_members=5, duration_s=40, fps=30, offsets_ms=[0, 33, -33, 67, 0])
    cases = []
    for seed in range(seeds):
        s = 200 + seed
        for noise in (0.0, 0.005, 0.01, 0.02, 0.04):
            cases.append(Case("noise", f"{noise * 100:g}% of height", noise * 100, GroupSpec(**base, noise=noise, dropout=0.03, style_deg=3.0, seed=s)))
        for fps in (60, 30, 24, 15, 10):
            cases.append(Case("frame_rate", f"{fps} fps", fps, GroupSpec(**{**base, "fps": fps}, **REALISTIC, seed=s)))
        for length in (0.25, 0.5, 1.0, 2.0, 4.0):
            gaps = [Interval(1, t, t + length) for t in (8.0, 20.0, 32.0)]
            cases.append(Case("occlusion", f"{length:g} s hidden", length, GroupSpec(**base, **REALISTIC, occlusions=gaps, seed=s), affected=[(1, g.t0, g.t1) for g in gaps]))
        cases.append(Case("leaves_frame", "leaves for 10 s", 10, GroupSpec(**base, **REALISTIC, occlusions=[Interval(3, 15.0, 25.0)], seed=s), affected=[(3, 15.0, 25.0)]))
        cases.append(Case("leaves_frame", "legs cut off 20 s", 20, GroupSpec(**base, **REALISTIC, cropped=[Interval(2, 10.0, 30.0)], seed=s), affected=[(2, 10.0, 30.0)]))
        spins = [(8.0, 0.8), (18.0, 0.8), (28.0, 0.8)]
        spin_variants = (("spin, labels kept", False), ("spin, every member's labels swap", True), ("spin, 2 of 5 members' labels swap", [False, True, False, True, False]))
        for value, (label, flip) in enumerate(spin_variants):
            cases.append(Case("spin", label, float(value), GroupSpec(**base, **REALISTIC, spins=spins, spin_label_flip=flip, seed=s), affected=[(None, t0 - 0.2, t0 + d + 0.2) for t0, d in spins]))
        for penalty in (0.0, 0.1):
            label = f"repeated groove, jump penalty {penalty:g}"
            spec = GroupSpec(**base, **REALISTIC, loops=[(12.0, 30.0, 0.18, 2)], seed=s)
            cases.append(Case("repeated_groove", label, penalty, spec, affected=[(None, 12.0, 30.0)], config_overrides={"sync.lag_penalty": penalty}))
        for degrees in (5.0, 10.0, 15.0, 20.0, 30.0):
            spec = GroupSpec(**base, **REALISTIC, limb_errors=[LimbError(1, "l_elbow", degrees, 10.0, 30.0)], seed=s)
            cases.append(Case("limb_error", f"{degrees:g} deg", degrees, spec))
    return cases


def _run(args):
    case, config_path = args
    return evaluate(case, load_config(config_path))


def run_benchmark(out_dir: str | Path, seeds: int = 3, jobs: int | None = None, config_path: str | None = None, quick: bool = False) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = headline_cases(seeds) + sweep_cases(1 if quick else seeds)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    print(f"running {len(cases)} synthetic groups on {jobs} processes ...")
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(_run, [(case, config_path) for case in cases]))
    (out / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    from .report import write_report

    return write_report(rows, out)
