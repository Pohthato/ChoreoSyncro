"""The beat grid from a soundtrack: generated click tracks with a known tempo and phase."""
import numpy as np
import pytest

from syncscore.audio.beats import beat_grid, load_audio

SR = 22050


def _clicks(tempo_bpm, first_beat_s, seconds, sr=SR, seed=0):
    """Kick (low thump + click) on every beat, a quieter hi-hat on every off-beat, faint noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.002, int(seconds * sr))
    period = 60.0 / tempo_bpm
    k = np.arange(int(0.08 * sr)) / sr
    kick = np.sin(2 * np.pi * 60 * k) * np.exp(-k / 0.03)
    kick[: int(0.003 * sr)] += rng.normal(0, 0.8, int(0.003 * sr))
    h = np.arange(int(0.03 * sr)) / sr
    hat = rng.normal(0, 0.25, h.size) * np.exp(-h / 0.008)
    for t in np.arange(first_beat_s, seconds - 0.1, period):
        i = int(t * sr)
        x[i:i + kick.size] += kick
        j = int((t + period / 2) * sr)
        if j + hat.size < x.size:
            x[j:j + hat.size] += hat
    return x.astype(np.float32)


def test_tempo_and_phase_of_a_click_track():
    grid = beat_grid(_clicks(120.0, 0.31, 60), SR)
    assert grid is not None and grid.steady
    assert abs(grid.tempo_bpm - 120.0) < 0.3
    assert abs(((grid.beat_phase_s - 0.31 + 0.25) % 0.5) - 0.25) < 0.010


def test_the_beat_not_the_off_beat():
    grid = beat_grid(_clicks(120.0, 0.31, 60), SR)
    on = [t for t, is_beat in grid.beats_between(10.0, 12.0) if is_beat]
    assert all(abs(((t - 0.31 + 0.25) % 0.5) - 0.25) < 0.010 for t in on)
    assert len(grid.beats_between(10.0, 12.0)) == 8  # four beats and four "and"s


def test_another_tempo():
    grid = beat_grid(_clicks(96.0, 0.12, 60), SR)
    assert grid.steady and abs(grid.tempo_bpm - 96.0) < 0.3


def test_a_tempo_change_is_not_steady():
    x = np.concatenate([_clicks(120.0, 0.2, 30), _clicks(100.0, 0.05, 30, seed=1)])
    grid = beat_grid(x, SR)
    assert grid is None or not grid.steady


def test_silence_gives_no_grid():
    grid = beat_grid(np.zeros(SR * 30, dtype=np.float32), SR)
    assert grid is None or not grid.steady


def test_offsets_are_to_the_nearest_half_beat():
    grid = beat_grid(_clicks(120.0, 0.31, 60), SR)
    base = 0.31 + 20 * 0.5
    off = grid.offset_ms(np.array([base, base + 0.02, base + 0.25 - 0.03]))
    assert np.allclose(off, [0, 20, -30], atol=10)


def test_load_audio_reads_an_encoded_file(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "tone.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("aac", rate=SR)
        samples = (0.3 * np.sin(2 * np.pi * 440 * np.arange(2 * SR) / SR)).astype(np.float32)
        for start in range(0, samples.size, 1024):
            frame = av.AudioFrame.from_ndarray(samples[None, start:start + 1024], format="fltp", layout="mono")
            frame.rate = SR
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    x = load_audio(path, SR)
    assert x is not None and abs(x.size / SR - 2.0) < 0.1


def test_no_audio_stream_gives_none(tmp_path):
    import cv2

    path = tmp_path / "silent.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 24))
    for _ in range(30):
        writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()
    assert load_audio(path, SR) is None


def test_a_late_tempo_change_is_not_steady():
    # Two thirds at 120 BPM, the last third at 100: most tracked beats still sit on one grid, but the
    # grid is wrong for the whole last third. Steadiness is judged per stretch of the song too.
    x = np.concatenate([_clicks(120.0, 0.2, 40), _clicks(100.0, 0.05, 20, seed=1)])
    grid = beat_grid(x, SR)
    assert grid is None or not grid.steady
