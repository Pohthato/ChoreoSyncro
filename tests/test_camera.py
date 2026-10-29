import cv2
import numpy as np

from syncscore.pose.camera import CameraPath, estimate_frames, still_camera, to_stage

SIZE = (640, 360)


def _background(seed: int) -> np.ndarray:
    noise = np.random.default_rng(seed).uniform(0, 255, (1200, 1600)).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (0, 0), 3)
    return cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _view(f: int, pan: float, zoom: float) -> np.ndarray:
    """Background -> frame f: centre the background, shift by pan * f px, zoom by zoom ** f about the frame centre."""
    z = zoom ** f
    cx, cy = SIZE[0] / 2, SIZE[1] / 2
    shift = np.array([cx - 800 + pan * f, cy - 600])
    return np.array([[z, 0, cx + z * (shift[0] - cx)], [0, z, cy + z * (shift[1] - cy)]])


def _apply(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ m[:, :2].T + m[:, 2]


def _frames(background, n, pan=0.0, zoom=1.0):
    return [cv2.warpAffine(background, _view(f, pan, zoom), SIZE, flags=cv2.INTER_LINEAR) for f in range(n)]


def _no_people(n):
    return [np.zeros((0, 4))] * n


def test_a_pan_and_zoom_are_recovered():
    cam = estimate_frames(_frames(_background(0), 60, pan=3.0, zoom=1.003), _no_people(60))
    grid = np.stack(np.meshgrid(np.linspace(700, 900, 5), np.linspace(520, 680, 5)), -1).reshape(-1, 2)
    stage = _apply(_view(0, 3.0, 1.003), grid)
    for f in range(60):
        seen = _apply(_view(f, 3.0, 1.003), grid)
        assert np.abs(to_stage(cam, np.full(len(grid), f), seen) - stage).max() < 1.5, f
    assert (cam.segment == 0).all()


def test_a_fixed_camera_gives_identity_transforms():
    cam = estimate_frames(_frames(_background(1), 30), _no_people(30))
    assert np.abs(cam.transform - np.array([[1, 0, 0], [0, 1, 0]])).max() < 1e-3


def test_a_hard_cut_starts_a_new_segment():
    frames = _frames(_background(2), 20) + _frames(_background(3), 20)
    cam = estimate_frames(frames, _no_people(40))
    assert (cam.segment[:20] == 0).all() and (cam.segment[20:] == 1).all()


def _person_scene(background: np.ndarray, n: int):
    """A textured 200x300 block moving 5 px a frame, inside a person box covering about 30% of the frame."""
    block = _background(9)[:300, :200]
    frames, boxes = [], []
    for f in range(n):
        image = background.copy()
        x = 100 + 5 * f
        image[30:330, x:x + 200] = block
        frames.append(image)
        boxes.append(np.array([[x - 10, 20, x + 210, 340]], float))
    return frames, boxes


def test_moving_people_do_not_pull_the_estimate():
    background = _frames(_background(4), 1)[0]
    cam = estimate_frames(*_person_scene(background, 30))
    assert np.abs(cam.transform[:, :, 2]).max() < 0.5
    assert (cam.segment == 0).all()


def test_a_plain_backdrop_with_a_fixed_camera_stays_one_segment():  # Review Focus 5
    grey = np.full(SIZE[::-1], 128, np.uint8)
    cam = estimate_frames(*_person_scene(grey, 30))
    assert (cam.segment == 0).all()
    assert np.abs(cam.transform - np.array([[1, 0, 0], [0, 1, 0]])).max() < 1e-6


def test_the_path_is_saved_with_the_detections_key(tmp_path):
    cam = still_camera(5)
    cam.save(tmp_path / "camera.npz", "abc")
    loaded = CameraPath.load(tmp_path / "camera.npz", "abc")
    assert np.array_equal(loaded.transform, cam.transform) and np.array_equal(loaded.segment, cam.segment)
    assert CameraPath.load(tmp_path / "camera.npz", "other") is None
    assert CameraPath.load(tmp_path / "missing.npz", "abc") is None


def test_a_frame_almost_covered_by_people_is_not_a_cut():
    # A dancer fills all but a thin strip at the right edge; an arm sweeping through that strip
    # changes most of the background that is left. That is not a new shot.
    background = _frames(_background(5), 1)[0]
    frames, boxes = [], []
    arm = _background(6)[:SIZE[1], :40]
    for f in range(20):
        image = background.copy()
        if f >= 10:
            image[:, 600:] = arm
        frames.append(image)
        boxes.append(np.array([[0, 0, 560, 360]], float))
    cam = estimate_frames(frames, boxes)
    assert (cam.segment == 0).all()
