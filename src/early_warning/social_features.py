"""Interpretable social-signal features for escalation early warning.

Input is the multi-person pose format written by tracked_pose.py:

    keypoints  (T, M, 17, 2) float32  COCO-17 order, pixel coordinates
    scores     (T, M, 17)    float32  per-joint confidence, 0 where absent
    track_ids  (T, M)        int32    -1 where the slot is empty

Each slot m must hold the same tracked person over time.

Every distance is divided by body height, so a feature means the same thing
for a near camera and a far one, and for children and adults. Speeds are in
body heights per second. Geometry is in the image plane, which compresses
depth; a per-camera floor homography is the planned upgrade.

Phase mapping:
    precursor  mutual facing at close range, gesturing while standing still,
               arm raise, one person advancing while the other gives ground
    build-up   encirclement (occupied o-space), outnumbering, retreat then
               stall (cornering), contact followed by recoil (shove),
               outsiders converging on a group
    assault    left to the existing AGCN detector
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

NOSE, L_SH, R_SH, L_WR, R_WR = 0, 5, 6, 9, 10
L_HIP, R_HIP, L_ANK, R_ANK = 11, 12, 15, 16

PERSON_KEYS = ("speed", "gesture", "arm_raise", "heading_conf")
TARGET_KEYS = ("n_near", "n_facing", "coverage", "enclosure", "cornered", "retreat")
EDGE_KEYS = (
    "dist",
    "closing",
    "advance",
    "recoil",
    "facing",
    "mutual_facing",
    "reach",
    "contact",
    "shove",
)
SCENE_KEYS = ("n_people", "largest_group", "converging", "ring_still", "kinetic")


@dataclass
class FeatureConfig:
    fps: float = 10.0
    conf_thr: float = 0.3
    near_radius: float = 1.5  # body heights: inside the interaction space
    contact_thr: float = 0.15  # wrist-to-torso distance that counts as contact
    min_move_speed: float = 0.25  # body heights / s
    facing_thr: float = 0.5  # cosine, roughly +/- 60 degrees
    smooth_s: float = 0.3
    height_smooth_s: float = 1.0
    corner_window_s: float = 3.0


@dataclass
class PersonGeometry:
    valid: np.ndarray  # (T, M) bool
    foot: np.ndarray  # (T, M, 2) ground contact point, px
    torso_c: np.ndarray  # (T, M, 2) torso centre, px
    height: np.ndarray  # (T, M) body height, px; NaN where invalid
    heading: np.ndarray  # (T, M, 2) unit facing vector, image plane
    heading_conf: np.ndarray  # (T, M) 0..1
    vel: np.ndarray  # (T, M, 2) foot velocity, body heights / s
    speed: np.ndarray  # (T, M)
    gesture: np.ndarray  # (T, M) wrist speed relative to the torso, heights / s
    arm_raise: np.ndarray  # (T, M) 1.0 if a wrist is above the shoulder line
    wrists: np.ndarray  # (T, M, 2, 2) px


@dataclass
class FeatureBundle:
    geometry: PersonGeometry
    pairwise: dict
    target: dict
    scene: dict
    signals: dict  # name -> (T,) scene-level series for baselines and alert reasons


# --------------------------------------------------------------------- helpers


def _joint(kp: np.ndarray, sc: np.ndarray, j: int, thr: float) -> np.ndarray:
    p = kp[:, :, j, :].astype(np.float64)
    p[sc[:, :, j] < thr] = np.nan
    return p


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint that falls back to whichever side is visible."""
    return np.where(np.isnan(a), b, np.where(np.isnan(b), a, 0.5 * (a + b)))


def causal_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Trailing mean over the last k frames along axis 0, ignoring NaN."""
    k = max(int(k), 1)
    valid = ~np.isnan(x)
    c = np.cumsum(np.where(valid, x, 0.0), axis=0)
    n = np.cumsum(valid, axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c])
    n = np.concatenate([np.zeros_like(n[:1]), n])
    hi = np.arange(1, x.shape[0] + 1)
    lo = np.maximum(hi - k, 0)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / cnt
    return np.where(cnt > 0, out, np.nan)


def _rate(x: np.ndarray, fps: float) -> np.ndarray:
    """Per-second first difference along axis 0 (NaN in the first frame)."""
    d = np.full_like(x, np.nan, dtype=np.float64)
    d[1:] = (x[1:] - x[:-1]) * fps
    return d


# -------------------------------------------------------------- per person


def person_geometry(kp: np.ndarray, sc: np.ndarray, cfg: FeatureConfig) -> PersonGeometry:
    """Ground point, height, heading, velocity and gesture energy per person."""
    thr, fps = cfg.conf_thr, cfg.fps
    k = max(1, round(cfg.smooth_s * fps))
    lsh, rsh = _joint(kp, sc, L_SH, thr), _joint(kp, sc, R_SH, thr)
    sh = _mid(lsh, rsh)
    hip = _mid(_joint(kp, sc, L_HIP, thr), _joint(kp, sc, R_HIP, thr))
    ank = _mid(_joint(kp, sc, L_ANK, thr), _joint(kp, sc, R_ANK, thr))
    nose = _joint(kp, sc, NOSE, thr)
    wrists = np.stack([_joint(kp, sc, L_WR, thr), _joint(kp, sc, R_WR, thr)], axis=2)

    with np.errstate(invalid="ignore", divide="ignore"):
        up = sh - hip  # torso axis, pointing up the body
        torso = np.linalg.norm(up, axis=-1)
        # Height from whatever is visible. The factors keep the estimates roughly
        # consistent: nose-to-ankle ~ 1.15 x shoulder-to-ankle ~ 3 x torso.
        h = np.linalg.norm(nose - ank, axis=-1)
        h = np.where(np.isnan(h), 1.15 * np.linalg.norm(sh - ank, axis=-1), h)
        h = np.where(np.isnan(h), 3.0 * torso, h)
        height = causal_mean(h, round(cfg.height_smooth_s * fps))
        foot = np.where(np.isnan(ank), hip - 1.6 * up, ank)
        torso_c = _mid(sh, hip)
        valid = ~np.isnan(foot[..., 0]) & (np.nan_to_num(height) > 1.0)

        # Heading in the image plane from three cues:
        #   lateral - the nose sits ahead of the shoulder midpoint, perpendicular
        #             to the torso axis (strong in profile views);
        #   depth   - a person facing the camera shows their left shoulder on the
        #             image right (strong in front and back views); facing the
        #             camera maps to +y, down the image for an elevated camera;
        #   motion  - direction of travel, used only when body cues are weak,
        #             because a retreating target often walks backwards while
        #             still facing the person advancing on them.
        axis = up / (torso[..., None] + 1e-6)
        off = nose - sh
        off = off - np.sum(off * axis, axis=-1, keepdims=True) * axis
        lat_mag = np.linalg.norm(off, axis=-1)
        lat_dir = np.nan_to_num(off / (lat_mag[..., None] + 1e-6))
        lat_conf = np.nan_to_num(np.clip(lat_mag / (0.25 * torso + 1e-6), 0.0, 1.0))
        spread = (lsh[..., 0] - rsh[..., 0]) / (torso + 1e-6)
        dep_conf = np.nan_to_num(np.clip((np.abs(spread) - 0.3) / 0.4, 0.0, 1.0))
        dep_dir = np.stack([np.zeros_like(spread), np.sign(np.nan_to_num(spread))], axis=-1)

        vel = np.nan_to_num(_rate(causal_mean(foot, k), fps) / height[..., None])
        speed = np.linalg.norm(vel, axis=-1)
        mot_dir = vel / (speed[..., None] + 1e-6)
        mot_conf = np.where(speed > cfg.min_move_speed, 0.5, 0.0) * (
            1.0 - np.maximum(lat_conf, dep_conf)
        )
        mix = (
            lat_conf[..., None] * lat_dir
            + dep_conf[..., None] * dep_dir
            + mot_conf[..., None] * mot_dir
        )
        mix_n = np.linalg.norm(mix, axis=-1)
        heading = mix / (mix_n[..., None] + 1e-6)
        heading_conf = np.clip(mix_n, 0.0, 1.0)

        # Gesture energy: wrist speed relative to the torso centre, so walking
        # doesn't count; high values while standing still suggest arguing.
        rel = causal_mean(wrists - torso_c[:, :, None, :], k)
        rel_speed = np.linalg.norm(_rate(rel, fps), axis=-1)  # (T, M, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gesture = np.nan_to_num(np.nanmean(rel_speed, axis=-1) / height)
        raise_line = np.nan_to_num(sh[..., 1] - 0.05 * height, nan=-np.inf)
        wrist_y = np.nan_to_num(wrists[..., 1], nan=np.inf)
        arm_raise = (wrist_y < raise_line[..., None]).any(-1).astype(float)

    v = valid.astype(float)
    return PersonGeometry(
        valid=valid,
        foot=foot,
        torso_c=torso_c,
        height=np.where(valid, height, np.nan),
        heading=heading * v[..., None],
        heading_conf=heading_conf * v,
        vel=vel * v[..., None],
        speed=speed * v,
        gesture=gesture * v,
        arm_raise=arm_raise * v,
        wrists=wrists,
    )


# ------------------------------------------------------------------ per pair


def pairwise_features(g: PersonGeometry, cfg: FeatureConfig) -> dict:
    """(T, M, M) arrays; entry [t, i, j] describes person i relative to person j."""
    M = g.valid.shape[1]
    k = max(1, round(cfg.smooth_s * cfg.fps))
    both = g.valid[:, :, None] & g.valid[:, None, :] & ~np.eye(M, dtype=bool)[None]
    vec = g.foot[:, None, :, :] - g.foot[:, :, None, :]  # [t, i, j] = foot_j - foot_i
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.linalg.norm(vec, axis=-1)
        pair_h = 0.5 * (g.height[:, :, None] + g.height[:, None, :])
        dist = np.where(both, norm / pair_h, np.nan)
        u = np.nan_to_num(vec / (norm[..., None] + 1e-6))  # unit vector i -> j
        closing = -_rate(causal_mean(dist, k), cfg.fps)  # > 0: the pair is getting closer
        advance = np.einsum("tic,tijc->tij", g.vel, u)  # i moving toward j
        recoil = np.einsum("tjc,tijc->tij", g.vel, u)  # j moving away from i
        facing = np.einsum("tic,tijc->tij", g.heading, u) * g.heading_conf[:, :, None]
        mutual = np.minimum(facing, facing.transpose(0, 2, 1))  # "squaring up"
        wd = np.linalg.norm(g.wrists[:, :, :, None, :] - g.torso_c[:, None, None, :, :], axis=-1)
        reach = np.fmin(wd[:, :, 0, :], wd[:, :, 1, :]) / g.height[:, None, :]
    contact = both & (np.nan_to_num(reach, nan=np.inf) < cfg.contact_thr)
    shove = np.where(contact, np.clip(recoil, 0.0, None), 0.0)

    def masked(x: np.ndarray) -> np.ndarray:
        return np.where(both, np.nan_to_num(x), 0.0)

    return {
        "both": both,
        "dist": dist,
        "closing": masked(closing),
        "advance": masked(advance),
        "recoil": masked(recoil),
        "facing": masked(facing),
        "mutual_facing": masked(mutual),
        "reach": np.where(both, reach, np.nan),
        "contact": contact,
        "shove": shove,
    }


# ---------------------------------------------------------------- per target


def target_features(g: PersonGeometry, pw: dict, cfg: FeatureConfig) -> dict:
    """(T, M) arrays describing the pressure on each person k as a potential target."""
    T, M = g.valid.shape
    near = np.nan_to_num(pw["dist"], nan=np.inf) < cfg.near_radius  # [t, k, j]
    toward_k = pw["facing"].transpose(0, 2, 1) > cfg.facing_thr  # [t, k, j]: j faces k
    n_near = near.sum(-1).astype(float)
    n_facing = (near & toward_k).sum(-1).astype(float)

    # Angular coverage of k's neighbours: 0 for one neighbour, 0.5 for two on
    # opposite sides, 0.75 for four evenly spaced. A conversation circle has an
    # empty centre, so each member sees its neighbours on one side only; an
    # encircled target sees them all around.
    vec = g.foot[:, None, :, :] - g.foot[:, :, None, :]  # [t, k, j] = foot_j - foot_k
    ang = np.arctan2(vec[..., 1], vec[..., 0])
    coverage = np.zeros((T, M))
    for t in range(T):
        for kk in np.flatnonzero(n_near[t] >= 2):
            a = np.sort(ang[t, kk][near[t, kk]])
            gaps = np.diff(np.concatenate([a, a[:1] + 2 * np.pi]))
            coverage[t, kk] = 1.0 - gaps.max() / (2 * np.pi)
    inward = n_facing / np.maximum(n_near, 1.0)
    enclosure = coverage * inward * np.clip(n_near / 3.0, 0.0, 1.0)

    # Retreat then stall ("cornering"): k backed away from someone advancing on
    # them, then stopped moving while that person kept closing (a wall, a corner,
    # or being held in place).
    pressing = near & (pw["advance"].transpose(0, 2, 1) > cfg.min_move_speed)  # j advances on k
    retreat = np.where(pressing, pw["recoil"].transpose(0, 2, 1), 0.0).max(-1)  # k moves away
    pressure = np.where(near, np.clip(pw["closing"], 0.0, None), 0.0).max(-1)
    w = max(3, round(cfg.corner_window_s * cfg.fps))
    lag = w // 3
    past = np.zeros_like(retreat)
    past[lag:] = causal_mean(retreat, w)[:-lag]
    stalled = np.clip(1.0 - causal_mean(g.speed, lag) / cfg.min_move_speed, 0.0, 1.0)
    cornered = np.clip(past / cfg.min_move_speed, 0.0, 1.0) * stalled * (pressure > 0.05)

    v = g.valid.astype(float)
    return {
        "n_near": n_near * v,
        "n_facing": n_facing * v,
        "coverage": coverage * v,
        "enclosure": enclosure * v,
        "cornered": cornered * v,
        "retreat": retreat * v,
    }


# -------------------------------------------------------------------- scene


def _groups(adj: np.ndarray, valid: np.ndarray) -> list[list[int]]:
    """Connected components of the 'near' graph among valid people."""
    idx = [int(i) for i in np.flatnonzero(valid)]
    parent = {i: i for i in idx}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in idx:
        for b in idx:
            if a < b and adj[a, b]:
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def scene_features(g: PersonGeometry, pw: dict, cfg: FeatureConfig) -> dict:
    """(T,) series: largest group, outsiders converging on it, still inward ring, kinetic energy."""
    T = g.valid.shape[0]
    near = np.nan_to_num(pw["dist"], nan=np.inf) < cfg.near_radius
    largest, converging, ring = np.zeros(T), np.zeros(T), np.zeros(T)
    for t in range(T):
        groups = _groups(near[t], g.valid[t])
        if not groups:
            continue
        big = max(groups, key=len)
        largest[t] = len(big)
        if len(big) < 2:
            continue
        centre = g.foot[t, big].mean(0)
        outsiders = [
            int(i)
            for i in np.flatnonzero(g.valid[t])
            if i not in big and g.speed[t, i] > cfg.min_move_speed
        ]
        if outsiders:
            to_c = centre - g.foot[t, outsiders]
            to_c /= np.linalg.norm(to_c, axis=-1, keepdims=True) + 1e-6
            dirs = g.vel[t, outsiders] / (g.speed[t, outsiders][:, None] + 1e-6)
            converging[t] = float((np.sum(to_c * dirs, -1) > 0.7).sum())
        to_c = centre - g.foot[t, big]
        to_c /= np.linalg.norm(to_c, axis=-1, keepdims=True) + 1e-6
        inward = np.sum(g.heading[t, big] * to_c, -1) * g.heading_conf[t, big] > cfg.facing_thr
        still = g.speed[t, big] < cfg.min_move_speed
        ring[t] = float((inward & still).mean())
    speed2 = np.where(g.valid, g.speed, np.nan) ** 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        kinetic = np.nan_to_num(np.nanmean(speed2, axis=1))
    return {
        "n_people": g.valid.sum(1).astype(float),
        "largest_group": largest,
        "converging": converging,
        "ring_still": ring,
        "kinetic": kinetic,
    }


# ------------------------------------------------------------------ public API


def compute(kp: np.ndarray, sc: np.ndarray, cfg: FeatureConfig | None = None) -> FeatureBundle:
    """All feature families plus scene-level summary signals for one clip."""
    cfg = cfg or FeatureConfig()
    g = person_geometry(kp, sc, cfg)
    pw = pairwise_features(g, cfg)
    tf = target_features(g, pw, cfg)
    sf = scene_features(g, pw, cfg)
    T = g.valid.shape[0]
    near = np.nan_to_num(pw["dist"], nan=np.inf) < cfg.near_radius
    close = near.any(-1)

    def pair_max(x: np.ndarray) -> np.ndarray:
        return np.where(near, x, 0.0).reshape(T, -1).max(-1)

    dist_inf = np.where(pw["both"], np.nan_to_num(pw["dist"], nan=np.inf), np.inf)
    signals = {
        "min_pair_dist": np.minimum(dist_inf.reshape(T, -1).min(-1), 3.0),
        "max_closing": pair_max(pw["closing"]),
        "max_mutual_facing": pair_max(pw["mutual_facing"]),
        "max_advance": pair_max(pw["advance"]),
        "gesture_close_still": np.where(close & (g.speed < cfg.min_move_speed), g.gesture, 0.0).max(
            -1
        ),
        "arm_raise_close": np.where(close, g.arm_raise, 0.0).max(-1),
        "contact": pair_max(pw["contact"].astype(float)),
        "shove": pair_max(pw["shove"]),
        "enclosure": tf["enclosure"].max(-1),
        "outnumbering": tf["n_facing"].max(-1),
        "cornered": tf["cornered"].max(-1),
        **sf,
    }
    return FeatureBundle(g, pw, tf, sf, signals)


def window_stats(
    signals: dict, fps: float, windows_s=(2.0, 5.0, 10.0)
) -> tuple[np.ndarray, list[str]]:
    """Trailing mean, max and slope of each scene signal -> ((T, F) matrix, names).

    This is the input to the Stage 0 baseline.
    """
    cols, names = [], []
    for name, x in signals.items():
        x = np.nan_to_num(np.asarray(x, dtype=np.float64))
        for w_s in windows_s:
            w = max(2, round(w_s * fps))
            padded = np.concatenate([np.full(w - 1, x[0]), x])
            win = np.lib.stride_tricks.sliding_window_view(padded, w)  # (T, w)
            cols += [win.mean(1), win.max(1), (win[:, -1] - win[:, 0]) / w_s]
            names += [f"{name}_mean{w_s:g}s", f"{name}_max{w_s:g}s", f"{name}_slope{w_s:g}s"]
    return np.stack(cols, axis=1), names


def observation_quality(g: PersonGeometry, min_height_px: float = 90.0) -> np.ndarray:
    """(T,) 0..1 confidence that the geometry is worth acting on.

    Tiny or barely-visible skeletons produce confident-looking nonsense, the same
    failure the detection side hit on corridor footage. The policy uses this to
    abstain: poor quality can still raise WATCH, never a staff-facing WARN.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        height = np.where(g.valid, g.height, np.nan)
        tallest = np.nan_to_num(np.nanmax(height, axis=1))
        conf = np.nan_to_num(np.nanmax(np.where(g.valid, g.heading_conf, np.nan), axis=1))
    size = np.clip(tallest / max(min_height_px, 1.0), 0.0, 1.0)
    seen = np.clip(g.valid.sum(1) / 2.0, 0.0, 1.0)  # at least two people to have an interaction
    return (size * seen * (0.5 + 0.5 * conf)).astype(np.float64)


def to_model_inputs(b: FeatureBundle) -> dict:
    """Arrays for EscalationNet: person (T, M, P), edge (T, M, M, E), scene (T, S), mask (T, M)."""
    g = b.geometry
    person = np.stack([getattr(g, k) for k in PERSON_KEYS] + [b.target[k] for k in TARGET_KEYS], -1)
    edges = []
    for k in EDGE_KEYS:
        x = b.pairwise[k].astype(np.float64)
        if k in ("dist", "reach"):
            x = np.clip(np.nan_to_num(x, nan=5.0), 0.0, 5.0)
        edges.append(x)
    return {
        "person": np.nan_to_num(person).astype(np.float32),
        "edge": np.nan_to_num(np.stack(edges, -1)).astype(np.float32),
        "scene": np.stack([b.scene[k] for k in SCENE_KEYS], -1).astype(np.float32),
        "mask": g.valid,
    }
