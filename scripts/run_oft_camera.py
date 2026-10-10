"""E004 runner: OpenVLA-OFT on ORIGINAL LIBERO tasks with a simulated camera applied to observations.

Wraps openvla-oft's `eval_libero` unchanged. The env returned by `get_libero_env` is wrapped so that
every observation's selected camera images pass through `isp.camera` (or the image-space control)
before the policy sees them. Only images change: the instruction is asserted to equal the original
LIBERO instruction (E003 lesson).

Env vars:
  CONDITION     JSON, e.g. {"tag": "lam0.012", "mode": "raw", "cams": ["agentview", "wrist"],
                            "cfg": {"shot_noise": 0.012, "denoise_h": 18}}
                mode: "render" (no camera) | "raw" (isp.camera.simulate with cfg)
                      | "imgspace" (clean camera, then 8-bit Gaussian noise with "sigma")
                      | "infinite" (our sensor model + Infinite-ISP stages, E005): keys "lam",
                        "preset" (off|repo_default|tuned|strong), "bnr_scale", "wts"
  ORIG_TASK_IDS comma-separated original task ids within the suite (required)
  OUT_CSV       per-episode CSV (appended)
  SEED          base seed for camera noise (default 0)
  EPISODE_OFFSET  skip the first k init states, so episodes use init states k, k+1, ... and are recorded
                with episode index k+i (default 0; E006 fresh-episode confirmation)

[ai-scaffold]
"""

import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.expanduser(os.environ.get("OPENVLA_OFT_ROOT", "~/research/openvla-oft")))

import run_oft_original  # noqa: E402,F401  registers original-task suites (uses ORIG_TASK_IDS)
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from isp.camera import CameraConfig, image_space_gaussian, read_noise_for, simulate  # noqa: E402
from libero.libero import benchmark as bm  # noqa: E402
from libero_original_task_map import libero_task_map as ORIGINAL_TASKS  # noqa: E402

COND = json.loads(os.environ["CONDITION"])
OUT_CSV = os.environ["OUT_CSV"]
SEED = int(os.environ.get("SEED", "0"))
EPISODE_OFFSET = int(os.environ.get("EPISODE_OFFSET", "0"))
TASK_IDS = [int(i) for i in os.environ["ORIG_TASK_IDS"].split(",")]
CAM_KEYS = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"}
FIELDS = ["condition", "suite", "task_id", "episode", "success", "error", "policy_steps", "n_queries", "wall_s"]

MODE = COND["mode"]
CFG = CameraConfig(**COND.get("cfg", {}))
CLEAN = CameraConfig(input_domain=CFG.input_domain)
SAMPLE_DIR = os.path.join(os.path.dirname(OUT_CSV), "samples")
os.makedirs(SAMPLE_DIR, exist_ok=True)


if MODE == "infinite":
    from isp.infinite import preset as inf_preset, simulate_infinite  # noqa: E402
    INF_PRESET = inf_preset(COND["preset"], bnr_scale=COND.get("bnr_scale", 1.0), wts=COND.get("wts", 10.0))
    INF_LAM = float(COND.get("lam", 0.0))
    POOL = ThreadPoolExecutor(max_workers=2)


def transform(img, rng):
    if MODE == "render":
        return img
    if MODE == "infinite":
        return simulate_infinite(img, INF_LAM, read_noise_for(INF_LAM), INF_PRESET, rng)
    if MODE == "raw":
        return simulate(img, CFG, rng)
    if MODE == "imgspace":
        return image_space_gaussian(simulate(img, CLEAN, rng), COND["sigma"], rng)
    raise ValueError(MODE)


class CameraEnv:
    """Delegates to the LIBERO env; rewrites camera images in every returned observation."""

    def __init__(self, env):
        self._env = env
        self.rng = np.random.default_rng(SEED)
        self.cam_rngs = {c: np.random.default_rng([SEED, i]) for i, c in enumerate(CAM_KEYS)}
        self.first_obs = None

    def _obs(self, obs):
        obs = dict(obs)
        cams = COND.get("cams", [])
        if MODE == "infinite" and len(cams) > 1:
            # slow Python ISP: process cameras in parallel threads, one RNG stream per camera
            futs = {c: POOL.submit(transform, obs[CAM_KEYS[c]], self.cam_rngs[c]) for c in cams}
            for c, f in futs.items():
                obs[CAM_KEYS[c]] = f.result()
            return obs
        for cam in cams:
            obs[CAM_KEYS[cam]] = transform(obs[CAM_KEYS[cam]], self.cam_rngs.get(cam, self.rng) if MODE == "infinite" else self.rng)
        return obs

    def reset(self):
        return self._obs(self._env.reset())

    def set_init_state(self, state):
        return self._obs(self._env.set_init_state(state))

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._obs(obs)
        self.first_obs = self.first_obs or obs
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self._env, name)


from experiments.robot.libero import run_libero_eval as rle  # noqa: E402

_get_env = rle.get_libero_env
_load_init = rle.load_initial_states
_run_task = rle.run_task
_run_episode = rle.run_episode
_get_action = rle.get_action
_log = rle.log_message
state = {"suite": None, "task_idx": None, "episode": 0, "errors": 0, "consecutive": 0, "queries": 0}


def get_libero_env(task, *args, **kwargs):
    env, desc = _get_env(task, *args, **kwargs)
    return CameraEnv(env), desc


def load_initial_states(cfg, task_suite, task_id, log_file=None):
    initial_states, all_states = _load_init(cfg, task_suite, task_id, log_file)
    assert len(initial_states) >= EPISODE_OFFSET + cfg.num_trials_per_task, "not enough init states"
    return initial_states[EPISODE_OFFSET:], all_states


def run_task(cfg, task_suite, task_id, *args, **kwargs):
    state.update(suite=cfg.task_suite_name, task_idx=TASK_IDS[task_id], episode=EPISODE_OFFSET)
    return _run_task(cfg, task_suite, task_id, *args, **kwargs)


def log_message(message, log_file=None):
    if str(message).startswith("Episode error"):
        state["errors"] += 1
    _log(message, log_file)


def get_action(*args, **kwargs):
    state["queries"] += 1
    return _get_action(*args, **kwargs)


def run_episode(cfg, env, task_description, *args, **kwargs):
    suite, tid, ep = state["suite"], state["task_idx"], state["episode"]
    state["episode"] += 1
    expected = bm.grab_language_from_filename(suite, ORIGINAL_TASKS[suite][tid] + ".bddl")
    assert task_description == expected, f"instruction changed: {task_description!r} != {expected!r}"
    env.rng = np.random.default_rng([SEED, tid, ep])  # camera noise reproducible per episode
    env.cam_rngs = {c: np.random.default_rng([SEED, tid, ep, i]) for i, c in enumerate(CAM_KEYS)}  # infinite mode
    env.first_obs = None
    state["errors"], state["queries"] = 0, 0
    t0 = time.perf_counter()
    success, images = _run_episode(cfg, env, task_description, *args, **kwargs)
    error = state["errors"] > 0
    if ep == 0 and env.first_obs is not None:
        np.savez_compressed(os.path.join(SAMPLE_DIR, f"{COND['tag']}_{suite}_{tid}.npz"),
                            **{c: env.first_obs[CAM_KEYS[c]] for c in CAM_KEYS})
    row = dict(condition=COND["tag"], suite=suite, task_id=tid, episode=ep, success=int(bool(success) and not error),
               error=int(error), policy_steps=len(images), n_queries=state["queries"],
               wall_s=round(time.perf_counter() - t0, 2))
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    state["consecutive"] = state["consecutive"] + 1 if error else 0
    if state["consecutive"] >= 3:
        raise SystemExit("[E004] 3 consecutive episode errors; aborting")
    return success, images


rle.get_libero_env = get_libero_env
rle.load_initial_states = load_initial_states
rle.run_task = run_task
rle.log_message = log_message
rle.get_action = get_action
rle.run_episode = run_episode

if __name__ == "__main__":
    print(f"[E004] condition={COND} tasks={TASK_IDS}", flush=True)
    rle.eval_libero()
