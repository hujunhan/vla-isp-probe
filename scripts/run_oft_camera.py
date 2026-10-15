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
                        "preset" (off|repo_default|tuned|strong), "bnr_scale", "wts"; E009 Stage C adds
                        optional "read_var" (default: Brooks pairing with "lam") and "sharpen": false
                      | "filament" (E008): key "fil" = post | oetf | raw_isp (e008/conditions.py); images come
                        from a Filament render of the current sim state, not from the classic obs
                      | "c1" (E009): keys "t_ms" (None = instantaneous), "noise", "nr", optional "dump"
                        (dir under results/<exp>/ for NR-tuning frames); Stage B "k", "imgblur"; Stage C
                        "signed" (no black-level rectification before Bayer NR); E010 "probe" {"n", "t_ms"} (open-loop
                        action probe: N noisy variants per query, actions saved to <results>/probe/); e009/c1.py. Only OFT's query frames
                        (every 8th policy step) get the chain; the sub-frames are the physics substeps of the
                        control step that ends at the query observation
  ORIG_TASK_IDS comma-separated original task ids within the suite (required)
  OUT_CSV       per-episode CSV (appended)
  SEED          base seed for camera noise (default 0)
  MOTION_LOG    (E007) condition key "motion_log": true -> per env step, save qpos/qvel/mocap to
                <results>/motion/<suite>_<task>_<ep>.npz, plus the model XML once per task
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
MOTION_DIR = os.path.join(os.path.dirname(OUT_CSV), "motion") if COND.get("motion_log") else None
if MOTION_DIR:
    os.makedirs(MOTION_DIR, exist_ok=True)


if MODE == "filament":
    sys.path.insert(0, os.path.join(HERE, "..", "e008"))
    from conditions import FilamentConditions  # noqa: E402
    FIL = FilamentConditions(COND["fil"], log_path=os.path.join(os.path.dirname(OUT_CSV), "render_server.log"))
    POOL = ThreadPoolExecutor(max_workers=2)

if MODE == "c1":
    sys.path.insert(0, os.path.join(HERE, "..", "e009"))
    from c1 import C1Conditions  # noqa: E402
    from fil_client import FilamentClient as _FC  # noqa: E402
    C1 = C1Conditions(COND, log_path=os.path.join(os.path.dirname(OUT_CSV), "render_server.log"),
                      dump_dir=os.path.join(os.path.dirname(OUT_CSV), os.path.expanduser(COND["dump"])) if COND.get("dump") else None)
    POOL = ThreadPoolExecutor(max_workers=2)
    NUM_WAIT, CHUNK = 10, 8          # openvla-oft cfg.num_steps_wait and NUM_ACTIONS_CHUNK (checked per episode)

if MODE == "infinite":
    from isp.infinite import preset as inf_preset, simulate_infinite  # noqa: E402
    INF_PRESET = inf_preset(COND["preset"], bnr_scale=COND.get("bnr_scale", 1.0), wts=COND.get("wts", 10.0))
    INF_LAM = float(COND.get("lam", 0.0))
    INF_READ = float(COND["read_var"]) if "read_var" in COND else read_noise_for(INF_LAM)   # E009 Stage C: explicit
    if COND.get("sharpen") is False:
        INF_PRESET["sharpen"]["is_enable"] = False
    POOL = ThreadPoolExecutor(max_workers=2)


def transform(img, rng):
    if MODE == "render":
        return img
    if MODE == "infinite":
        return simulate_infinite(img, INF_LAM, INF_READ, INF_PRESET, rng)
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
        self.motion = []
        self.count, self.n_rendered, self.sub, self.armed = 0, 0, [], False

    def _hook(self):
        """E009: record the state after every physics substep while armed (robosuite calls sim.step()
        control_timestep / model_timestep times per env step)."""
        sim = self._env.sim
        if getattr(sim, "_c1_hooked", False):
            return
        orig = sim.step

        def step():
            orig()
            if self.armed:
                self.sub.append(_FC.state_vector(sim.data._data))
        sim.step = step
        sim._c1_hooked = True

    def _log_state(self):
        if MOTION_DIR:
            d = self._env.sim.data
            self.motion.append((np.array(d.qpos), np.array(d.qvel), np.array(d.mocap_pos), np.array(d.mocap_quat)))

    def _obs(self, obs, frame0=False):
        obs = dict(obs)
        cams = COND.get("cams", [])
        if MODE == "c1":          # reset() obs (discarded by OFT); query frames are handled in _step_c1
            return obs
        if MODE == "filament":
            classic = {c: obs[CAM_KEYS[c]] for c in CAM_KEYS}
            out = FIL.apply(self._env.sim, classic, frame0, pool=POOL)
            if out is not None:
                for cam in cams:
                    obs[CAM_KEYS[cam]] = out[cam]
            return obs
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
        obs = self._env.set_init_state(state)
        self.motion = []
        self._log_state()
        self.count = 0
        if MODE == "c1":
            C1.calibrate(self._env.sim, {c: obs[CAM_KEYS[c]] for c in CAM_KEYS})
            return obs
        return self._obs(obs, frame0=True)

    def step(self, action):
        if MODE == "c1":
            return self._step_c1(action)
        obs, reward, done, info = self._env.step(action)
        self._log_state()
        obs = self._obs(obs)
        self.first_obs = self.first_obs or obs
        return obs, reward, done, info

    def _step_c1(self, action):
        self._hook()
        self.count += 1
        query = self.count >= NUM_WAIT and (self.count - NUM_WAIT) % CHUNK == 0
        self.sub, self.armed = [], query
        obs, reward, done, info = self._env.step(action)
        self.armed = False
        if query:
            sim = self._env.sim
            nq = sim.model._model.nq
            assert np.array_equal(self.sub[-1][:nq], sim.data._data.qpos), "last substep != obs state"
            out = C1.query(sim, self.sub, self.cam_rngs, pool=POOL, tag=f"{state['suite']}_{state['task_idx']}_{state['episode'] - 1}")
            self.n_rendered += 1
            obs = dict(obs)
            for cam in COND.get("cams", []):
                obs[CAM_KEYS[cam]] = out[cam]
            if C1.probe:                     # E010: noisy variants of this query frame, for the action probe
                PROBE["obs"] = []
                for var in C1.probe_variants(self.probe_rng, COND.get("cams", []), pool=POOL):
                    o = dict(obs)
                    for cam, im in var.items():
                        o[CAM_KEYS[cam]] = im
                    PROBE["obs"].append(o)
                PROBE["step"] = self.count
                PROBE["clean_obs"] = obs
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
    if MODE == "filament":
        FIL.new_task()
    if MODE == "c1":
        C1.new_task()
    return _run_task(cfg, task_suite, task_id, *args, **kwargs)


def log_message(message, log_file=None):
    if str(message).startswith("Episode error"):
        state["errors"] += 1
    _log(message, log_file)


PROBE = {"obs": None, "log": []}
PROBE_DIR = os.path.join(os.path.dirname(OUT_CSV), "probe")


def get_action(*args, **kwargs):
    state["queries"] += 1
    actions = _get_action(*args, **kwargs)
    if PROBE["obs"] is not None:     # E010: same query, images replaced by the noisy variants
        cfg = args[0]
        resize = rle.get_image_resize_size(cfg)
        variants = []
        for o in PROBE["obs"]:
            ob, _ = rle.prepare_observation(o, resize)
            variants.append(np.asarray(_get_action(args[0], args[1], ob, *args[3:], **kwargs)))
        rec = dict(step=PROBE["step"], clean=np.asarray(actions), noisy=np.stack(variants))
        if not PROBE["log"]:         # determinism check: the clean query repeated once per episode, from a fresh
            ob, _ = rle.prepare_observation(PROBE["clean_obs"], resize)   # observation: get_vla_action normalises
            rec["clean_repeat"] = np.asarray(_get_action(args[0], args[1], ob, *args[3:], **kwargs))  # obs["state"] in place
        PROBE["log"].append(rec)
        PROBE["obs"] = None
    return actions


def run_episode(cfg, env, task_description, *args, **kwargs):
    suite, tid, ep = state["suite"], state["task_idx"], state["episode"]
    state["episode"] += 1
    expected = bm.grab_language_from_filename(suite, ORIGINAL_TASKS[suite][tid] + ".bddl")
    assert task_description == expected, f"instruction changed: {task_description!r} != {expected!r}"
    env.rng = np.random.default_rng([SEED, tid, ep])  # camera noise reproducible per episode
    env.cam_rngs = {c: np.random.default_rng([SEED, tid, ep, i]) for i, c in enumerate(CAM_KEYS)}  # infinite mode
    env.first_obs = None
    env.n_rendered = 0
    env.probe_rng = np.random.default_rng([SEED, tid, ep, 99])   # E010 probe noise
    PROBE["obs"], PROBE["log"] = None, []
    state["errors"], state["queries"] = 0, 0
    t0 = time.perf_counter()
    success, images = _run_episode(cfg, env, task_description, *args, **kwargs)
    error = state["errors"] > 0
    if MODE == "c1":
        assert cfg.num_steps_wait == NUM_WAIT, cfg.num_steps_wait
        assert state["queries"] <= env.n_rendered <= state["queries"] + 1, (state["queries"], env.n_rendered)
        if ep == EPISODE_OFFSET:
            with open(os.path.join(SAMPLE_DIR, f"{COND['tag']}_{suite}_{tid}_exposure.json"), "w") as f:
                json.dump(C1.stats, f)
    if MODE == "filament" and ep == EPISODE_OFFSET:
        with open(os.path.join(SAMPLE_DIR, f"{COND['tag']}_{suite}_{tid}_exposure.json"), "w") as f:
            json.dump(FIL.stats, f)
    if MOTION_DIR and env.motion:
        if ep == EPISODE_OFFSET:
            with open(os.path.join(MOTION_DIR, f"{suite}_{tid}.xml"), "w") as f:
                f.write(env.sim.model.get_xml())
        q, v, mp, mq = (np.stack(x) for x in zip(*env.motion))
        np.savez_compressed(os.path.join(MOTION_DIR, f"{suite}_{tid}_{ep}.npz"), qpos=q, qvel=v, mocap_pos=mp,
                            mocap_quat=mq, control_freq=env.env.control_freq, num_steps_wait=cfg.num_steps_wait,
                            success=int(bool(success)))
    if PROBE["log"]:
        os.makedirs(PROBE_DIR, exist_ok=True)
        L = PROBE["log"]
        np.savez_compressed(os.path.join(PROBE_DIR, f"{os.path.basename(OUT_CSV)[:-4]}_{suite}_{tid}_{ep}.npz"),
                            step=np.array([r["step"] for r in L]), clean=np.stack([r["clean"] for r in L]),
                            noisy=np.stack([r["noisy"] for r in L]), clean_repeat=L[0]["clean_repeat"],
                            success=int(bool(success)))
    if ep == EPISODE_OFFSET and env.first_obs is not None:
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
