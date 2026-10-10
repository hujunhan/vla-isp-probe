"""Run OpenVLA-OFT's LIBERO eval on selected LIBERO-plus perturbation categories, one episode per task.

Wraps openvla-oft's `eval_libero` unchanged, and adds:
  * task selection by category / explicit ids (via task_classification.json)
  * instruction handling: LIBERO-plus derives `task.language` from the file name, so perturbed tasks get
    suffixes like "... view 0 0 100 0 0 initstate 0 noise 3" or "... light 1". LANG_MODE=clean strips the
    suffix (asserting the result is an original instruction of that suite); LANG_MODE=raw passes it through.
  * fail-loud accounting: "Episode error" lines are recorded as errors, not policy failures; a run
    aborts after 3 consecutive errors (e.g. CUDA OOM)
  * one CSV row per episode, with resume (tasks already in the CSV are skipped)

Env vars:
  CATEGORIES   comma-separated categories (default "Sensor Noise,Light Conditions")
  TASK_IDS     optional comma-separated classification ids (1-based) to restrict to
  LANG_MODE    clean (default) | raw
  OUT_CSV      per-episode CSV path (required)

Usage (from ~/research/openvla-oft, env `oft`, PYTHONPATH=~/research/LIBERO-plus, MUJOCO_GL=egl):
  OUT_CSV=... python run_oft_libero_plus.py --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
      --task_suite_name libero_spatial --num_trials_per_task 1 --load_in_4bit True

[ai-scaffold]
"""

import csv
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser(os.environ.get("OPENVLA_OFT_ROOT", "~/research/openvla-oft")))

from libero.libero import benchmark as bm  # noqa: E402
from libero_original_task_map import libero_task_map as ORIGINAL_TASKS  # noqa: E402

CLASSIFICATION = os.path.join(os.path.expanduser(os.environ.get("LIBERO_PLUS_ROOT", "~/research/LIBERO-plus")),
                              "libero/libero/benchmark/task_classification.json")
CATEGORIES = os.environ.get("CATEGORIES", "Sensor Noise,Light Conditions").split(",")
TASK_IDS = {int(i) for i in os.environ["TASK_IDS"].split(",")} if os.environ.get("TASK_IDS") else None
LANG = os.environ.get("LANG_MODE", "clean")
OUT_CSV = os.environ["OUT_CSV"]
SUFFIX = re.compile(r"_(view|light)_.*$")
FIELDS = ["suite", "task_id", "name", "category", "level", "noise_code", "lang_mode", "language",
          "success", "error", "policy_steps", "n_queries", "wall_s"]


def noise_code(name):
    m = re.search(r"_noise_(\d+)$", name)
    return int(m.group(1)) if m else ""


def clean_language(suite, name):
    base = SUFFIX.sub("", name)
    assert base in ORIGINAL_TASKS[suite], f"stripped name is not an original {suite} task: {name}"
    # Same filename->instruction rule LIBERO uses for the original task (strips "SCENEk_" prefixes in
    # libero_10). An earlier version used " ".join(base.split("_")), which fed Long tasks
    # "LIVING ROOM SCENE2 put ..." (bug found and fixed during E002 Stage 2).
    return bm.grab_language_from_filename(suite, base + ".bddl")


def done_keys():
    if not os.path.exists(OUT_CSV):
        return set()
    with open(OUT_CSV) as f:
        return {(r["suite"], r["name"]) for r in csv.DictReader(f)}


def make_suite(suite_name):
    base_cls = bm.BENCHMARK_MAPPING[suite_name]
    classes = json.load(open(CLASSIFICATION))[suite_name]
    done = done_keys()
    selected = [c for c in classes
                if c["category"] in CATEGORIES
                and (TASK_IDS is None or c["id"] in TASK_IDS)
                and (suite_name, c["name"]) not in done]

    class SelectedSuite(base_cls):
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            full = self.tasks  # identity order for task_order_index=0, so id-1 indexes it
            self.meta, self.tasks = [], []
            for c in selected:
                t = full[c["id"] - 1]
                assert t.name == c["name"], (t.name, c["name"])
                lang = clean_language(suite_name, t.name) if LANG == "clean" else t.language
                self.tasks.append(t._replace(language=lang))
                self.meta.append(c)
            self.n_tasks = len(self.tasks)
            self._full = full

        def get_task_init_states(self, i):
            # Parent looks tasks up by position; point it at the selected task's original position.
            full_tasks, self.tasks = self.tasks, self._full
            try:
                return super().get_task_init_states(self.meta[i]["id"] - 1)
            finally:
                self.tasks = full_tasks

    print(f"[E002] {suite_name}: {len(selected)} tasks selected ({len(done)} already in CSV), LANG={LANG}")
    return SelectedSuite


for _suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    bm.BENCHMARK_MAPPING[_suite] = make_suite(_suite)

from experiments.robot.libero import run_libero_eval as rle  # noqa: E402

state = {"errors_in_episode": 0, "consecutive_errors": 0, "queries": 0, "task": None, "suite": None}

_log = rle.log_message
_get_action = rle.get_action
_run_episode = rle.run_episode
_run_task = rle.run_task


def log_message(message, log_file=None):
    if str(message).startswith("Episode error"):
        state["errors_in_episode"] += 1
    _log(message, log_file)


def get_action(*args, **kwargs):
    state["queries"] += 1
    return _get_action(*args, **kwargs)


def run_task(cfg, task_suite, task_id, *args, **kwargs):
    state["suite"], state["task"] = task_suite, task_id
    return _run_task(cfg, task_suite, task_id, *args, **kwargs)


def run_episode(cfg, env, task_description, *args, **kwargs):
    state["errors_in_episode"], state["queries"] = 0, 0
    t0 = time.perf_counter()
    success, images = _run_episode(cfg, env, task_description, *args, **kwargs)
    suite, i = state["suite"], state["task"]
    c = suite.meta[i]
    error = state["errors_in_episode"] > 0
    row = dict(suite=cfg.task_suite_name, task_id=c["id"], name=c["name"], category=c["category"],
               level=c["difficulty_level"], noise_code=noise_code(c["name"]), lang_mode=LANG,
               language=task_description, success=int(bool(success) and not error), error=int(error),
               policy_steps=len(images), n_queries=state["queries"], wall_s=round(time.perf_counter() - t0, 2))
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    state["consecutive_errors"] = state["consecutive_errors"] + 1 if error else 0
    if state["consecutive_errors"] >= 3:
        raise SystemExit("[E002] 3 consecutive episode errors; aborting (see log)")
    return success, images


rle.log_message = log_message
rle.get_action = get_action
rle.run_episode = run_episode
rle.run_task = run_task

if __name__ == "__main__":
    rle.eval_libero()
