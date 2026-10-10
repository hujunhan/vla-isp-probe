"""Run OpenVLA-OFT's LIBERO eval on the ORIGINAL (unperturbed) LIBERO tasks inside a LIBERO-plus install.

LIBERO-plus replaces each suite's 10 tasks with thousands of perturbed variants, so `libero_spatial`
etc. no longer contain the original tasks. This wrapper re-registers the four suites with their
original 10-task lists (vendored from upstream LIBERO) and then calls openvla-oft's `eval_libero`
unchanged. It also records per-query latency, per-episode wall time, and peak GPU memory.

Env vars:
  ORIG_TASK_IDS   comma-separated task ids to run (default: all 10)
  E001_METRICS    path of a JSON file to write timing/memory metrics to

Usage (from ~/research/openvla-oft, env `oft`, MUJOCO_GL=egl):
  python ~/research/camera-isp-probe/scripts/run_oft_original.py \
      --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
      --task_suite_name libero_spatial --num_trials_per_task 3 [--load_in_4bit True]

[ai-scaffold]
"""

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser(os.environ.get("OPENVLA_OFT_ROOT", "~/research/openvla-oft")))

from libero.libero import benchmark as bm  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero_original_task_map import libero_task_map as ORIGINAL_TASKS  # noqa: E402

TASK_IDS = os.environ.get("ORIG_TASK_IDS")
METRICS_PATH = os.environ.get("E001_METRICS")


def make_original_suite(suite_name):
    class OriginalSuite(bm.Benchmark):
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name = suite_name
            names = ORIGINAL_TASKS[suite_name]
            if TASK_IDS:
                names = [names[int(i)] for i in TASK_IDS.split(",")]
            self.tasks = [
                bm.Task(
                    name=n,
                    language=bm.grab_language_from_filename(suite_name, n + ".bddl"),
                    problem="Libero",
                    problem_folder=suite_name,
                    bddl_file=f"{n}.bddl",
                    init_states_file=f"{n}.pruned_init",
                )
                for n in names
            ]
            self.n_tasks = len(self.tasks)

        def get_task_init_states(self, i):
            path = os.path.join(
                get_libero_path("init_states"), self.tasks[i].problem_folder, self.tasks[i].init_states_file
            )
            return torch.load(path)

    return OriginalSuite


for _suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    bm.BENCHMARK_MAPPING[_suite] = make_original_suite(_suite)

from experiments.robot.libero import run_libero_eval as rle  # noqa: E402

metrics = {"query_s": [], "episode_s": [], "episode_success": []}

_get_action = rle.get_action
_run_episode = rle.run_episode


def timed_get_action(*args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = _get_action(*args, **kwargs)
    torch.cuda.synchronize()
    metrics["query_s"].append(time.perf_counter() - t0)
    return out


def timed_run_episode(*args, **kwargs):
    t0 = time.perf_counter()
    success, images = _run_episode(*args, **kwargs)
    metrics["episode_s"].append(time.perf_counter() - t0)
    metrics["episode_success"].append(bool(success))
    return success, images


rle.get_action = timed_get_action
rle.run_episode = timed_run_episode


def dump_metrics():
    metrics["peak_mem_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    metrics["peak_mem_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    metrics["argv"] = sys.argv[1:]
    if metrics["query_s"]:
        q = sorted(metrics["query_s"][1:] or metrics["query_s"])  # drop first (warm-up)
        metrics["query_s_median"] = q[len(q) // 2]
    print("E001_METRICS", json.dumps({k: v for k, v in metrics.items() if k != "query_s"}))
    if METRICS_PATH:
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    try:
        rle.eval_libero()
    finally:
        dump_metrics()
