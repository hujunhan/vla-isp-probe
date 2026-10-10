"""Dump rendered frames (both cameras) from original LIBERO tasks for ISP unit tests and PSNR
calibration. No policy; dummy actions. [ai-scaffold]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser(os.environ.get("OPENVLA_OFT_ROOT", "~/research/openvla-oft")))
import run_oft_original  # noqa: F401  registers original-task suites
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

frames = {"agentview": [], "wrist": [], "meta": []}
for suite in ["libero_spatial", "libero_object", "libero_goal"]:
    s = benchmark.get_benchmark_dict()[suite]()
    for task_id in range(3):
        env, desc = get_libero_env(s.get_task(task_id), "openvla", resolution=256)
        states = s.get_task_init_states(task_id)
        for ep in range(2):
            env.reset(); obs = env.set_init_state(states[ep])
            for _ in range(10):
                obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
            frames["agentview"].append(obs["agentview_image"]); frames["wrist"].append(obs["robot0_eye_in_hand_image"])
            frames["meta"].append(f"{suite}:{task_id}:{ep}")
        env.close()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "e004", "frames.npz")
np.savez_compressed(out, agentview=np.stack(frames["agentview"]), wrist=np.stack(frames["wrist"]), meta=np.array(frames["meta"]))
print("saved", out, np.stack(frames["agentview"]).shape)
