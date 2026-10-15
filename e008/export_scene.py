"""E008 pre-flight G1 (oft env, mujoco 3.2.6): export a LIBERO scene + one state + the classic render.

For task 0 of each suite: build the env exactly as openvla-oft does (256 px), set init state 0, take
the 10 no-op steps OFT takes before querying, then save
  <out>/<suite>.xml        compiled model as MJCF (mj_saveLastXML via robosuite's MjModel.get_xml)
  <out>/<suite>_state.npz  qpos, qvel, mocap_pos/quat, classic agentview + wrist images (as in obs,
                           i.e. NOT flipped; OFT flips [::-1, ::-1] later)

Usage (env oft): MUJOCO_GL=egl python e008/export_scene.py --out ~/research/logs/e008/g1
[ai-scaffold]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, os.path.expanduser(os.environ.get("OPENVLA_OFT_ROOT", "~/research/openvla-oft")))

import run_oft_original as roo  # noqa: E402
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--init", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    roo.TASK_IDS = str(a.task)
    for suite in SUITES:
        bench = roo.make_original_suite(suite)()
        task = bench.get_task(0)
        env, desc = get_libero_env(task, "openvla", resolution=256)
        env.reset()
        obs = env.set_init_state(bench.get_task_init_states(0)[a.init])
        for _ in range(10):
            obs, *_ = env.step(get_libero_dummy_action("openvla"))
        xml = env.sim.model.get_xml()
        with open(os.path.join(a.out, f"{suite}.xml"), "w") as f:
            f.write(xml)
        d = env.sim.data
        np.savez_compressed(os.path.join(a.out, f"{suite}_state.npz"), qpos=d.qpos.copy(), qvel=d.qvel.copy(),
                            mocap_pos=d.mocap_pos.copy(), mocap_quat=d.mocap_quat.copy(),
                            agentview=obs["agentview_image"], wrist=obs["robot0_eye_in_hand_image"],
                            desc=desc)
        m = env.sim.model._model
        print(f"{suite}: '{desc}' nq={m.nq} nbody={m.nbody} ncam={m.ncam} nlight={m.nlight} "
              f"ntex={m.ntex} nmat={m.nmat} nmesh={m.nmesh} xml={len(xml)} chars", flush=True)
        env.close()


if __name__ == "__main__":
    main()
