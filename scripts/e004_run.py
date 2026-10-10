"""E004 driver: run blocks of camera conditions for one policy, one process per (condition, suite).

  python e004_run.py --policy oft --blocks B01 --tasks libero_spatial:0,libero_object:0,libero_goal:0

Resumable: a (condition, suite, task) already holding --episodes rows in the CSV is skipped.
Policy 'oft_w' only uses the third-person image, so its camera model is applied to agentview only
(the wrist image is never seen by the policy). [ai-scaffold]
"""
import argparse
import csv
import glob
import json
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OFT_ROOT = os.path.expanduser(os.environ.get("OPENVLA_OFT_ROOT", "~/research/openvla-oft"))
LIBERO_PLUS_ROOT = os.path.expanduser(os.environ.get("LIBERO_PLUS_ROOT", "~/research/LIBERO-plus"))
LOG_DIR = os.path.expanduser(os.environ.get("LOG_DIR", "~/research/logs"))
RUNNER = os.path.join(ROOT, "scripts", "run_oft_camera.py")
OFT_W = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Sylvest--openvla-7b-oft-finetuned-libero-without-wrist/snapshots/*"))


def local_snapshot(repo):
    """Local HF cache snapshot dir. Runs use local paths + HF_HUB_OFFLINE=1 because a transient Hub
    lookup failure once made the OFT loader treat a repo id as a missing local dir (E004 run log)."""
    paths = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))
    assert len(paths) == 1, (repo, paths)
    return paths[0]


def checkpoint_args(policy, suite):
    if policy == "oft":
        return ["--pretrained_checkpoint",
                local_snapshot(f"moojink/openvla-7b-oft-finetuned-{suite.replace('libero_', 'libero-')}")]
    if policy == "oft_w":
        return ["--pretrained_checkpoint", os.path.join(OFT_W[0], suite), "--num_images_in_input", "1"]
    raise ValueError(policy)


def counts(path):
    c = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                k = (r["condition"], r["suite"], r["task_id"])
                c[k] = c.get(k, 0) + 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--tasks", required=True, help="suite:task_id,...")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--exp", default="e004", help="results/<exp>/conditions.json and <policy>.csv")
    ap.add_argument("--episode-offset", type=int, default=0, help="start at init state k (fresh episodes)")
    a = ap.parse_args()
    plan = json.load(open(os.path.join(ROOT, "results", a.exp, "conditions.json")))
    out_csv = os.path.join(ROOT, "results", a.exp, f"{a.policy}.csv")
    env = dict(os.environ, PYTHONPATH=LIBERO_PLUS_ROOT, MUJOCO_GL="egl", OUT_CSV=out_csv,
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for block in a.blocks.split(","):
        for cond in plan["blocks"][block]:
            cond = dict(cond)
            if a.policy == "oft_w":
                cond["cams"] = [c for c in cond.get("cams", []) if c == "agentview"]
            for st in a.tasks.split(","):
                suite, tid = st.split(":")
                if counts(out_csv).get((cond["tag"], suite, tid), 0) >= a.episodes:
                    print(f"[skip] {a.policy} {cond['tag']} {suite}:{tid}", flush=True)
                    continue
                print(f"[run] {a.policy} {block} {cond['tag']} {suite}:{tid}", flush=True)
                e = dict(env, CONDITION=json.dumps(cond), ORIG_TASK_IDS=tid, EPISODE_OFFSET=str(a.episode_offset))
                cmd = ["python", RUNNER, *checkpoint_args(a.policy, suite), "--task_suite_name", suite,
                       "--num_trials_per_task", str(a.episodes), "--load_in_4bit", "True",
                       "--local_log_dir", os.path.join(LOG_DIR, f"{a.exp}-{a.policy}")]
                r = subprocess.run(cmd, env=e, cwd=OFT_ROOT)
                print(f"[exit {r.returncode}] {a.policy} {cond['tag']} {suite}:{tid}", flush=True)
                if counts(out_csv).get((cond["tag"], suite, tid), 0) < a.episodes:
                    raise SystemExit(f"run incomplete: {cond['tag']} {suite}:{tid} (exit {r.returncode})")
    print("E004_DRIVER_DONE", flush=True)


if __name__ == "__main__":
    main()
