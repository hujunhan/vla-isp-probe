"""E007: image motion per control step in LIBERO OFT rollouts -> blur length at a given exposure.

Input: results/e007/motion/<suite>_<task>_<ep>.npz (per env step qpos/qvel/mocap, logged by
scripts/run_oft_camera.py with "motion_log") and <suite>_<task>.xml (model, once per task).
Frame index 0 = after set_init_state, 1..num_steps_wait = settle steps (dummy action),
then one frame per policy action.

For each consecutive pair (t, t+1) and each camera, at 256 px:
  * render depth + segmentation at t (robosuite's visibility: collision geoms and sites hidden);
  * back-project each pixel to a world point p with the camera pose at t;
  * rigid-body flow (primary): p moves with its geom's body, p' = X_b(t+1) X_b(t)^-1 p;
    static-scene flow (registered): p' = p;
  * project p' with the camera pose at t+1; |dpx| = image displacement over one control step.
  * agentview EEF: project the gripper0_grip_site position at t and t+1.
Blur at exposure t_e (linear motion within the step): blur_px = |dpx| * t_e * control_freq.

Notebook: experiments/E007-c1-step0-libero-motion.md. [ai-scaffold]; the flow/blur metric is core
metric code -> owner explain-back owed (decisions/002 safeguard 4).
"""
import argparse
import csv
import glob
import os

import mujoco
import numpy as np

SIZE = 256
CAMS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
EEF_SITE = "gripper0_grip_site"
FINGERS = ("gripper0_finger_joint1", "gripper0_finger_joint2")
MAX_DEPTH = 10.0   # m; anything farther is treated as background (none expected in LIBERO's room)
FIELDS = ["suite", "task", "ep", "success", "pair", "kind", "phase",
          "wrist_med", "wrist_p95", "wrist_static_med", "wrist_static_p95",
          "agent_eef", "agent_p99", "agent_static_p99"]


class Scene:
    def __init__(self, xml_path):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, SIZE, SIZE)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[0] = 0                  # robosuite hides collision geoms (group 0)
        self.opt.sitegroup[:] = 0                  # and sites (e.g. the 10 m grip_site_cylinder)
        self.cam_id = {k: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, n) for k, n in CAMS.items()}
        self.eef = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        self.fingers = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in FINGERS]
        c = np.arange(SIZE) + 0.5 - SIZE / 2.0
        self.cu, self.cv = np.meshgrid(c, c)       # pixel centres relative to the image centre (u right, v down)

    def set(self, ep, t):
        self.d.qpos[:] = ep["qpos"][t]
        self.d.qvel[:] = ep["qvel"][t]
        if self.m.nmocap:
            self.d.mocap_pos[:] = ep["mocap_pos"][t]
            self.d.mocap_quat[:] = ep["mocap_quat"][t]
        mujoco.mj_forward(self.m, self.d)

    def focal(self, cam):
        return (SIZE / 2.0) / np.tan(np.radians(self.m.cam_fovy[self.cam_id[cam]]) / 2.0)

    def pose(self, cam):
        i = self.cam_id[cam]
        return self.d.cam_xpos[i].copy(), self.d.cam_xmat[i].reshape(3, 3).copy()

    def render(self, cam):
        """-> depth (m, along the optical axis), geom id per pixel (-1 = none). Rows top to bottom."""
        self.r.update_scene(self.d, camera=CAMS[cam], scene_option=self.opt)
        self.r.enable_depth_rendering()
        depth = self.r.render().copy()
        self.r.disable_depth_rendering()
        self.r.enable_segmentation_rendering()
        seg = self.r.render().copy()
        self.r.disable_segmentation_rendering()
        gid = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        return depth, gid

    def project(self, cam, pos, R, pw):
        """World points (..., 3) -> pixel offsets from the image centre (u right, v down), and depth."""
        pc = (pw - pos) @ R                        # world -> camera frame (x right, y up, looking down -z)
        z = -pc[..., 2]
        f = self.focal(cam)
        return f * pc[..., 0] / z, -f * pc[..., 1] / z, z


def bodies(scene):
    return scene.d.xpos.copy(), scene.d.xmat.reshape(-1, 3, 3).copy()


def pair_flow(scene, ep, t):
    """Image motion (px per control step) between frames t and t+1 for both cameras."""
    scene.set(ep, t)
    out, pts = {}, {}
    xb0, Rb0 = bodies(scene)
    for cam in CAMS:
        depth, gid = scene.render(cam)
        pos, R = scene.pose(cam)
        f = scene.focal(cam)
        valid = (gid >= 0) & (depth > 0) & (depth < MAX_DEPTH)
        pc = np.stack([scene.cu * depth / f, -scene.cv * depth / f, -depth], -1)    # camera frame
        pw = pos + pc @ R.T                                                          # world
        b = scene.m.geom_bodyid[np.where(valid, gid, 0)]
        p_body = np.einsum("hwji,hwj->hwi", Rb0[b], pw - xb0[b])                     # body frame at t
        pts[cam] = (valid, pw, b, p_body)
        if cam == "agentview":
            u0, v0, _ = scene.project(cam, pos, R, scene.d.site_xpos[scene.eef])
            eef0 = np.array([u0, v0])
    width0 = ep["qpos"][t][scene.fingers[0]] - ep["qpos"][t][scene.fingers[1]]
    scene.set(ep, t + 1)
    xb1, Rb1 = bodies(scene)
    for cam in CAMS:
        valid, pw, b, p_body = pts[cam]
        pos, R = scene.pose(cam)
        u, v = scene.cu, scene.cv
        pw_rigid = xb1[b] + np.einsum("hwij,hwj->hwi", Rb1[b], p_body)
        res = {}
        for name, p1 in (("rigid", pw_rigid), ("static", pw)):
            u1, v1, z1 = scene.project(cam, pos, R, p1)
            ok = valid & (z1 > 1e-4)
            res[name] = np.hypot(u1 - u, v1 - v)[ok]
        out[cam] = res
    pos, R = scene.pose("agentview")
    u1, v1, _ = scene.project("agentview", pos, R, scene.d.site_xpos[scene.eef])
    out["eef"] = float(np.hypot(u1 - eef0[0], v1 - eef0[1]))
    return out, width0


def phases(widths, n_wait, n):
    """approach / grasp / transport from the finger opening (m): the grasp event is the first policy frame where
    the opening is > 1 cm below its running maximum (a grasped LIBERO object stops the fingers ~1.5 cm short of
    fully open); grasp = +-5 steps around it. All "approach" if the gripper never closes."""
    w = np.asarray(widths)
    closed = np.where(w[n_wait:] < np.maximum.accumulate(w[n_wait:]) - 0.01)[0]
    lab = np.array(["approach"] * n, dtype=object)
    if len(closed):
        c = n_wait + closed[0]
        lab[c + 5:] = "transport"
        lab[max(c - 5, 0):c + 5] = "grasp"
    return lab


def episode_rows(scene, path, suite, task, epi):
    ep = dict(np.load(path))
    n = len(ep["qpos"]) - 1
    n_wait = int(ep["num_steps_wait"])
    rows, widths = [], []
    for t in range(n):
        flow, w = pair_flow(scene, ep, t)
        widths.append(w)
        rows.append(dict(
            suite=suite, task=task, ep=epi, success=int(ep["success"]), pair=t,
            kind="settle" if t + 1 <= n_wait else "policy",
            wrist_med=np.median(flow["wrist"]["rigid"]), wrist_p95=np.percentile(flow["wrist"]["rigid"], 95),
            wrist_static_med=np.median(flow["wrist"]["static"]),
            wrist_static_p95=np.percentile(flow["wrist"]["static"], 95),
            agent_eef=flow["eef"], agent_p99=np.percentile(flow["agentview"]["rigid"], 99),
            agent_static_p99=np.percentile(flow["agentview"]["static"], 99)))
    # pair t is labelled with the phase of its start frame t
    lab = phases(widths, n_wait, n)
    for r, p in zip(rows, lab):
        r["phase"] = p if r["kind"] == "policy" else "settle"
    return rows


def photometric_check(xml, npz, pairs, cam="wrist"):
    """Sanity check of the flow conventions: warp frame t+1's RGB back onto frame t's pixels with the predicted
    flow (nearest neighbour) and compare to frame t. Returns mean |dRGB| for zero flow, static flow and rigid
    flow, per pair; a correct flow should give clearly the smallest error for rigid."""
    scene = Scene(xml)
    ep = dict(np.load(npz))
    out = []
    for t in pairs:
        imgs = []
        for k in (t, t + 1):
            scene.set(ep, k)
            scene.r.update_scene(scene.d, camera=CAMS[cam], scene_option=scene.opt)
            imgs.append(scene.r.render().astype(float))
        scene.set(ep, t)
        xb0, Rb0 = bodies(scene)
        depth, gid = scene.render(cam)
        pos, R = scene.pose(cam)
        f = scene.focal(cam)
        valid = (gid >= 0) & (depth < MAX_DEPTH)
        pw = pos + np.stack([scene.cu * depth / f, -scene.cv * depth / f, -depth], -1) @ R.T
        b = scene.m.geom_bodyid[np.where(valid, gid, 0)]
        p_body = np.einsum("hwji,hwj->hwi", Rb0[b], pw - xb0[b])
        scene.set(ep, t + 1)
        xb1, Rb1 = bodies(scene)
        pos1, R1 = scene.pose(cam)
        errs = {}
        for name, p1 in (("zero", None), ("static", pw),
                         ("rigid", xb1[b] + np.einsum("hwij,hwj->hwi", Rb1[b], p_body))):
            if p1 is None:
                u1, v1 = scene.cu, scene.cv
            else:
                u1, v1, _ = scene.project(cam, pos1, R1, p1)
            x = np.round(u1 + SIZE / 2 - 0.5).astype(int)
            y = np.round(v1 + SIZE / 2 - 0.5).astype(int)
            ok = valid & (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE)
            errs[name] = float(np.abs(imgs[1][y[ok], x[ok]] - imgs[0][ok]).mean())
        out.append((t, errs))
    scene.r.close()
    return out


def eef_overlay(xml, npz, out_png, t=None):
    """Sanity check: agentview RGB with the projected EEF site marked (red cross)."""
    import imageio.v2 as imageio
    scene = Scene(xml)
    ep = dict(np.load(npz))
    t = len(ep["qpos"]) // 2 if t is None else t
    scene.set(ep, t)
    scene.r.update_scene(scene.d, camera=CAMS["agentview"], scene_option=scene.opt)
    img = scene.r.render().copy()
    pos, R = scene.pose("agentview")
    u, v, _ = scene.project("agentview", pos, R, scene.d.site_xpos[scene.eef])
    x, y = int(round(u + SIZE / 2 - 0.5)), int(round(v + SIZE / 2 - 0.5))
    img[max(y - 6, 0):y + 7, max(x - 1, 0):x + 2] = [255, 0, 0]
    img[max(y - 1, 0):y + 2, max(x - 6, 0):x + 7] = [255, 0, 0]
    imageio.imwrite(out_png, img)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "..", "results", "e007"))
    a = ap.parse_args()
    mdir = os.path.join(a.dir, "motion")
    out_csv = os.path.join(a.dir, "frames.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for xml in sorted(glob.glob(os.path.join(mdir, "*.xml"))):
            stem = os.path.basename(xml)[:-4]
            suite, task = stem.rsplit("_", 1)
            scene = Scene(xml)
            for npz in sorted(glob.glob(os.path.join(mdir, f"{stem}_*.npz"))):
                epi = int(os.path.basename(npz)[:-4].rsplit("_", 1)[1])
                for r in episode_rows(scene, npz, suite, task, epi):
                    w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
                print(f"{stem} ep {epi}", flush=True)
            scene.r.close()
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
