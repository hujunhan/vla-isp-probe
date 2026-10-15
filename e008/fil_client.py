"""E008 client side (runs in the `oft` env): spawns render_server.py in the `mjhdr` env and renders the
current robosuite sim state with Filament. [ai-scaffold]"""
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MJHDR_PY = os.path.expanduser(os.environ.get("MJHDR_PY", "~/miniconda3/envs/mjhdr/bin/python"))
CAMS = ("agentview", "wrist")


class FilamentClient:
    def __init__(self, matched, log_path=os.devnull, ss=1):
        self.matched, self.ss = matched, ss
        self.log = open(log_path, "ab")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)          # keep the oft env's paths out of the mjhdr interpreter
        env.pop("MUJOCO_GL", None)
        self.p = subprocess.Popen([MJHDR_PY, os.path.join(HERE, "render_server.py")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, env=env)
        self.model_key = None

    def _call(self, header, payload=b""):
        header = dict(header, n=len(payload))
        self.p.stdin.write((json.dumps(header) + "\n").encode())
        self.p.stdin.write(payload)
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("render server died (see its log)")
        h = json.loads(line)
        data = self.p.stdout.read(h["n"])
        if not h.get("ok"):
            raise RuntimeError(f"render server error: {h.get('error')}")
        return h, data

    def ensure_model(self, sim):
        """(Re)load when robosuite rebuilt its model (hard reset creates a new MjModel)."""
        key = id(sim.model._model)
        if key != self.model_key:
            xml = sim.model.get_xml().encode()
            h, _ = self._call({"cmd": "load", "matched": self.matched, "ss": self.ss}, xml)
            assert h["nq"] == sim.model._model.nq, (h["nq"], sim.model._model.nq)
            self.model_key = key

    @staticmethod
    def state_vector(d):
        """d: raw mujoco MjData (sim.data._data). The vector the server's set_state expects."""
        return np.concatenate([d.qpos, d.mocap_pos.ravel(), d.mocap_quat.ravel()]).astype(np.float64)

    def render_each(self, sim, states):
        """E009: HDR render of each state vector in `states` (K of them). -> float16 array (K, 2, H, W, 3),
        cameras in CAMS order, obs orientation."""
        self.ensure_model(sim)
        v = np.stack(states).astype(np.float64)
        h, data = self._call({"cmd": "render_each", "k": len(states)}, v.tobytes())
        return np.frombuffer(data, dtype=h["dtype"]).reshape(h["shape"])

    def render(self, sim, mode):
        """-> dict cam -> image in obs orientation: float32 HDR (H, W, 3) or uint8 post (H, W, 3)."""
        self.ensure_model(sim)
        d = sim.data._data
        v = np.concatenate([d.qpos, d.mocap_pos.ravel(), d.mocap_quat.ravel()]).astype(np.float64)
        h, data = self._call({"cmd": "render", "mode": mode}, v.tobytes())
        imgs = np.frombuffer(data, dtype=h["dtype"]).reshape(h["shape"])
        if mode == "hdr":
            imgs = imgs.astype(np.float32)
        return {c: imgs[i] for i, c in enumerate(CAMS)}

    def close(self):
        try:
            self._call({"cmd": "quit"})
        except Exception:
            pass
        self.p.wait(timeout=10)
