"""E008 render server (runs in the `mjhdr` env: mujoco 3.15 + Filament RGBA16F). Talks to fil_client.py.

Protocol on stdin/stdout: each message is one JSON header line, then a binary payload of the length the
header declares. Real stdout is reserved for the protocol; fd 1 is redirected to stderr so Filament/C log
lines cannot corrupt it.
  {"cmd": "load", "n": <bytes>, "matched": bool}  + MJCF text   -> {"ok": true, "nq": .., "nmocap": ..}
      matched=True  : E008 L2 frozen lighting/materials (filament_match.FROZEN)
      matched=False : mjrf as shipped (fallback lighting); collision geoms and sites hidden in both
      "ss": k (optional) : render k x larger and average k x k blocks in linear (pixel-area integration)
  {"cmd": "render", "mode": "hdr"|"post", "n": <bytes>} + float64 [qpos, mocap_pos, mocap_quat]
      -> {"ok": true, "dtype": .., "shape": [2, H, W, 3], "n": <bytes>} + images (agentview, wrist), obs orientation
  {"cmd": "render_each", "k": K, "n": <bytes>} + K concatenated state vectors (E009 sub-frames)
      -> {"ok": true, "dtype": "float16", "shape": [K, 2, H, W, 3]} + HDR images, obs orientation
  {"cmd": "quit"}
[ai-scaffold]
"""
import json
import os
import sys

proto_out = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filament_match import FROZEN, prepare_xml  # noqa: E402
from filament_render import CAMS, FilamentCams  # noqa: E402

API = os.environ.get("E008_API", "vulkan")


def reply(header, payload=b""):
    header = dict(header, n=len(payload))
    proto_out.write((json.dumps(header) + "\n").encode())
    proto_out.write(payload)
    proto_out.flush()


def set_state(model, data, v):
    nq, nm = model.nq, model.nmocap
    data.qpos[:] = v[:nq]
    data.mocap_pos[:] = v[nq:nq + 3 * nm].reshape(nm, 3)
    data.mocap_quat[:] = v[nq + 3 * nm:nq + 7 * nm].reshape(nm, 4)
    mujoco.mj_forward(model, data)
    return data


def main():
    inp = sys.stdin.buffer
    model = data = fc = None
    while True:
        line = inp.readline()
        if not line:
            return
        h = json.loads(line)
        payload = inp.read(h.get("n", 0))
        try:
            if h["cmd"] == "quit":
                reply({"ok": True})
                return
            if h["cmd"] == "load":
                xml = payload.decode()
                if h.get("matched"):
                    xml = prepare_xml(xml, FROZEN)
                fc = None
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
                fc = FilamentCams(model, api=API, ss=int(h.get("ss", 1)))
                reply({"ok": True, "nq": model.nq, "nmocap": model.nmocap})
            elif h["cmd"] == "render_each":
                vs = np.frombuffer(payload, dtype=np.float64).reshape(int(h["k"]), -1)
                imgs = np.stack([np.stack([fc.render(set_state(model, data, v), c, "hdr") for c in CAMS])
                                 for v in vs]).astype(np.float16)
                reply({"ok": True, "dtype": str(imgs.dtype), "shape": list(imgs.shape)}, imgs.tobytes())
            elif h["cmd"] == "render":
                set_state(model, data, np.frombuffer(payload, dtype=np.float64))
                imgs = np.stack([fc.render(data, c, h["mode"]) for c in CAMS])
                if h["mode"] == "hdr":
                    imgs = imgs.astype(np.float16)   # lossless: the readback is half float
                reply({"ok": True, "dtype": str(imgs.dtype), "shape": list(imgs.shape)}, imgs.tobytes())
            else:
                reply({"ok": False, "error": f"unknown cmd {h['cmd']}"})
        except Exception as e:  # report, keep serving
            reply({"ok": False, "error": repr(e)})


if __name__ == "__main__":
    main()
