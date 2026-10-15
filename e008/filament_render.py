"""E008: Filament renderer for exported LIBERO scenes (runs in the `mjhdr` env, mujoco 3.15 + RGBA16F).

FilamentCams holds one Filament context for a model and renders named cameras from an MjData state:
  * mode "hdr":  RGBA16F, post-processing off -> float32 (H, W, 3) scene-linear x Filament exposure
  * mode "post": post-processing on (Filament tone mapper), read through the same RGBA16F target and
                 quantised to uint8 (H, W, 3); equals RGB8 readback within 1 LSB (HDR_VALIDATION.md 4.3).
                 One target only: Filament supports one read request per render.
Images are returned in the SAME orientation as robosuite's obs images (see `flip` below).

CLI (pre-flight G2/G3/G4): renders the exported scene states and saves npz next to them.
  python e008/filament_render.py --scenes ~/research/logs/e008/g1 [--api vulkan] [--light-intensity X]
[ai-scaffold]
"""
import argparse
import os
import time

import mujoco
import numpy as np
from mujoco import mjrf
from mujoco.experimental.studio import window  # noqa: F401  registers the "filament:" asset provider
from mujoco.rendering.filament import renderer

CAMS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}


class FilamentCams:
    def __init__(self, model, size=256, api="vulkan", flip=True, ss=1):
        # ss > 1: render at ss*size and average ss x ss blocks in linear HDR (pixel-area integration, as a
        # sensor pixel integrates irradiance; ISETCam spatialIntegration). Filament disables MSAA for
        # render-to-texture (scene_view.cc:287-288), while LIBERO's classic render uses offsamples = 4.
        self.model, self.out_size, self.flip, self.ss = model, size, flip, ss
        size = size * ss
        gapi = mjrf.GraphicsApi.GRAPHICS_API_VULKAN if api == "vulkan" else mjrf.GraphicsApi.GRAPHICS_API_OPENGL
        self.ctx = mjrf.Context(mjrf.ContextConfig(graphics_api=gapi))
        self.r = renderer.Renderer(self.ctx)
        self.objs = mjrf.ModelObjects(self.ctx, model)
        self.scene = self.r.scene("scn")
        self.scene.configure_from_model(model)
        self.lights = mjrf.ModelLights(self.scene, self.objs)
        self.rends = mjrf.ModelRenderables(self.scene, self.objs)
        # match robosuite's offscreen render (environments/base.py:300-301): collision geoms (group 0)
        # hidden, visual geoms (group 1) shown
        self.vopt = mujoco.MjvOption()
        self.vopt.geomgroup[0] = 0
        self.vopt.geomgroup[1] = 1
        # robosuite hides its sites with rgba alpha <= 0 (e.g. the 10 m grip_site_cylinder); the classic
        # renderer skips them but Filament draws them, so hide all site groups
        for k in range(len(self.vopt.sitegroup)):
            self.vopt.sitegroup[k] = 0
        self.rends.set_options(self.vopt)
        self.buf_hdr = np.zeros((size, size, 4), dtype=np.float16)
        self.r.target("hdr", self.buf_hdr, pixel_format=mjrf.PixelFormat.PIXEL_FORMAT_RGBA16F)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    def render(self, data, cam, mode="hdr"):
        from filament_match import aim_headlight  # no-op unless the model has the E008 headlight
        aim_headlight(self.model, data, CAMS[cam])
        self.lights.update(data)
        self.rends.update(data)
        self.cam.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMS[cam])
        post = mode == "post"
        self.r.view("v", scene="scn", target="hdr", camera=mujoco.mjv_camera2GLCamera(self.model, data, self.cam),
                    enable_post_processing=post)
        self.r.render()
        img = self.buf_hdr[..., :3].astype(np.float32)
        if self.ss > 1:
            n, k = self.out_size, self.ss
            img = img.reshape(n, k, n, k, 3).mean(axis=(1, 3))
        if post:
            img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        return img[::-1] if self.flip else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--api", default="vulkan")
    ap.add_argument("--light-intensity", type=float, default=None, help="override MJCF light intensity")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--tag", default="fil")
    a = ap.parse_args()
    for suite in ["libero_spatial", "libero_object", "libero_goal"]:
        m = mujoco.MjModel.from_xml_path(os.path.join(a.scenes, f"{suite}.xml"))
        if a.light_intensity is not None:
            m.light_intensity[:] = a.light_intensity
        d = mujoco.MjData(m)
        z = np.load(os.path.join(a.scenes, f"{suite}_state.npz"))
        d.qpos[:], d.qvel[:] = z["qpos"], z["qvel"]
        d.mocap_pos[:], d.mocap_quat[:] = z["mocap_pos"], z["mocap_quat"]
        mujoco.mj_forward(m, d)
        fc = FilamentCams(m, api=a.api, flip=not a.no_flip)
        out = {}
        for cam in CAMS:
            for mode in ["hdr", "post"]:
                out[f"{cam}_{mode}"] = fc.render(d, cam, mode)
        t0 = time.perf_counter()
        for _ in range(20):
            for cam in CAMS:
                fc.render(d, cam, "hdr")
        ms = (time.perf_counter() - t0) / 20 * 1e3
        np.savez_compressed(os.path.join(a.scenes, f"{suite}_{a.tag}.npz"), **out)
        h = out["agentview_hdr"]
        print(f"{suite}: hdr agentview min {h.min():.3g} p50 {np.median(h):.3g} p99 {np.percentile(h, 99):.3g} "
              f"max {h.max():.3g}; 2-cam hdr render {ms:.1f} ms/step", flush=True)


if __name__ == "__main__":
    main()
