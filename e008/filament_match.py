"""E008 L2 (decisions: E008 record, G4): renderer matching for Filament vs LIBERO's classic render.

Replaces mjrf's "classic scene" fallback lighting (brown ibl.ktx environment, model_lights.cc:186-212)
with explicit, neutral lights whose intensities are fitted to classic frame-0 images (image difference
only, never policy success), then frozen:
  * ambient:   an image light with a flat grey cube texture (neutral colour)          -> intensity amb
  * scene:     LIBERO's own two lights, both set to the same intensity                  -> intensity scene
  * headlight: a directional light re-aimed along the rendering camera before every render -> intensity head
               (the C RenderRequest has enable_headlight, scene_view.cc:293-303, but the Python
               RenderRequest does not expose it, so the fallback headlight never turns on from Python)
  * textures:  file textures decoded as sRGB (exported flag was linear)
  * materials: mjrf's phong->PBR multipliers via MJCF custom numerics (model_objects.cc:431-436):
               spec (default 0.2), shin (default 0.1)
[ai-scaffold]
"""
import re

import mujoco

HEADLIGHT = "e008_headlight"
DEFAULT = dict(amb=5000.0, scene=40000.0, head=40000.0, spec=0.2, shin=0.1, tex_srgb=True)
# FROZEN 2026-10-11 (E008 L2 fit, fit_lighting.py + session grids; 6 frame-0 images; image error only).
# cutoff/softness apply to LIBERO's light1/light2 (light2's cone misses the scene; no effect at any
# intensity). Mean |8-bit diff| to classic after D3 exposure matching + sRGB OETF: 12.81 (fallback 26.0;
# Phase 2 clean RAW->Infinite-ISP chain on classic: 2.11).
FROZEN = dict(amb=0.0, scene=40000.0, head=1000.0, spec=0.0, shin=0.1, tex_srgb=True, cutoff=28.0, softness=0.6)


def prepare_xml(xml, p):
    """Return MJCF text with the matched lighting/material setup for parameters p (see DEFAULT)."""
    p = {**DEFAULT, **p}
    if p["tex_srgb"]:
        # the exported LIBERO textures are flagged linear, so Filament would use sRGB-encoded PNG bytes as
        # linear albedo (double gamma after the OETF); decode file textures as sRGB (they are photographs)
        xml = re.sub(r'<texture ([^>]*file="[^"]*"[^>]*?)(/?)>',
                     lambda m: "<texture " + re.sub(r'\s*colorspace="[^"]*"', "", m.group(1)) + ' colorspace="sRGB"' + m.group(2) + ">",
                     xml)
    asset = ('<texture name="e008_amb" type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" '
             'width="8" height="8"/>')
    xml = xml.replace("<asset>", "<asset>" + asset, 1)
    # LIBERO's lights carry no intensity; give both the fitted scene intensity
    extra = "".join(f' {k}="{p[k]}"' for k in ("cutoff", "softness") if k in p)
    xml = re.sub(r'<light name="(light[12])"', lambda m: f'<light name="{m.group(1)}" intensity="{p["scene"]}"{extra}', xml)
    lights = (f'<light name="e008_ambient" type="image" texture="e008_amb" intensity="{p["amb"]}"/>'
              f'<light name="{HEADLIGHT}" directional="true" castshadow="false" pos="0 0 2" dir="0 0 -1" '
              f'diffuse="1 1 1" intensity="{p["head"]}"/>')
    xml = xml.replace("<worldbody>", "<worldbody>" + lights, 1)
    custom = (f'<numeric name="filament.phong.specular_multiplier" data="{p["spec"]}"/>'
              f'<numeric name="filament.phong.shininess_multiplier" data="{p["shin"]}"/>')
    if "<custom>" in xml:
        xml = xml.replace("<custom>", "<custom>" + custom, 1)
    else:
        xml = xml.replace("</mujoco>", f"<custom>{custom}</custom></mujoco>")
    return xml


def aim_headlight(model, data, cam_name):
    """Point the headlight along the camera's viewing direction (MuJoCo cameras look along -z)."""
    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, HEADLIGHT)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if lid < 0:
        return
    data.light_xpos[lid] = data.cam_xpos[cid]
    data.light_xdir[lid] = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
