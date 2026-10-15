# Building the HDR renderer (Tier 2, C1 only)

E008–E010 render LIBERO scenes with MuJoCo's Filament renderer and read back **linear, unclamped
RGBA16F** images. Upstream MuJoCo cannot do that read-back yet, so these runs need MuJoCo built from
[hujunhan/mujoco@8f82efa](https://github.com/hujunhan/mujoco/tree/filament-rgba16f) (branch
`filament-rgba16f`, one commit on top of upstream `d9741ee`). The build goes into its own conda env,
`mjhdr`. The policy env `oft` keeps mujoco 3.2.6, and `e008/fil_client.py` talks to a render server
running in `mjhdr` over a pipe.

These are the steps we used on Ubuntu 24.04 (GCC 13.3, CMake 3.28, RTX 3080 Laptop, driver 580). We
have not re-run them from a clean machine; please open an issue if a step fails.

## 1. Toolchain and env

```bash
sudo apt install build-essential cmake git libgl1-mesa-dev libvulkan-dev mesa-vulkan-drivers \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
conda create -n mjhdr python=3.12 -y
conda activate mjhdr
pip install numpy absl-py
```

The render server uses the Vulkan backend by default, so `vulkaninfo --summary` should list your
NVIDIA GPU. The apt list above is our best reconstruction; CMake names any missing development
package when it fails.

## 2. Source

```bash
git clone https://github.com/hujunhan/mujoco.git mujoco_hdr && cd mujoco_hdr
git checkout 8f82efa
```

CMake downloads Filament and the other dependencies on first configure, which needs network access
and a few GB of disk.

## 3. Native Studio build

The Python `mujoco.mjrf` module only builds against the Studio platform libraries, so build those
with Filament on:

```bash
cmake -S . -B build-studio -DCMAKE_BUILD_TYPE=Release \
  -DMUJOCO_USE_FILAMENT=ON -DMUJOCO_BUILD_STUDIO=ON \
  -DMUJOCO_BUILD_TESTS=OFF -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF \
  -DCMAKE_CXX_FLAGS="-Wno-error=pedantic -Wno-pedantic"
cmake --build build-studio -j"$(nproc)"
```

The `-Wno-pedantic` flags work around an upstream issue: with GCC, `-Werror=pedantic` fails on
Filament's anonymous structs, and the existing suppressions only apply to Clang.

## 4. Python wheel into `mjhdr`

Use the repository's own CI step. It gathers the Studio libraries into an install prefix, builds the
wheel and pip-installs it into the active env. Plain `cmake --install` misses the Studio platform
libraries.

```bash
conda activate mjhdr
MUJOCO_NATIVE_BUILD_DIR=$PWD/build-studio bash .github/workflows/build_steps.sh build_studio_python
```

## 5. Check

Run this from **outside** the source tree, so Python imports the installed wheel instead of
`python/mujoco`:

```bash
cd /tmp
python -c "import mujoco; from mujoco import mjrf; print(mujoco.__version__, mjrf.PixelFormat.PIXEL_FORMAT_RGBA16F)"
# 3.15.1 PixelFormat.PIXEL_FORMAT_RGBA16F
python <path-to>/mujoco_hdr/python/mujoco/rendering/filament/renderer_test.py
```

The GPU tests in `renderer_test.py` skip themselves when no Filament context can be created. If they
all skip, Vulkan is not finding the GPU.

## 6. Point this repo at it

```bash
export MJHDR_PY=~/miniconda3/envs/mjhdr/bin/python   # this is the default
```

- `e008/render_server.py` imports `mujoco.experimental.studio.window` only to register Filament's
  asset provider. Without that import, the first render fails with
  `Failed to open filament asset 'pbr.filamat'`.
- `E008_API=opengl` switches the server to the OpenGL backend. On hybrid Intel + NVIDIA laptops,
  Filament's OpenGL backend picks the Intel iGPU unless you also set
  `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`. Check the renderer string Filament
  logs at start-up. All of our runs used Vulkan on the NVIDIA GPU.
