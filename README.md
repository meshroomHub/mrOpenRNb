# mrOpenRNb

Meshroom plugin for [Open-RNb](https://github.com/RobinBruneau/Open-RNb): open-source neural surface reconstruction from
multi-view normal and reflectance maps, equivalent to RNb-NeuS2 (R. Bruneau et al.).

Open-RNb reconstructs a mesh from the normal and albedo maps of a photometric stereo method, such as
[SDM-UniPS](https://github.com/meshroomHub/mrSDMUniPS). It replaces the proprietary CUDA libraries of RNb-NeuS2 with standard PyTorch and
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

| Node | Step | Output |
|------|------|--------|
| `OpenRNb` | Mesh from normal maps, and optional albedo maps and masks | training folder, mesh |

---

## Requirements

- **[Meshroom](https://github.com/alicevision/Meshroom)** 2026.1 or later (develop branch until its release, tested on
  commit [`edc4b6a2`](https://github.com/alicevision/Meshroom/commit/edc4b6a2)): Meshroom 2025.1.x has neither the plugin
  manager nor the `[tool.meshroom]` settings of `pyproject.toml`
- **NVIDIA GPU** with a driver supporting CUDA 12.4 (RTX 2080 Ti or newer recommended)
- **Python** 3.10 to 3.12 (prebuilt `tinycudann` and `nerfacc` wheels)

All Python dependencies, Open-RNb included, are declared in [`pyproject.toml`](pyproject.toml):
PyTorch 2.5.1 (CUDA 12.4), `tinycudann` and `nerfacc` as prebuilt wheels from the
[meshroomHubWarehouse wheel index](https://meshroomhubwarehouse.github.io/wheels/cu124-pt25/),
and `open-rnb` pinned to a commit of [meshroomHubWarehouse/Open-RNb](https://github.com/meshroomHubWarehouse/Open-RNb),
a fork kept in sync with [RobinBruneau/Open-RNb](https://github.com/RobinBruneau/Open-RNb).

---

## Installation

### 1. Install the plugin

From Meshroom's plugin manager (*Manage Plugins*), or from the command line:

```bash
meshroom_plugins install mrOpenRNb
```

The installer creates the plugin `venv/` with `uv sync`: nothing needs to be compiled.
The `OpenRNb` node then appears in the *Neural Reconstruction* category.

### 2. Manual install (without the plugin manager)

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd /path/to/plugins
git clone https://github.com/meshroomHub/mrOpenRNb.git
cd mrOpenRNb
UV_PROJECT_ENVIRONMENT=venv uv sync   # Meshroom looks for venv/ at the plugin root
export MESHROOM_PLUGINS_PATH=/path/to/plugins/mrOpenRNb:$MESHROOM_PLUGINS_PATH
```

On Windows (PowerShell): `$env:UV_PROJECT_ENVIRONMENT = "venv"; uv sync`, then
`$env:MESHROOM_PLUGINS_PATH = "C:\path\to\plugins\mrOpenRNb"`.

Check the environment:

```bash
venv/bin/python -c "import tinycudann, nerfacc, torch, systems; print('CUDA:', torch.cuda.is_available())"
```

---

## Pipeline template

`meshroom/linoUniPsOpenRnb.mgt` (*File > New Pipeline > Lino Uni Ps Open Rnb*) runs the structure-from-motion of
AliceVision, segments the object with SAM 3, exports the undistorted images, estimates the normals with `LINOUniPS` and
reconstructs the mesh with `OpenRNb` (quick settings: 1000 steps, resolution 512).

It needs [mrLINOUniPS](https://github.com/meshroomHub/mrLINOUniPS), from Meshroom's plugin manager, and
[mrSegmentation](https://github.com/meshroomHub/mrSegmentation) (node `ImageSegmentationSam3`), not in the plugin
registry yet: install it by hand.

CameraInit groups the images of a folder into one pose only if the folder name starts with `PS_`. Elsewhere, each image
becomes its own pose and the photometric stereo nodes find no multi-lighting pose, so images must follow this layout:

```
dataset/
└── PS/
    ├── PS_01/
    │   ├── ambient_all_leds.jpg   # reference image, lit by all the lights
    │   ├── led_01.jpg             # single-light images
    │   ├── led_02.jpg
    │   └── ...
    ├── PS_02/
    └── ...
```

The filename of the reference image must contain `ambient` (CameraInit makes it the pose reference) and `all_leds`
(`SfMFilter_1` keeps these images for the structure from motion and the segmentation).

Drag and drop the parent folder `dataset/` into the Image Gallery of Meshroom, then set the SAM 3 prompt
(`ImageSegmentationSam3_1`, default `stone`) to your object.

---

## Nodes

### `OpenRNb`

#### Inputs

| Parameter | Label | Description |
|-----------|-------|-------------|
| `inputNormalSfm` | Normal Maps SfMData | SfMData file pointing to normal map images **(required)** |
| `inputAlbedoSfm` | Albedo Maps SfMData | SfMData file pointing to albedo images (enables two-phase training) |
| `inputMaskSfm` | Mask SfMData | SfMData file pointing to mask images |
| `inputMaskFolder` | Mask Folder | Folder with mask images containing viewId in filename (auto-generates SfMData) |
| `maxSteps` | Max Training Steps | Total training iterations (default: 20000) |
| `meshResolution` | Mesh Resolution | Marching cubes grid resolution (default: 512) |
| `scalingMode` | Scaling Mode | Scene normalization: `auto` (prefers silhouettes when masks available, then pcd, then cameras), `pcd`, `silhouettes`, `cameras`, `none` |
| `sphereScale` | Sphere Scale | Bounding sphere scale after normalization (default: 1.0) |
| `warmupRatio` | Phase 1 Ratio | Fraction of steps for geometry-only phase (default: 0.1) |
| `configFile` | Config File | Optional Open-RNb YAML config merged over the default `configs/sfm.yaml` (see [GPU memory](#gpu-memory)); the node inputs are applied on top of it |
| `useGpu` | Use GPU | Use GPU for training (default: true) |
| `openRnbPath` | Open-RNb Path | Fallback path to Open-RNb code, `${OPEN_RNB_PATH}` (see [Working on a local Open-RNb clone](#working-on-a-local-open-rnb-clone)) |
| `verboseLevel` | Verbose Level | Log verbosity |

#### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Folder containing all training artifacts |
| `outputMesh` | Reconstructed OBJ mesh in world coordinates |

---

## GPU memory

The training views are loaded on the GPU: memory grows with the number of pixels of the normal (and albedo) maps,
about 40 bytes per pixel in phase 1 and 52 in phase 2 (with albedos), plus about 4 GB for training and rendering.
For example, 13 views of 4144×2760 pixels peak at about 12 GB.

To reduce it, downscale the input maps, or set a smaller ray batch in a YAML file given to the
**Config File** input (`configFile`), e.g.:

```yaml
model:
  max_train_num_rays: 4096   # default 8192
  ray_chunk: 2048            # default 4096
```

---

## Development

### Working on a local Open-RNb clone

The node imports Open-RNb from the plugin `venv/`. To run a local clone instead, install it in editable mode in that venv:

```bash
uv pip install --python venv/bin/python -e /path/to/Open-RNb
```

Updating or reinstalling the plugin from the plugin manager recreates `venv/`: run this command again afterwards.

`OPEN_RNB_PATH` (`[tool.meshroom]` in `pyproject.toml`, default `venv`; an environment variable of the same name
overrides it) is only a fallback: the node adds it to the Python path when Open-RNb is not installed in the venv.

### Plugin structure

```
mrOpenRNb/
├── meshroom/
│   ├── linoUniPsOpenRnb.mgt       # Pipeline template
│   └── OpenRNb/
│       ├── __init__.py
│       └── OpenRNb.py             # Meshroom node definition
├── pyproject.toml                 # Dependencies and Meshroom settings (uv)
├── venv/                          # Created by the installer or by uv sync
└── README.md
```

For more details on how Meshroom plugins work, see:
- [Meshroom Plugin Install Guide](https://github.com/alicevision/Meshroom/blob/develop/INSTALL_PLUGINS.md)
- [mrHelloWorld](https://github.com/meshroomHub/mrHelloWorld): step-by-step tutorials for building Meshroom plugins

---

## Acknowledgements

This work is supported by [**DOPAMIn**](https://www.cnrsinnovation.com/actualite/une-seconde-promotion-pour-le-programme-open-7-nouveaux-logiciels-scientifiques-a-valoriser/) (*Diffusion Open de Photogrammetrie par AliceVision/Meshroom pour l'Industrie*), selected in the 2024 cohort of the [**OPEN**](https://www.cnrsinnovation.com/open/) programme run by [CNRS Innovation](https://www.cnrsinnovation.com/). OPEN supports the valorization of open-source scientific software by providing dedicated developer resources, governance expertise, and industry partnership support.

**Lead researcher:** [Jean-Denis Durou](https://cv.hal.science/jean-denis-durou), [IRIT](https://www.irit.fr/) (INP-Toulouse)
**Co-lead:** [Lilian Calvet](https://fr.linkedin.com/in/lilian-calvet-42b1a689), [Balgrist University Hospital](https://www.balgrist.ch/)

---

## Related Projects

| Project | Description |
|---------|-------------|
| [Open-RNb](https://github.com/RobinBruneau/Open-RNb) | Open-source reimplementation of RNb-NeuS2, used by this plugin |
| [RNb-NeuS2](https://github.com/meshroomHubWarehouse/RNb-NeuS2) | Original RNb-NeuS2 method (IJCV 2025) |
| [mrLINOUniPS](https://github.com/meshroomHub/mrLINOUniPS) | Meshroom plugin for LINO-UniPS photometric stereo |
| [mrSDMUniPS](https://github.com/meshroomHub/mrSDMUniPS) | Meshroom plugin for SDM-UniPS photometric stereo |
| [mrUniMSPS](https://github.com/meshroomHub/mrUniMSPS) | Meshroom plugin for Uni-MS-PS photometric stereo |
| [mrLedGetter](https://github.com/meshroomHub/mrLedGetter) | Meshroom plugin for LED-getter near-light photometric stereo |
| [mrRNbNeuS2](https://github.com/meshroomHub/mrRNbNeuS2) | Meshroom plugin for RNb-NeuS2 neural surface reconstruction |

---

## Citation

- **Multi-view Surface Reconstruction Using Normal and Reflectance Cues**  
  R. Bruneau, B. Brument, Y. Quéau, J. Mélou, F. B. Lauze, J.-D. Durou, L. Calvet (2025)  
  In *International Journal of Computer Vision (IJCV)*.  
  [arXiv](https://arxiv.org/abs/2506.04115)

- **RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction**  
  B. Brument, R. Bruneau, Y. Quéau, J. Mélou, F. Lauze, J.-D. Durou, L. Calvet (2024)  
  In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.  
  [Project page](https://robinbruneau.github.io/publications/rnb_neus.html)

---

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).
