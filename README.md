<div align="center">

# mrOpenRNb

### Meshroom Plugin for Open-RNb

<p>
Integrate <a href="https://github.com/RobinBruneau/Open-RNb">Open-RNb</a> neural surface reconstruction directly into your <a href="https://github.com/alicevision/Meshroom">Meshroom</a> photogrammetry pipeline.
</p>

<a href="https://github.com/RobinBruneau/Open-RNb"><img src="https://img.shields.io/badge/Core-Open--RNb-green" alt="Open-RNb" height="25"></a>

</div>

---

## What is Open-RNb?

**Open-RNb** is an open-source method for high-quality 3D surface reconstruction from multi-view normal and reflectance (albedo) maps, estimated by photometric stereo methods such as [SDM-UniPS](https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023/) and [Uni-MS-PS](https://github.com/Clement-Hardy/Uni-MS-PS). Built on [NeuS](https://lingjie0206.github.io/papers/NeuS/) neural implicit surfaces and [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl), it combines normal supervision with a two-phase albedo scaling pipeline to produce accurate geometry even when per-view reflectance maps have inconsistent scales.

### Brief history

- **RNb-NeuS** (CVPR 2024): the original method introducing reflectance and normal-based multi-view reconstruction.
  [Project page](https://robinbruneau.github.io/publications/rnb_neus.html) | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Brument_RNb-NeuS_Reflectance_and_Normal-based_Multi-View_3D_Reconstruction_CVPR_2024_paper.html)

- **RNb-NeuS2** (IJCV 2025): an extended journal version that represents reflectance and surface normals as radiance vectors under simulated illumination, enabling integration into both traditional multi-view stereo and neural volume rendering pipelines. Achieves state-of-the-art results on DiLiGenT-MV, LUCES-MV, and Skoltech3D benchmarks.
  [Project page](https://robinbruneau.github.io/publications/rnb_neus2.html) | [arXiv](https://arxiv.org/abs/2506.04115)

- **Open-RNb**: a fully open-source reimplementation equivalent to RNb-NeuS2, replacing all proprietary CUDA libraries with standard PyTorch + [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). This is the implementation used by this plugin.
  [GitHub](https://github.com/RobinBruneau/Open-RNb)

---

## Requirements

- **[Meshroom](https://github.com/alicevision/Meshroom)** 2026.1 or later (develop branch until its release):
  Meshroom 2025.1.x has neither the plugin manager nor the `[tool.meshroom]` settings of `pyproject.toml`
- **NVIDIA GPU** with a driver supporting CUDA 12.4 (RTX 2080 Ti or newer recommended)
- **Python** 3.10 to 3.12 (prebuilt `tinycudann` and `nerfacc` wheels)

All Python dependencies, Open-RNb included, are declared in [`pyproject.toml`](pyproject.toml):
PyTorch 2.5.1 (CUDA 12.4), `tinycudann` and `nerfacc` as prebuilt wheels from the
[meshroomHubWarehouse wheel index](https://meshroomhubwarehouse.github.io/wheels/cu124-pt25/),
and `open-rnb` pinned to a commit of [meshroomHubWarehouse/Open-RNb](https://github.com/meshroomHubWarehouse/Open-RNb),
a fork kept in sync with [RobinBruneau/Open-RNb](https://github.com/RobinBruneau/Open-RNb).

---

## Installation

### With the plugin manager

From Meshroom's plugin manager (*Manage Plugins*), or from the command line:

```bash
meshroom_plugins install mrOpenRNb
```

The installer creates the plugin `venv/` with `uv sync`: nothing needs to be compiled.
The **OpenRNb** node then appears under the **Neural Reconstruction** category.

### Manual install (without the plugin manager)

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd /path/to/plugins
git clone https://github.com/meshroomHub/mrOpenRNb.git
cd mrOpenRNb
UV_PROJECT_ENVIRONMENT=venv uv sync   # Meshroom looks for venv/ at the plugin root
export MESHROOM_PLUGINS_PATH=/path/to/plugins/mrOpenRNb
```

On Windows (PowerShell): `$env:UV_PROJECT_ENVIRONMENT = "venv"; uv sync`, then
`$env:MESHROOM_PLUGINS_PATH = "C:\path\to\plugins\mrOpenRNb"`.

Check the environment:

```bash
venv/bin/python -c "import tinycudann, nerfacc, torch, systems; print('CUDA:', torch.cuda.is_available())"
```

### Working on a local Open-RNb clone

The node imports Open-RNb from the plugin `venv/`. To run a local clone instead, install it
in editable mode in that venv:

```bash
uv pip install --python venv/bin/python -e /path/to/Open-RNb
```

Updating or reinstalling the plugin from the plugin manager recreates `venv/`: run this command again afterwards.
`OPEN_RNB_PATH` (`[tool.meshroom]` in `pyproject.toml`, default `venv`; an environment variable of the same name
overrides it) is only a fallback: the node adds it to the Python path when Open-RNb is not installed in the venv.

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

## Plugin Structure

```
mrOpenRNb/
├── meshroom/
│   └── OpenRNb/
│       ├── __init__.py
│       └── OpenRNb.py             # Meshroom node definition
├── pyproject.toml                 # Dependencies and Meshroom settings (uv)
├── requirements.txt               # Python dependencies (pip)
├── venv/                          # Created by the installer or by uv sync
└── README.md
```

For more details on how Meshroom plugins work, see:
- [Meshroom Plugin Install Guide](https://github.com/alicevision/Meshroom/blob/develop/INSTALL_PLUGINS.md)
- [mrHelloWorld](https://github.com/meshroomHub/mrHelloWorld): step-by-step tutorials for building Meshroom plugins

---

## Node Parameters

### Inputs

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

### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Folder containing all training artifacts |
| `outputMesh` | Reconstructed OBJ mesh in world coordinates |

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
| [RNb-NeuS2](https://robinbruneau.github.io/publications/rnb_neus2.html) | Original RNb-NeuS2 method (IJCV 2025) |
| [mrSDMUniPS](https://github.com/meshroomHub/mrSDMUniPS) | Meshroom plugin for SDM-UniPS photometric stereo |
| [mrLINOUniPS](https://github.com/meshroomHub/mrLINOUniPS) | Meshroom plugin for LINO_UniPS photometric stereo |
| [mrUniMSPS](https://github.com/meshroomHub/mrUniMSPS) | Meshroom plugin for Uni-MS-PS photometric stereo |

---

## Citation

If you use this work, please cite:

```bibtex
@article{bruneau25,
    title={{Multi-view Surface Reconstruction Using Normal and Reflectance Cues}},
    author={Robin Bruneau and Baptiste Brument and Yvain Qu{\'e}au and Jean M{\'e}lou
            and Fran{\c{c}}ois Bernard Lauze and Jean-Denis Durou and Lilian Calvet},
    journal={International Journal of Computer Vision (IJCV)},
    year={2025},
    eprint={2506.04115},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2506.04115},
}
```

```bibtex
@inproceedings{brument24,
    title={{RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction}},
    author={Baptiste Brument and Robin Bruneau and Yvain Qu{\'e}au and Jean M{\'e}lou
            and Fran{\c{c}}ois Lauze and Jean-Denis Durou and Lilian Calvet},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition},
    year={2024}
}
```

---

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).
