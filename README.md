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

- **Python** 3.10+
- **CUDA** 12.x + NVIDIA GPU (RTX 2080 Ti or newer recommended)
- **[Meshroom](https://github.com/alicevision/Meshroom)** 2025+ (develop branch)
- [`tinycudann`](https://github.com/NVlabs/tiny-cuda-nn)
- [`nerfacc`](https://github.com/nerfstudio-project/nerfacc) 0.3.3
- [`torch_efficient_distloss`](https://github.com/sunset1995/torch_efficient_distloss)

Full dependency list: [`requirements.txt`](requirements.txt)

---

## Quick Start

> **Prerequisite:** a working [Meshroom](https://github.com/alicevision/Meshroom) installation.

### 1. Clone the plugin

```bash
cd /path/to/your/plugins
git clone https://github.com/meshroomHub/mrOpenRNb.git
```

### 2. Clone the Open-RNb core code

```bash
git clone https://github.com/RobinBruneau/Open-RNb.git
```

### 3. Set up the virtual environment

Meshroom looks for a folder named **`venv`** at the plugin root and uses its Python interpreter to run the node. You have two options:

#### Option A: Symlink an existing venv

If you already have a working virtual environment from the [Open-RNb](https://github.com/RobinBruneau/Open-RNb) repository (e.g. its `.venv`), you can simply symlink it:

```bash
cd mrOpenRNb
ln -s /absolute/path/to/Open-RNb/.venv venv
```

#### Option B: Create a fresh venv

```bash
cd mrOpenRNb

# Create the venv (must be named "venv", not ".venv")
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip
pip install setuptools==69.5.1 ninja

# Install PyTorch (CUDA 12.x)
pip install torch torchvision

# Install main dependencies
pip install -r requirements.txt

# Install tinycudann
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# If the pip install above fails (ModuleNotFoundError: pkg_resources), build from source:
cd /tmp
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git
cd tiny-cuda-nn/bindings/torch
# Set your GPU architecture: 70=V100, 75=T4, 80=A100, 86=RTX 3080/3090, 89=RTX 4090
TCNN_CUDA_ARCHITECTURES=86 python setup.py install
cd /path/to/your/plugins/mrOpenRNb

deactivate
```

### 4. Configure the plugin

Edit `meshroom/config.json` to point to your Open-RNb clone:

```json
[
    {
        "key": "OPEN_RNB_PATH",
        "type": "path",
        "value": "/absolute/path/to/Open-RNb"
    }
]
```

### 5. Register the plugin in Meshroom

Set the `MESHROOM_PLUGINS_PATH` environment variable:

```bash
# Linux
export MESHROOM_PLUGINS_PATH=/path/to/your/plugins/mrOpenRNb:$MESHROOM_PLUGINS_PATH

# Windows
set MESHROOM_PLUGINS_PATH=C:\path\to\mrOpenRNb;%MESHROOM_PLUGINS_PATH%
```

Launch Meshroom: the **OpenRNb** node appears under the **Neural Reconstruction** category.

### 6. Verify installation

```bash
source venv/bin/activate
python -c "
import tinycudann; print('tinycudann OK')
import nerfacc; print('nerfacc OK')
from torch_efficient_distloss import flatten_eff_distloss; print('distloss OK')
import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## Plugin Structure

```
mrOpenRNb/
├── meshroom/
│   ├── config.json                # Plugin configuration (OPEN_RNB_PATH)
│   └── OpenRNb/
│       ├── __init__.py
│       └── OpenRNb.py             # Meshroom node definition
├── venv/                          # Python virtual environment (or symlink, see step 3)
├── requirements.txt               # Python dependencies
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
| `scalingMode` | Scaling Mode | Scene normalization: `auto`, `pcd`, `silhouettes`, `cameras`, `none` |
| `sphereScale` | Sphere Scale | Bounding sphere scale after normalization (default: 1.0) |
| `warmupRatio` | Phase 1 Ratio | Fraction of steps for geometry-only phase (default: 0.1) |
| `useGpu` | Use GPU | Use GPU for training (default: true) |
| `openRnbPath` | Open-RNb Path | Path to Open-RNb code (set via `config.json`) |

### Outputs

| Parameter | Description |
|-----------|-------------|
| `outputFolder` | Folder containing all training artifacts |
| `outputMesh` | Reconstructed OBJ mesh in world coordinates |

---

## Acknowledgements

This work is supported by [**DOPAMIn**](https://www.cnrsinnovation.com/actualite/une-seconde-promotion-pour-le-programme-open-7-nouveaux-logiciels-scientifiques-a-valoriser/) (*Diffusion Open de Photogrammetrie par AliceVision/Meshroom pour l'Industrie*), selected in the 2024 cohort of the [**OPEN**](https://www.cnrsinnovation.com/open/) programme run by [CNRS Innovation](https://www.cnrsinnovation.com/). OPEN supports the valorization of open-source scientific software by providing dedicated developer resources, governance expertise, and industry partnership support.

**Lead researcher:** [Jean-Denis Durou](https://cv.hal.science/jean-denis-durou), [IRIT](https://www.irit.fr/) (INP-Toulouse)

---

## Related Projects

| Project | Description |
|---------|-------------|
| [Open-RNb](https://github.com/RobinBruneau/Open-RNb) | Open-source reimplementation of RNb-NeuS2, used by this plugin |
| [RNb-NeuS2](https://robinbruneau.github.io/publications/rnb_neus2.html) | Original RNb-NeuS2 method (IJCV 2025) |
| [mrSDMUniPS](https://github.com/meshroomHub/mrSDMUniPS) | Meshroom plugin for SDM-UniPS photometric stereo |

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
