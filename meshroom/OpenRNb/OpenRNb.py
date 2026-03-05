__version__ = "2.0"

import os
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class OpenRNb(desc.Node):
    """
    Neural surface reconstruction from normal maps using Open-RNb.

    Accepts SfMData files for normals, albedos, and masks.
    When albedos are provided, uses two-phase training with
    automatic albedo scaling via multi-view consistency.
    Exports mesh in world coordinates.
    """

    category = "Neural Reconstruction"
    gpu = desc.Level.INTENSIVE
    size = desc.DynamicNodeSize('inputNormalSfm')

    documentation = """
    Neural surface reconstruction from multi-view normal maps.
    Uses Open-RNb with NeuS neural implicit surfaces.

    **Inputs:**
    - Normal maps SfMData (required)
    - Albedo maps SfMData (optional — enables two-phase training)
    - Mask SfMData (optional — used for silhouette-based scaling)

    **Scene normalization:** Auto-detected from 3D landmarks or silhouettes.

    **Output:** OBJ mesh in world coordinates.
    """

    inputs = [
        desc.File(
            name="inputNormalSfm",
            label="Normal Maps SfMData",
            description="SfMData file pointing to normal map images. Required.",
            value="",
        ),
        desc.File(
            name="inputAlbedoSfm",
            label="Albedo Maps SfMData",
            description="SfMData file pointing to albedo images. "
                        "If provided, enables two-phase training with albedo scaling.",
            value="",
        ),
        desc.File(
            name="inputMaskSfm",
            label="Mask SfMData",
            description="SfMData file pointing to mask images. "
                        "Used for silhouette-based scene normalization if no landmarks.",
            value="",
        ),
        desc.File(
            name="inputMaskFolder",
            label="Mask Folder",
            description="Folder containing mask images with viewId in filename "
                        "(e.g. '12345.png' or 'mask_12345.exr'). "
                        "If set, generates a mask SfMData from the normal SfMData. "
                        "Ignored when Mask SfMData is already provided.",
            value="",
        ),
        desc.IntParam(
            name="maxSteps",
            label="Max Training Steps",
            description="Total training iterations (phase 2 uses full count, phase 1 uses warmupRatio fraction).",
            value=20000,
            range=(1000, 100000, 1000),
        ),
        desc.IntParam(
            name="meshResolution",
            label="Mesh Resolution",
            description="Marching cubes resolution for final mesh extraction.",
            value=512,
            range=(256, 2048, 64),
        ),
        desc.ChoiceParam(
            name="scalingMode",
            label="Scaling Mode",
            description="Scene normalization: auto detects landmarks or falls back to silhouettes.",
            values=["auto", "pcd", "silhouettes", "cameras", "none"],
            value="auto",
            exclusive=True,
        ),
        desc.FloatParam(
            name="sphereScale",
            label="Sphere Scale",
            description="Target scale within unit sphere after normalization.",
            value=1.0,
            range=(0.1, 2.0, 0.1),
        ),
        desc.FloatParam(
            name="warmupRatio",
            label="Phase 1 Ratio",
            description="Fraction of maxSteps for phase 1 (geometry only). "
                        "Only used when albedos are provided.",
            value=0.1,
            range=(0.01, 0.5, 0.01),
        ),
        desc.BoolParam(
            name="useGpu",
            label="Use GPU",
            description="Use GPU for training. Falls back to CPU if CUDA unavailable.",
            value=True,
            invalidate=False,
        ),
        desc.File(
            name="openRnbPath",
            label="Open-RNb Path",
            description="Path to Open-RNb code directory. "
                        "Set via config.json key OPEN_RNB_PATH.",
            value="${OPEN_RNB_PATH}",
            advanced=True,
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level for logging.",
            values=VERBOSE_LEVEL,
            value="info",
            exclusive=True,
        ),
    ]

    outputs = [
        desc.File(
            name="outputFolder",
            label="Output Folder",
            description="Output folder containing all training artifacts.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="outputMesh",
            label="Output Mesh",
            description="Reconstructed mesh in world coordinates.",
            value="{nodeCacheFolder}/mesh.obj",
            semantic="mesh",
            group="",
        ),
    ]

    def processChunk(self, chunk):
        initialized = False
        dm = None
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            # --- Input validation ---
            normal_sfm = chunk.node.inputNormalSfm.value
            if not normal_sfm:
                raise RuntimeError("inputNormalSfm is required but empty.")
            if not os.path.exists(normal_sfm):
                raise RuntimeError(
                    "Normal SfM file not found: {}".format(normal_sfm))

            albedo_sfm_path = chunk.node.inputAlbedoSfm.value or ""
            if albedo_sfm_path and not os.path.exists(albedo_sfm_path):
                raise RuntimeError(
                    "Albedo SfM file not found: {}".format(albedo_sfm_path))

            mask_sfm = chunk.node.inputMaskSfm.value or ""
            if mask_sfm and not os.path.exists(mask_sfm):
                raise RuntimeError(
                    "Mask SfM file not found: {}".format(mask_sfm))

            # --- Generate mask SfM from folder if needed ---
            mask_folder = chunk.node.inputMaskFolder.value or ""
            if not mask_sfm and mask_folder:
                if not os.path.isdir(mask_folder):
                    raise RuntimeError(
                        "Mask folder not found: {}".format(mask_folder))
                mask_sfm = self._generate_mask_sfm(
                    chunk, normal_sfm, mask_folder)

            # --- Resolve RNb-NeuS code path from config.json ---
            rnbneus_path = chunk.node.openRnbPath.evalValue
            if not rnbneus_path or not os.path.isdir(rnbneus_path):
                raise RuntimeError(
                    "OPEN_RNB_PATH is empty or not a valid directory. "
                    "Set it in config.json. Got: '{}'".format(rnbneus_path))

            # --- Standard imports ---
            import sys
            import copy
            import time
            import numpy as np
            import torch
            import pytorch_lightning as pl
            from pytorch_lightning import Trainer
            from pytorch_lightning.callbacks import ModelCheckpoint
            from pytorch_lightning.utilities.rank_zero import rank_zero_info
            from omegaconf import OmegaConf

            # --- Add Open-RNb to sys.path temporarily for imports ---
            original_path = sys.path[:]
            sys.path.insert(0, rnbneus_path)
            try:
                from utils.misc import load_config
                import datasets
                import systems
                from datasets.utils import (
                    compute_scaling_from_mesh, neus_c2w_to_standard,
                    SPACE_NORMALIZED,
                )
                from models.geometry import MarchingCubeHelper
                from utils.albedo_scaling import (
                    compute_albedo_scale_ratios, scale_albedo_images,
                )
            except ImportError as e:
                raise RuntimeError(
                    "Failed to import Open-RNb modules from {}: {}".format(
                        rnbneus_path, e))
            finally:
                sys.path[:] = original_path

            initialized = True

            # --- Device selection ---
            use_gpu = chunk.node.useGpu.value
            if use_gpu and torch.cuda.is_available():
                accelerator = 'gpu'
                chunk.logger.info("Using GPU (CUDA) for training.")
            else:
                accelerator = 'cpu'
                if use_gpu:
                    chunk.logger.warning(
                        "CUDA not available. Falling back to CPU.")
                else:
                    chunk.logger.info("Using CPU for training.")

            # --- Build config ---
            node_cache = chunk.node.outputFolder.value
            os.makedirs(node_cache, exist_ok=True)

            max_steps = chunk.node.maxSteps.value
            mesh_resolution = chunk.node.meshResolution.value

            override = {
                'dataset': {
                    'normal_sfm': normal_sfm,
                    'albedo_sfm': albedo_sfm_path,
                    'mask_sfm': mask_sfm,
                    'scaling_mode': chunk.node.scalingMode.value,
                    'sphere_scale': chunk.node.sphereScale.value,
                },
                'model': {
                    'geometry': {
                        'isosurface': {
                            'resolution': mesh_resolution,
                        }
                    }
                },
                'system': {
                    'albedo_scaling': {
                        # Don't set 'enabled' — let auto-detection in sfm.yaml
                        # handle it (null = auto based on albedo presence).
                        'warmup_ratio': chunk.node.warmupRatio.value,
                    }
                },
                'trainer': {
                    'max_steps': max_steps,
                },
            }

            override_path = os.path.join(node_cache, 'config_override.yaml')
            OmegaConf.save(OmegaConf.create(override), override_path)

            base_yaml = os.path.join(rnbneus_path, 'configs', 'sfm.yaml')
            if not os.path.exists(base_yaml):
                raise RuntimeError(
                    "Base config not found: {}".format(base_yaml))
            config = load_config(base_yaml, override_path)

            # --- Redirect all output paths to nodeCacheFolder ---
            config.cmd_args = {}
            config.trial_name = 'meshroom'
            config.exp_dir = node_cache
            config.save_dir = node_cache
            config.ckpt_dir = os.path.join(node_cache, 'checkpoints')
            config.code_dir = os.path.join(node_cache, 'code')
            config.config_dir = os.path.join(node_cache, 'config')

            # --- Seed ---
            if 'seed' not in config:
                config.seed = int(time.time() * 1000) % (2 ** 31)
            pl.seed_everything(config.seed)

            # --- Instantiate datamodule ---
            dm = datasets.make(config.dataset.name, config.dataset)

            # Match launch.py line 68: standalone creates a system object
            # before the two-phase branch, consuming random state for weight
            # init (~1M values). Without this, system_p1/p2 get different
            # initializations, causing divergence and NaN in phase 2.
            _dummy = systems.make(
                config.system.name, config, load_from_checkpoint=None)
            del _dummy

            # --- Progress + cancellation callback ---
            class MeshroomProgressCallback(pl.Callback):
                """Reports training progress and loss to Meshroom logs."""
                def __init__(self, chunk_ref, total_steps, step_offset=0,
                             log_every=100):
                    self.chunk_ref = chunk_ref
                    self.total_steps = total_steps
                    self.step_offset = step_offset
                    self.log_every = log_every

                def on_train_batch_end(self, trainer, pl_module,
                                       outputs, batch, batch_idx):
                    import math as _math
                    step = trainer.global_step
                    abs_step = self.step_offset + step
                    progress = abs_step / self.total_steps

                    if hasattr(self.chunk_ref.node, 'setProgress'):
                        self.chunk_ref.node.setProgress(progress)

                    # Detect NaN/Inf loss and stop early
                    if isinstance(outputs, dict) and 'loss' in outputs:
                        loss_val = outputs['loss'].item()
                        if _math.isnan(loss_val) or _math.isinf(loss_val):
                            chunk.logger.error(
                                "NaN/Inf loss detected at step {}. "
                                "Stopping training.".format(abs_step))
                            trainer.should_stop = True
                            return

                    # Log progress + loss periodically
                    if step % self.log_every == 0 or step == 1:
                        loss_str = ""
                        if isinstance(outputs, dict) and 'loss' in outputs:
                            loss_str = " | loss={:.6f}".format(
                                outputs['loss'].item())
                        elif hasattr(trainer, 'callback_metrics'):
                            loss_val = trainer.callback_metrics.get(
                                'train/loss', None)
                            if loss_val is not None:
                                loss_str = " | loss={:.6f}".format(
                                    float(loss_val))
                        chunk.logger.info(
                            "Step {}/{} ({:.1f}%){}".format(
                                abs_step, self.total_steps,
                                progress * 100, loss_str))

                    if hasattr(self.chunk_ref.node, 'stopped'):
                        if self.chunk_ref.node.stopped():
                            chunk.logger.info(
                                "Cancellation requested at step {}".format(
                                    step))
                            trainer.should_stop = True

            # --- Helper: build trainer config dict (pop conflicting keys) ---
            def _make_trainer_cfg(cfg):
                tc = OmegaConf.to_container(cfg.trainer, resolve=True)
                for key in ['devices', 'accelerator', 'strategy',
                            'logger', 'callbacks', 'enable_progress_bar']:
                    tc.pop(key, None)
                return tc

            # --- Helper: recompute scheduler gamma for a given max_steps ---
            # (mirrors launch.py _recompute_scheduler exactly)
            warmup_steps = config.system.warmup_steps

            def _recompute_scheduler(cfg, new_max_steps):
                cfg.trainer.max_steps = new_max_steps
                decay_steps = max(new_max_steps - warmup_steps, 1)
                new_gamma = 0.1 ** (1.0 / decay_steps)
                cfg.system.scheduler.schedulers[1].args.gamma = new_gamma
                cfg.checkpoint.every_n_train_steps = new_max_steps

            # --- Detect two-phase training ---
            albedo_cfg = config.system.get('albedo_scaling', {})
            two_phase_explicit = albedo_cfg.get('enabled', None)
            if two_phase_explicit is None:
                two_phase = bool(albedo_sfm_path)
            else:
                two_phase = two_phase_explicit

            if two_phase:
                self._run_two_phase(
                    chunk, config, dm, max_steps, node_cache, accelerator,
                    albedo_cfg,
                    MeshroomProgressCallback, _make_trainer_cfg,
                    _recompute_scheduler,
                    # Open-RNb modules
                    torch=torch, np=np, copy=copy,
                    systems=systems, OmegaConf=OmegaConf,
                    Trainer=Trainer, ModelCheckpoint=ModelCheckpoint,

                    rank_zero_info=rank_zero_info,
                    compute_scaling_from_mesh=compute_scaling_from_mesh,
                    neus_c2w_to_standard=neus_c2w_to_standard,
                    SPACE_NORMALIZED=SPACE_NORMALIZED,
                    MarchingCubeHelper=MarchingCubeHelper,
                    compute_albedo_scale_ratios=compute_albedo_scale_ratios,
                    scale_albedo_images=scale_albedo_images,
                )
            else:
                self._run_single_phase(
                    chunk, config, dm, max_steps, node_cache, accelerator,
                    MeshroomProgressCallback, _make_trainer_cfg,
                    torch=torch, systems=systems, OmegaConf=OmegaConf,
                    Trainer=Trainer, ModelCheckpoint=ModelCheckpoint,

                )

            # --- Move mesh to fixed output path ---
            import glob as glob_mod
            output_mesh = chunk.node.outputMesh.value
            # Open-RNb exports .ply (not .obj); search both formats
            mesh_files = glob_mod.glob(
                os.path.join(node_cache, 'it*-mc*.obj'))
            mesh_files += glob_mod.glob(
                os.path.join(node_cache, 'it*-mc*.ply'))
            if not mesh_files:
                mesh_files = [
                    f for f in (
                        glob_mod.glob(os.path.join(node_cache, '*.obj'))
                        + glob_mod.glob(os.path.join(node_cache, '*.ply'))
                    )
                    if os.path.abspath(f) != os.path.abspath(output_mesh)
                    and 'intermediate' not in os.path.basename(f)
                ]

            if mesh_files:
                import trimesh as _trimesh
                mesh_file = max(mesh_files, key=os.path.getmtime)
                # Re-export as OBJ for Meshroom viewer compatibility
                # (Qt3D/Assimp fails on PLY files from Open-RNb)
                _mesh = _trimesh.load(mesh_file)
                _mesh.export(output_mesh, file_type='obj')
                del _mesh
                if os.path.abspath(mesh_file) != os.path.abspath(output_mesh):
                    os.remove(mesh_file)
                chunk.logger.info(
                    "Mesh exported (OBJ) to: {}".format(output_mesh))
            else:
                raise RuntimeError(
                    "Training completed but no mesh file found in {}".format(
                        node_cache))

            chunk.logger.info("Done.")

        finally:
            # --- GPU cleanup ---
            # dm is the only object from processChunk scope that holds GPU state.
            # system/trainer are local to _run_single_phase / _run_two_phase
            # and cleaned up there.
            if initialized:
                import gc as _gc
                import torch as _torch
                del dm
                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            chunk.logManager.end()

    def _run_single_phase(self, chunk, config, dm, max_steps, node_cache,
                          accelerator, ProgressCb, make_tcfg, **mods):
        """Standard single-phase training."""
        import gc
        torch = mods['torch']
        systems = mods['systems']
        OmegaConf = mods['OmegaConf']
        Trainer = mods['Trainer']
        ModelCheckpoint = mods['ModelCheckpoint']


        system = None
        trainer = None
        try:
            system = systems.make(
                config.system.name, config, load_from_checkpoint=None)

            ckpt_kwargs = OmegaConf.to_container(
                config.get('checkpoint', {}), resolve=True)
            callbacks = [
                ModelCheckpoint(dirpath=config.ckpt_dir, **ckpt_kwargs),

                ProgressCb(chunk, max_steps),
            ]

            trainer = Trainer(
                devices=1, accelerator=accelerator,
                callbacks=callbacks, logger=False, strategy='auto',
                enable_progress_bar=False,
                **make_tcfg(config)
            )

            chunk.logger.info("Starting training for {} steps...".format(max_steps))
            trainer.fit(system, datamodule=dm)
            trainer.test(system, datamodule=dm)
        finally:
            del system, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_two_phase(self, chunk, config, dm, max_steps, node_cache,
                       accelerator, albedo_cfg,
                       ProgressCb, make_tcfg, recompute_sched, **mods):
        """Two-phase training: geometry-only phase 1, then full training with scaled albedos."""
        import gc
        torch = mods['torch']
        np = mods['np']
        copy = mods['copy']
        systems = mods['systems']
        OmegaConf = mods['OmegaConf']
        Trainer = mods['Trainer']
        ModelCheckpoint = mods['ModelCheckpoint']

        compute_scaling_from_mesh = mods['compute_scaling_from_mesh']
        neus_c2w_to_standard = mods['neus_c2w_to_standard']
        SPACE_NORMALIZED = mods['SPACE_NORMALIZED']
        MarchingCubeHelper = mods['MarchingCubeHelper']
        compute_albedo_scale_ratios = mods['compute_albedo_scale_ratios']
        scale_albedo_images = mods['scale_albedo_images']
        import trimesh

        warmup_ratio = albedo_cfg.get('warmup_ratio', 0.1)
        phase1_steps = int(warmup_ratio * max_steps)

        # Validate: rendering lambdas must be scalar (not schedules)
        for key in ['lambda_rendering_mse', 'lambda_rendering_l1']:
            val = config.system.loss[key]
            if isinstance(val, (list, tuple)):
                raise ValueError(
                    "Two-phase training requires scalar {}, got schedule: {}".format(
                        key, val))

        # Validate config.export exists (required for mesh extraction)
        if not hasattr(config, 'export') or config.get('export') is None:
            raise RuntimeError(
                "Base config is missing the 'export' section required for mesh extraction.")

        # ---- PHASE 1: geometry only (no_albedo, shading-only rendering) ----
        chunk.logger.info(
            "[TwoPhase] Phase 1: {} steps, no_albedo=True".format(phase1_steps))
        config_p1 = copy.deepcopy(config)
        recompute_sched(config_p1, phase1_steps)
        config_p1.model.no_albedo = True

        system_p1 = systems.make(
            config_p1.system.name, config_p1, load_from_checkpoint=None)
        dm.setup('fit')

        # Diagnostic: phase 1 scene parameters
        ds_diag = dm.train_dataloader().dataset
        chunk.logger.info(
            "[TwoPhase][Diag] seed={}, P1 scene_center={}, "
            "scale_factor={:.6f}".format(
                config.seed,
                list(ds_diag.scene_center)
                if hasattr(ds_diag.scene_center, '__iter__')
                else ds_diag.scene_center,
                float(ds_diag.scale_factor)))

        # Save real albedos, replace with white for phase 1
        ds = dm.train_dataloader().dataset
        real_albedos = ds.all_images
        ds.all_images = torch.ones_like(ds.all_images)

        ckpt_kwargs_p1 = OmegaConf.to_container(
            config_p1.get('checkpoint', {}), resolve=True)
        callbacks_p1 = [
            ModelCheckpoint(dirpath=config_p1.ckpt_dir, **ckpt_kwargs_p1),
            ProgressCb(chunk, max_steps + phase1_steps, step_offset=0),
        ]
        trainer_p1 = Trainer(
            devices=1, accelerator=accelerator,
            callbacks=callbacks_p1, logger=False, strategy='auto',
            enable_progress_bar=False,
            **make_tcfg(config_p1)
        )

        trainer_p1.fit(system_p1, datamodule=dm)

        # Restore real albedos
        ds.all_images = real_albedos

        # ---- ALBEDO SCALING: extract intermediate mesh + compute ratios ----
        chunk.logger.info("[TwoPhase] Extracting intermediate mesh for albedo scaling")
        if torch.cuda.is_available():
            system_p1.model.cuda()
        mesh_res = albedo_cfg.get('intermediate_mesh_resolution', 512)
        geom = system_p1.model.geometry
        use_torch = config.model.geometry.isosurface.method == 'mc-torch'
        geom.helper = MarchingCubeHelper(mesh_res, use_torch=use_torch)

        export_cfg_p1 = copy.deepcopy(config.export)
        export_cfg_p1.export_vertex_color = False
        export_cfg_p1.isosurface_space = SPACE_NORMALIZED
        mesh = system_p1.model.export(export_cfg_p1)

        verts_norm = mesh['v_pos'].cpu().numpy()
        faces = mesh['t_pos_idx'].cpu().numpy()
        del mesh

        chunk.logger.info(
            "[TwoPhase][Diag] Intermediate mesh: {} vertices, {} faces, "
            "bbox_min={}, bbox_max={}".format(
                verts_norm.shape[0], faces.shape[0],
                verts_norm.min(axis=0).tolist(),
                verts_norm.max(axis=0).tolist()))

        # Save intermediate mesh in WORLD space (for debug)
        p1_center = np.array(ds.scene_center, dtype=np.float64)
        p1_scale = float(ds.scale_factor)
        verts_world_inter = verts_norm / p1_scale + p1_center
        trimesh.Trimesh(vertices=verts_world_inter, faces=faces).export(
            os.path.join(node_cache, 'intermediate_mesh.ply'))

        # ---- SCENE RENORMALIZATION from intermediate mesh ----
        sphere_scale_p2 = albedo_cfg.get('sphere_scale_p2', 1.5)
        new_center, new_scale = compute_scaling_from_mesh(
            verts_world_inter, sphere_scale=sphere_scale_p2)
        chunk.logger.info(
            "[TwoPhase] Scene renormalized: center={}, scale={:.6f}".format(
                new_center.tolist(), new_scale))

        # Free GPU memory before CPU-heavy albedo scaling
        del system_p1, trainer_p1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Albedo scaling in P1-normalized space
        tri_mesh_norm = trimesh.Trimesh(vertices=verts_norm, faces=faces)
        all_c2w_np = ds.all_c2w.cpu().numpy()
        norm_c2ws = [neus_c2w_to_standard(c2w34) for c2w34 in all_c2w_np]

        scale_ratios = compute_albedo_scale_ratios(
            albedo_images=[img.cpu().numpy() for img in ds.all_images],
            camera_Ks=ds.camera_Ks,
            camera_c2ws=norm_c2ws,
            tri_mesh=tri_mesh_norm,
            n_samples=albedo_cfg.get('n_samples', 2000),
        )
        del tri_mesh_norm
        scaled = scale_albedo_images(ds.all_images, scale_ratios)
        ds.update_albedos(scaled)
        chunk.logger.info(
            "[TwoPhase] Albedos scaled. Mean={}, Min={}, Max={}".format(
                scale_ratios.mean(axis=0).tolist(),
                scale_ratios.min(axis=0).tolist(),
                scale_ratios.max(axis=0).tolist()))

        # ---- RENORMALIZE CAMERAS P1 -> P2 ----
        # dm.setup() is IDEMPOTENT — it must not re-create train_dataset/val_dataset,
        # otherwise scaled albedos and renormalized cameras from phase 1 will be lost.
        dm.setup('test')
        p2_scale = float(new_scale)
        p2_center = np.array(new_center, dtype=np.float64)
        renorm_ratio = p2_scale / p1_scale
        renorm_offset = torch.tensor(
            p2_scale * (p1_center - p2_center), dtype=torch.float32)
        for split_ds in [dm.train_dataset, dm.val_dataset, dm.test_dataset]:
            dev = split_ds.all_c2w.device
            split_ds.all_c2w[:, :3, 3] = (
                renorm_ratio * split_ds.all_c2w[:, :3, 3]
                + renorm_offset.to(dev)
            )
            split_ds.scene_center = new_center
            split_ds.scale_factor = new_scale
        chunk.logger.info(
            "[TwoPhase] Cameras renormalized P1->P2 (ratio={:.6f})".format(
                renorm_ratio))

        # ---- PHASE 2: fresh model, full training with scaled albedos ----
        chunk.logger.info(
            "[TwoPhase] Phase 2: {} steps, fresh model, rendering loss active".format(
                max_steps))
        chunk.logger.info(
            "[TwoPhase][Diag] P2 scene_center={}, scale_factor={:.6f}, "
            "renorm_ratio={:.6f}".format(
                p2_center.tolist(), p2_scale, renorm_ratio))
        config_p2 = copy.deepcopy(config)
        recompute_sched(config_p2, max_steps)

        system_p2 = None
        trainer_p2 = None
        try:
            system_p2 = systems.make(config_p2.system.name, config_p2)
            ckpt_kwargs_p2 = OmegaConf.to_container(
                config_p2.get('checkpoint', {}), resolve=True)
            callbacks_p2 = [
                ModelCheckpoint(dirpath=config_p2.ckpt_dir, **ckpt_kwargs_p2),

                ProgressCb(chunk, max_steps + phase1_steps, step_offset=phase1_steps),
            ]
            trainer_p2 = Trainer(
                devices=1, accelerator=accelerator,
                callbacks=callbacks_p2, logger=False, strategy='auto',
                enable_progress_bar=False,
                **make_tcfg(config_p2)
            )

            trainer_p2.fit(system_p2, datamodule=dm)
            trainer_p2.test(system_p2, datamodule=dm)
        finally:
            del system_p2, trainer_p2
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _generate_mask_sfm(chunk, normal_sfm_path, mask_folder):
        """Generate a mask SfMData JSON from the normal SfMData + a folder of masks.

        Masks are matched to views by viewId: a mask file matches if its
        filename (without extension) contains the viewId string.
        E.g. viewId '12345' matches '12345.png', 'mask_12345.exr', etc.
        """
        import json
        import copy

        with open(normal_sfm_path, 'r') as f:
            sfm_data = json.load(f)

        # Index available mask files by scanning the folder once
        mask_files = [
            entry.name for entry in os.scandir(mask_folder)
            if entry.is_file()
        ]

        matched = 0
        views_out = []
        for view in sfm_data.get('views', []):
            view_id = str(view['viewId'])
            # Find mask whose stem contains the viewId
            candidates = [
                f for f in mask_files
                if view_id in os.path.splitext(f)[0]
            ]
            if not candidates:
                chunk.logger.warning(
                    "No mask found for viewId {} in {}".format(
                        view_id, mask_folder))
                continue
            if len(candidates) > 1:
                chunk.logger.warning(
                    "Multiple masks match viewId {}: {}. Using first.".format(
                        view_id, candidates))
            mask_path = os.path.join(mask_folder, candidates[0])
            view_copy = copy.deepcopy(view)
            view_copy['path'] = mask_path
            views_out.append(view_copy)
            matched += 1

        if matched == 0:
            raise RuntimeError(
                "No masks matched any viewId from {}. "
                "Check that mask filenames contain the viewId.".format(
                    normal_sfm_path))

        sfm_out = copy.deepcopy(sfm_data)
        sfm_out['views'] = views_out

        node_cache = chunk.node.outputFolder.value
        os.makedirs(node_cache, exist_ok=True)
        out_path = os.path.join(node_cache, 'generated_mask_sfm.json')
        with open(out_path, 'w') as f:
            json.dump(sfm_out, f, indent=2)

        chunk.logger.info(
            "Generated mask SfM from folder: {}/{} views matched -> {}".format(
                matched, len(sfm_data.get('views', [])), out_path))
        return out_path
