"""
Generate synthetic acoustic datasets using AudibleLight simulation framework.

This module provides functionality to generate realistic room impulse responses
and acoustic scenes with multiple sound events for dataset creation.
"""

import shutil
import tempfile
from pathlib import Path

import audiblelight
import numpy as np
import trimesh
from audiblelight.augmentation import (
    Compressor,
    Fade,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    LowpassFilter,
    LowShelfFilter,
)
from tqdm import tqdm

import utils


def main(config_path: Path | str = Path("config/config.yaml")) -> None:
    """Generate a dataset with AudibleLight."""
    cfg = utils.load_config(config_path)
    rng = np.random.default_rng(seed=cfg.runtime.seed)

    fg_files = utils.list_audio_files(cfg.paths.fg_dir)
    if not fg_files:
        raise ValueError(
            "No audio files found in the specified foreground directory: "
            f"'{cfg.paths.fg_dir}'"
        )

    audio_out = cfg.paths.audio_out
    meta_out = cfg.paths.meta_out
    audio_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    n_scenes = cfg.runtime.num_scenes

    meshes = utils.ensure_meshes(cfg.mesh.mesh_dir, cfg.mesh.download_gibson_flag)

    event_augs = [
        Compressor,
        Fade,
        Gain,
        HighpassFilter,
        HighShelfFilter,
        LowpassFilter,
        LowShelfFilter,
    ]

    for scene_idx in tqdm(range(n_scenes), desc="Generating scenes"):
        stem = f"scene_{scene_idx:05d}"

        mesh_path = meshes[int(rng.integers(0, len(meshes)))]
        mesh = trimesh.load(mesh_path)
        backend_kwargs = utils.build_backend_kwargs_rlr(mesh_path)

        scene = audiblelight.Scene(
            duration=cfg.scene.scene_duration,
            sample_rate=cfg.scene.sample_rate,
            backend="rlr",
            backend_kwargs=backend_kwargs,
            fg_path=cfg.paths.fg_dir,
            max_overlap=cfg.scene.max_overlap,
            class_mapping=utils.AlwaysClass0Mapping(),
            event_augmentations=event_augs,
            ref_db=cfg.scene.bg_noise_floor_db,
        )

        # Add microphones at random positions within the mesh bounds for this scene
        for _ in range(cfg.runtime.num_mics_per_scene):
            utils.add_random_microphone(
                scene,
                mic_type=cfg.scene.mic_type,
                mesh=mesh,
                rng=rng,
                max_attempts=30,
            )

        for _ in range(cfg.events.events_per_scene):
            utils.add_random_fg_event(
                fg_files=fg_files,
                scene=scene,
                scene_duration=cfg.scene.scene_duration,
                event_duration_min=cfg.events.event_duration_min,
                event_duration_max=cfg.events.event_duration_max,
                snr_min=cfg.events.snr_min,
                snr_max=cfg.events.snr_max,
                mesh=mesh,
                rng=rng,
                rng_needed=False,
                max_attempts=30,
            )

        scene.add_ambience(noise=utils.get_random_bg_noise(rng))

        # Render to a temporary directory and then move to the final output location
        # in the DCASE-like format.
        with tempfile.TemporaryDirectory(prefix="audiblelight_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            scene.generate(
                output_dir=tmpdir_path,
                audio=True,
                metadata_json=False,
                metadata_dcase=True,
                audio_fname=stem,
                metadata_fname=stem,
            )

            wavs = {p.stem: p for p in tmpdir_path.glob("*.wav")}
            csvs = {p.stem: p for p in tmpdir_path.glob("*.csv")}
            common = sorted(set(wavs) & set(csvs))
            if len(common) != cfg.runtime.num_mics_per_scene:
                raise RuntimeError(
                    f"Expected {cfg.runtime.num_mics_per_scene} wav/csv pairs, got {len(common)}. "
                    f"WAVs={len(wavs)}, CSVs={len(csvs)} in '{tmpdir_path}'."
                )

            for i, in_stem in enumerate(common):
                out_stem = f"{stem}_mic{i:02d}"
                shutil.move(str(wavs[in_stem]), str(audio_out.joinpath(f"{out_stem}.wav")))
                shutil.move(str(csvs[in_stem]), str(meta_out.joinpath(f"{out_stem}.csv")))

    print(f"Wrote {n_scenes} scenes")
    print(f"Audio:    {audio_out}")
    print(f"Metadata: {meta_out}")


if __name__ == "__main__":
    main()
