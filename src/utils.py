"""Utility functions for the AudibleLight dataset generator."""

import argparse
from pathlib import Path
from typing import Union

import audiblelight
import numpy as np
import trimesh
from audiblelight.class_mappings import ClassMapping
from audiblelight.download_data import download_gibson

DEFAULT_FG_DIR = Path("data/esc50/fg_esc50_24k_mono")
DEFAULT_OUT_ROOT = Path("output")
DEFAULT_AUDIO_OUT = DEFAULT_OUT_ROOT.joinpath("em32_dev", "dev-train")
DEFAULT_META_OUT = DEFAULT_OUT_ROOT.joinpath("metadata_dev", "dev-train")


class AlwaysClass0Mapping(ClassMapping):
    """A ClassMapping that always returns class index 0 with a dummy label."""

    def __init__(self) -> None:
        super().__init__(mapping={"dummy": 0})

    def infer_label_idx_from_filepath(
        self, filepath: Union[Path, str]
    ) -> tuple[int, str]:
        """Return class index 0 and label 'dummy'."""
        return (0, "dummy")


def add_arguments(ap: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Add command-line arguments to the argument parser for audio dataset generation.

    Parameters
    ----------
    ap: argparse.ArgumentParser
        The argument parser to add arguments to.

    Returns
    -------
    argparse.Namespace
        The parsed command-line arguments.
    """
    ap.add_argument(
        "--fg-dir",
        type=Path,
        default=DEFAULT_FG_DIR,
        help="Directory containing foreground audio files.",
    )
    ap.add_argument(
        "--audio-out",
        type=Path,
        default=DEFAULT_AUDIO_OUT,
        help="Directory to save generated audio files.",
    )
    ap.add_argument(
        "--meta-out",
        type=Path,
        default=DEFAULT_META_OUT,
        help="Directory to save metadata files.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    ap.add_argument(
    "--mesh-dir",
    type=Path,
    default=Path("data/gibson"),
    help="Directory containing .glb meshes for the rlr backend.",
    )
    ap.add_argument(
        "--download-gibson",
        action="store_true",
        help="If set and no meshes are found, download default Gibson meshes into --mesh-dir.",
    )

    ap.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Sample rate for audio generation.",
    )
    ap.add_argument(
        "--scene-duration",
        type=float,
        default=60.0,
        help="Duration of each generated audio file in seconds.",
    )
    ap.add_argument(
        "--events-per-scene",
        type=int,
        default=6,
        help="Number of foreground events to include in each scene.",
    )
    ap.add_argument(
        "--max-overlap",
        type=int,
        default=3,
        help="Maximum allowed overlap between foreground events in seconds.",
    )

    ap.add_argument(
        "--event-duration-min",
        type=float,
        default=0.5,
        help="Minimum duration of foreground events in seconds.",
    )
    ap.add_argument(
        "--event-duration-max",
        type=float,
        default=10.0,
        help="Maximum duration of foreground events in seconds.",
    )
    ap.add_argument(
        "--snr-min",
        type=float,
        default=0.0,
        help="Minimum signal-to-noise ratio (SNR) for foreground events in decibels.",
    )
    ap.add_argument(
        "--snr-max",
        type=float,
        default=30.0,
        help="Maximum signal-to-noise ratio (SNR) for foreground events in decibels.",
    )

    ap.add_argument(
        "--num-scenes",
        type=int,
        default=100,
        help="Number of scenes to generate.",
    )
    ap.add_argument(
        "--num-mics-per-scene",
        type=int,
        default=5,
        help="Number of microphones to include in each scene.",
    )

    ap.add_argument(
        "--mic-type",
        type=str,
        default="eigenmike32",
        help="Type of microphone to use for recording (e.g., 'eigenmike32' or 'eigenmike64').",
    )

    return ap.parse_args()

def list_audio_files(root_dir: Path) -> list[Path]:
    """
    List all audio files in the specified directory.

    Parameters
    ----------
    root_dir: Path
        The root directory to search for audio files.

    Returns
    -------
    list[Path]
        A sorted list of Path objects representing the audio files found in the directory.
    """
    if not root_dir.is_dir() or not root_dir.exists():
        raise ValueError(f"The specified path '{root_dir}' is not a valid directory.")
    audio_files = list(root_dir.glob("*.wav"))
    return sorted(audio_files)

def list_mesh_files(mesh_dir: Path) -> list[Path]:
    """
    Recursively retrieve all GLB mesh files from a directory.

    Parameters
    ----------
    mesh_dir: Path
        Object pointing to the directory to search for mesh files.

    Returns
    -------
    : list[Path]
        A sorted list of Path objects for all found GLB files.

    Raises
    ------
    ValueError
        If the specified path is not a valid directory or does not exist.
    """
    if not mesh_dir.is_dir() or not mesh_dir.exists():
        raise ValueError(f"The specified path '{mesh_dir}' is not a valid directory.")
    mesh_files = [p for p in mesh_dir.rglob("*.glb") if p.is_file()]
    return sorted(mesh_files)

def ensure_meshes(mesh_dir: Path, download_gibson_flag: bool) -> list[Path]:
    """
    Ensure mesh files are available in the specified directory.

    Attempts to find existing mesh files in the given directory. If none are found
    and downloading is enabled, downloads the Gibson dataset. Raises an error if
    no meshes are found after attempting to download.

    Parameters
    ----------
    mesh_dir: Path
        Path to the directory containing or to store mesh files.
    download_gibson_flag: bool
        If True, downloads Gibson dataset when no meshes are found.
        If False, raises an error when no meshes are found.

    Returns
    -------
    : list[Path]
        A list of Path objects pointing to the found mesh files (.glb format).

    Raises
    ------
    ValueError
        If no mesh files are found and download_gibson_flag is False.
    RuntimeError
        If no mesh files are found even after downloading Gibson dataset.
    """
    meshes = list_mesh_files(mesh_dir)
    if meshes:
        return meshes
    if not download_gibson_flag:
        raise ValueError(
            f"No mesh files found in '{mesh_dir}'. "
            "Use --download-gibson or point --mesh-dir to meshes."
        )

    mesh_dir.mkdir(parents=True, exist_ok=True)

    download_gibson(path=str(mesh_dir), cleanup=False, remote=["habitat_1.5gb"])

    meshes = list_mesh_files(mesh_dir)
    if not meshes:
        raise RuntimeError(f"Downloaded Gibson but still found no .glb meshes under '{mesh_dir}'.")
    return meshes

def build_backend_kwargs_rlr(mesh_path: Path) -> dict[str, object]:
    """
    Build backend keyword arguments for RLR (Ray-based Light Rendering) backend.

    Parameters
    ----------
    mesh_path: Path
        Path to the mesh file to be used in RLR rendering.

    Returns
    -------
    : dict[str, object]
        A dictionary containing backend configuration with mesh path (as string)
        and context addition flag set to False.
    """
    return {
        "mesh": str(mesh_path),
        "add_to_context": False,
    }

def add_random_microphone(  # noqa: PLR0913
    scene: audiblelight.Scene,
    mic_type: str,
    mesh: trimesh.Geometry,
    rng: np.random.Generator,
    rng_needed: bool = True,
    max_attempts: int = 30,
    ) -> bool:
    """
    Attempt to add a microphone at a random valid position in the scene.

    Parameters
    ----------
    scene: audiblelight.Scene
        The scene to which the microphone should be added.
    mic_type: str
        The type of microphone to add (e.g., 'eigenmike32').
    mesh: trimesh.Geometry
        The mesh geometry of the scene, used to determine valid microphone positions.
    rng: np.random.Generator
        A random number generator for reproducibility.
    rng_needed: bool, optional
        If True, use the provided RNG to generate random positions.
        If False, rely on the backend to choose random positions. Default is True.
    max_attempts: int, optional
        The maximum number of attempts to try adding a microphone at a random position before
        giving up.
        Default is 30.

    Returns
    -------
    : bool
        True if the microphone was successfully added, False otherwise.
    """
    if rng_needed:
        min_mesh_bound, max_mesh_bound = mesh.bounds
        for attempt in range(max_attempts):
            mic_position = min_mesh_bound + (max_mesh_bound - min_mesh_bound) * rng.random(3)
            try:
                scene.add_microphone(
                    microphone_type=mic_type,
                    position=mic_position,
                )
                return True  # Successfully added microphone, exit the retry loop
            except ValueError:
                if attempt == max_attempts - 1:
                # Give up on adding this microphone and move on to the next one
                    return False
        return False
    else:
        try:
            scene.add_microphone(
                microphone_type=mic_type,
                position=None,  # Let the backend choose a valid random position
            )
            return True
        except ValueError:
            return False

def add_random_fg_event(  # noqa: PLR0913
    fg_files: list[Path],
    scene: audiblelight.Scene,
    scene_duration: float,
    event_duration_min: float,
    event_duration_max: float,
    snr_min: float,
    snr_max: float,
    mesh: trimesh.Geometry,
    rng: np.random.Generator,
    rng_needed: bool = False,
    max_attempts: int = 30,
) -> bool:
    """
    Attempt to add a foreground event at a random valid position in the scene.

    Parameters
    ----------
    fg_files: list[Path]
        A list of Path objects pointing to available foreground audio files.
    scene: audiblelight.Scene
        The scene to which the foreground event should be added.
    scene_duration: float
        The total duration of the scene in seconds, used to determine valid event start times.
    event_duration_min: float
        The minimum duration of the foreground event in seconds.
    event_duration_max: float
        The maximum duration of the foreground event in seconds.
    snr_min: float
        The minimum signal-to-noise ratio (SNR) for the foreground event in decibels.
    snr_max: float
        The maximum signal-to-noise ratio (SNR) for the foreground event in decibels.
    mesh: trimesh.Geometry
        The mesh geometry of the scene, used to determine valid event positions.
    rng: np.random.Generator
        A random number generator for reproducibility.
    rng_needed: bool, optional
        If True, use the provided RNG to generate random event positions.
        If False, rely on the backend to choose random positions. Default is False.
    max_attempts: int, optional
        The maximum number of attempts to try adding a foreground event at a random position before
        giving up.
        Default is 30.
    """
    wav = Path(fg_files[int(rng.integers(0, len(fg_files)))])
    event_duration = float(rng.uniform(event_duration_min, event_duration_max))
    event_start = float(rng.uniform(0, scene_duration - event_duration))
    signal_to_noise_ratio = float(rng.uniform(snr_min, snr_max))

    if rng_needed:
        # Not working properly yet as the current RLR backend implementation
        # does not properly handle invalid event positions and gives us no way to query
        # valid positions before trying to add the event.
        min_mesh_bound, max_mesh_bound = mesh.bounds
        margins = (max_mesh_bound - min_mesh_bound) * 0.1
        min_mesh_bound += margins
        max_mesh_bound -= margins
        for attempt in range(max_attempts):
            event_position = min_mesh_bound + (max_mesh_bound - min_mesh_bound) * rng.random(3)
            try:
                scene.add_event_static(
                    filepath=wav,
                    position=event_position,
                    scene_start=event_start,
                    duration=event_duration,
                    snr=signal_to_noise_ratio,
                    class_id=0,
                )
                return True  # Successfully added event, exit the retry loop
            except ValueError:
                if attempt == max_attempts - 1:
                # Give up on adding this event and move on to the next one
                    return False
        return False
    else:
        try:
            scene.add_event_static(
                filepath=wav,
                position=None,  # Let the backend choose a valid random position
                scene_start=event_start,
                duration=event_duration,
                snr=signal_to_noise_ratio,
                class_id=0,
            )
            return True
        except ValueError:
            return False
