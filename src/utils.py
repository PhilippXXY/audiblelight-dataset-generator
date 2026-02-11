"""Utility functions for the AudibleLight dataset generator."""

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import audiblelight
import numpy as np
import trimesh
import yaml
from audiblelight.class_mappings import ClassMapping
from audiblelight.download_data import download_gibson

DEFAULT_FG_DIR = Path("data/esc50/fg_esc50_24k_mono")
DEFAULT_MESH_DIR = Path("data/gibson")


@dataclass(frozen=True)
class PathsConfig:
    """Path configuration for data input/output."""

    fg_dir: Path
    audio_out: Path
    meta_out: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime controls for scene generation."""

    seed: int
    num_scenes: int
    num_mics_per_scene: int


@dataclass(frozen=True)
class MeshConfig:
    """Mesh discovery."""

    mesh_dir: Path
    download_gibson_flag: bool


@dataclass(frozen=True)
class SceneConfig:
    """Scene-wide simulation configuration."""

    sample_rate: int
    scene_duration: float
    max_overlap: int
    mic_type: str
    bg_noise_floor_db: float


@dataclass(frozen=True)
class EventsConfig:
    """Foreground event generation configuration."""

    events_per_scene: int
    event_duration_min: float
    event_duration_max: float
    snr_min: float
    snr_max: float


@dataclass(frozen=True)
class GeneratorConfig:
    """Top-level configuration consumed by dataset generation."""

    paths: PathsConfig
    runtime: RuntimeConfig
    mesh: MeshConfig
    scene: SceneConfig
    events: EventsConfig


class AlwaysClass0Mapping(ClassMapping):  # type: ignore[no-any-unimported]
    """A ClassMapping that always returns class index 0 with a dummy label."""

    def __init__(self) -> None:
        super().__init__(mapping={"dummy": 0})

    def infer_label_idx_from_filepath(self, filepath: Union[Path, str]) -> tuple[int, str]:
        """Return class index 0 and label 'dummy'."""
        return (0, "dummy")


def _build_default_output_paths() -> tuple[Path, Path]:
    """
    Build timestamped default output paths.

    Returns
    -------
    : tuple[Path, Path]
        A tuple containing the default audio output directory and metadata output directory
        as Path objects.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_root = Path("output").joinpath(f"dataset_{timestamp}")
    audio_out = output_root.joinpath("em32_dev", "dev-train")
    meta_out = output_root.joinpath("metadata_dev", "dev-train")
    return audio_out, meta_out


def _default_config_dict() -> dict[str, dict[str, Any]]:
    """
    Return default config values equivalent to prior argparse defaults.

    Returns
    -------
    : dict[str, dict[str, Any]]
        A nested dictionary containing default configuration values for all config sections.
    """
    default_audio_out, default_meta_out = _build_default_output_paths()
    return {
        "paths": {
            "fg_dir": DEFAULT_FG_DIR,
            "audio_out": default_audio_out,
            "meta_out": default_meta_out,
        },
        "runtime": {
            "seed": 0,
            "num_scenes": 100,
            "num_mics_per_scene": 5,
        },
        "mesh": {
            "mesh_dir": DEFAULT_MESH_DIR,
            "download_gibson_flag": True,
        },
        "scene": {
            "sample_rate": 24000,
            "scene_duration": 60.0,
            "max_overlap": 15,
            "mic_type": "eigenmike32",
            "bg_noise_floor_db": -50.0,
        },
        "events": {
            "events_per_scene": 15,
            "event_duration_min": 0.5,
            "event_duration_max": 10.0,
            "snr_min": 0.0,
            "snr_max": 30.0,
        },
    }


def _normalise_mapping(value: Any, key_name: str) -> dict[str, Any]:
    """
    Validate that a YAML value is a mapping with string keys.

    Parameters
    ----------
    value: Any
        The value to validate and normalise.
    key_name: str
        The name of the config key being normalised, used for error messages.

    Returns
    -------
    : dict[str, Any]
        The normalised mapping with string keys.

    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key_name}' must be a mapping.")

    normalized: dict[str, Any] = {}
    for key, mapping_value in value.items():
        if not isinstance(key, str):
            raise ValueError(f"Config key '{key_name}' contains a non-string key: {key!r}.")
        normalized[key] = mapping_value
    return normalized


def _warn_unknown_keys(
    config_values: dict[str, Any],
    allowed_values: dict[str, Any],
    section_name: str | None = None,
) -> None:
    """
    Warn about unknown keys while leaving config parsing permissive.

    Parameters
    ----------
    config_values: dict[str, Any]
        The config values to check for unknown keys.
    allowed_values: dict[str, Any]
        The allowed config keys to check against.
    section_name: str | None, optional
        The name of the config section being checked, used for more specific warning messages.
        If None, a more generic warning message will be issued. Default is None.

    Returns
    -------
    : None
    """
    unknown_keys = sorted(set(config_values).difference(set(allowed_values)))
    for key in unknown_keys:
        if section_name is None:
            warnings.warn(f"Unknown config section '{key}' will be ignored.", stacklevel=2)
        else:
            warnings.warn(
                f"Unknown config key '{section_name}.{key}' will be ignored.",
                stacklevel=2,
            )


def _coerce_bool(value: Any, key_name: str) -> bool:
    """
    Coerce a config value to bool, failing on invalid values.

    Parameters
    ----------
    value: Any
        The value to coerce to bool.
    key_name: str
        The name of the config key being coerced, used for error messages.

    Returns
    -------
    : bool
        The coerced boolean value.
    """
    if isinstance(value, bool):
        return value
    raise ValueError(f"Config key '{key_name}' must be a boolean.")


def _coerce_int(value: Any, key_name: str) -> int:
    """
    Coerce a config value to int, failing on invalid values.

    Parameters
    ----------
    value: Any
        The value to coerce to int.
    key_name: str
        The name of the config key being coerced, used for error messages.

    Returns
    -------
    : int
        The coerced integer value.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config key '{key_name}' must be an integer.")
    return int(value)


def _coerce_float(value: Any, key_name: str) -> float:
    """
    Coerce a config value to float, failing on invalid values.

    Parameters
    ----------
    value: Any
        The value to coerce to float.
    key_name: str
        The name of the config key being coerced, used for error messages.

    Returns
    -------
    : float
        The coerced float value.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config key '{key_name}' must be a number.")
    return float(value)


def _coerce_str(value: Any, key_name: str) -> str:
    """
    Coerce a config value to str, failing on invalid values.

    Parameters
    ----------
    value: Any
        The value to coerce to str.
    key_name: str
        The name of the config key being coerced, used for error messages.

    Returns
    -------
    : str
        The coerced string value.
    """
    if not isinstance(value, str):
        raise ValueError(f"Config key '{key_name}' must be a string.")
    if value == "":
        raise ValueError(f"Config key '{key_name}' must not be empty.")
    return value


def _coerce_path(value: Any, key_name: str) -> Path:
    """
    Coerce a config value to Path, failing on invalid values.

    Parameters
    ----------
    value: Any
        The value to coerce to Path.
    key_name: str
        The name of the config key being coerced, used for error messages.

    Returns
    -------
    : Path
        The coerced Path value.
    """
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        if value == "":
            raise ValueError(f"Config key '{key_name}' must not be empty.")
        path = Path(value)
    else:
        raise ValueError(f"Config key '{key_name}' must be a path string.")
    return path


def load_config(config_path: Path | str) -> GeneratorConfig:
    """
    Load generator config from YAML with defaults, warnings, and validation.

    Parameters
    ----------
    config_path: Path | str
        The path to the YAML configuration file.

    Returns
    -------
    : GeneratorConfig
        The loaded and validated generator configuration.
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path.cwd().joinpath(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: '{config_file}'.")

    with config_file.open(encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)

    raw_config = _normalise_mapping(loaded_config, "root")
    defaults = _default_config_dict()
    _warn_unknown_keys(raw_config, defaults)

    merged_config: dict[str, dict[str, Any]] = {}
    for section_name, default_values in defaults.items():
        raw_section = _normalise_mapping(raw_config.get(section_name, {}), section_name)
        _warn_unknown_keys(raw_section, default_values, section_name=section_name)
        merged_section = default_values.copy()
        merged_section.update(raw_section)
        merged_config[section_name] = merged_section

    paths_section = merged_config["paths"]
    runtime_section = merged_config["runtime"]
    mesh_section = merged_config["mesh"]
    scene_section = merged_config["scene"]
    events_section = merged_config["events"]

    runtime_num_scenes = _coerce_int(runtime_section["num_scenes"], "runtime.num_scenes")
    if runtime_num_scenes <= 0:
        raise ValueError("Config key 'runtime.num_scenes' must be greater than 0.")

    return GeneratorConfig(
        paths=PathsConfig(
            fg_dir=_coerce_path(paths_section["fg_dir"], "paths.fg_dir"),
            audio_out=_coerce_path(paths_section["audio_out"], "paths.audio_out"),
            meta_out=_coerce_path(paths_section["meta_out"], "paths.meta_out"),
        ),
        runtime=RuntimeConfig(
            seed=_coerce_int(runtime_section["seed"], "runtime.seed"),
            num_scenes=runtime_num_scenes,
            num_mics_per_scene=_coerce_int(
                runtime_section["num_mics_per_scene"], "runtime.num_mics_per_scene"
            ),
        ),
        mesh=MeshConfig(
            mesh_dir=_coerce_path(mesh_section["mesh_dir"], "mesh.mesh_dir"),
            download_gibson_flag=_coerce_bool(
                mesh_section["download_gibson_flag"], "mesh.download_gibson_flag"
            ),
        ),
        scene=SceneConfig(
            sample_rate=_coerce_int(scene_section["sample_rate"], "scene.sample_rate"),
            scene_duration=_coerce_float(scene_section["scene_duration"], "scene.scene_duration"),
            max_overlap=_coerce_int(scene_section["max_overlap"], "scene.max_overlap"),
            mic_type=_coerce_str(scene_section["mic_type"], "scene.mic_type"),
            bg_noise_floor_db=_coerce_float(
                scene_section["bg_noise_floor_db"], "scene.bg_noise_floor_db"
            ),
        ),
        events=EventsConfig(
            events_per_scene=_coerce_int(
                events_section["events_per_scene"], "events.events_per_scene"
            ),
            event_duration_min=_coerce_float(
                events_section["event_duration_min"], "events.event_duration_min"
            ),
            event_duration_max=_coerce_float(
                events_section["event_duration_max"], "events.event_duration_max"
            ),
            snr_min=_coerce_float(events_section["snr_min"], "events.snr_min"),
            snr_max=_coerce_float(events_section["snr_max"], "events.snr_max"),
        ),
    )


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
        print(f"The specified mesh directory '{mesh_dir}' is not valid. Download dataset...")
        # raise ValueError(f"The specified path '{mesh_dir}' is not a valid directory.")
        return []
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
    if meshes is not None and len(meshes) > 0:
        return meshes
    elif not download_gibson_flag:
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


def add_random_microphone(  # type: ignore[no-any-unimported] # noqa: PLR0913
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
                    # Otherwise add one using backend random placement
                    return add_random_microphone(
                        scene=scene,
                        mic_type=mic_type,
                        mesh=mesh,
                        rng=rng,
                        rng_needed=False,
                        max_attempts=max_attempts,
                    )
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


def add_random_fg_event(  # type: ignore[no-any-unimported] # noqa: PLR0913
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
                    augmentations=3,
                )
                return True  # Successfully added event, exit the retry loop
            except ValueError:
                if attempt == max_attempts - 1:
                    # Otherwise add one using backend random placement
                    return add_random_fg_event(
                        fg_files=fg_files,
                        scene=scene,
                        scene_duration=scene_duration,
                        event_duration_min=event_duration_min,
                        event_duration_max=event_duration_max,
                        snr_min=snr_min,
                        snr_max=snr_max,
                        mesh=mesh,
                        rng=rng,
                        rng_needed=False,
                        max_attempts=max_attempts,
                    )
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


def get_random_bg_noise(rng: np.random.Generator) -> str:
    """
    Select a random background noise type.

    Parameters
    ----------
    rng: np.random.Generator
        A numpy random number generator instance.

    Returns
    -------
    : str
        A string representing a randomly selected background noise type.
        Possible values are: "white", "pink", or "gaussian".
    """
    bg_noises = ["white", "pink", "gaussian"]
    return bg_noises[int(rng.integers(0, len(bg_noises)))]
