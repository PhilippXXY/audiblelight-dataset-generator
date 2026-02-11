# AudibleLight Dataset Generator

<div style="display:flex;justify-content:space-evenly;align-items:center;">
  <img src="assets/img/logo_qmul.jpg" alt="QMUL" height="100">
</div>

## Overview

[![DOI](https://zenodo.org/badge/1152975535.svg)](https://doi.org/10.5281/zenodo.18607188)
![GitHub Release](https://img.shields.io/github/v/release/philippxxy/audiblelight-dataset-generator)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/PhilippXXY/audiblelight-dataset-generator/.github%2Fworkflows%2Frelease.yaml)
![platform](https://img.shields.io/badge/platform-linux%2Famd64-blue?logo=linux)

Generate synthetic acoustic datasets with the AudibleLight simulation framework [[1]](https://github.com/audiblelight/audiblelight).

---

## How to run

### Quickstart (Docker)

After cloning the repository, use the provided helper script to launch **Docker** with the appropriate directory mounts.
```bash
# First time download
./scripts/docker-infer.sh
```

**Docker notes**
- If no external config is provided, the script uses [`config/config.yaml`](config/config.yaml).
- Outputs default to:
  - `~/Downloads/audiblelight-dataset-generator/output/dataset_<timestamp>/em32_dev/dev-train`
  - `~/Downloads/audiblelight-dataset-generator/output/dataset_<timestamp>/metadata_dev/dev-train`
- **Gibson meshes** are cached in a Docker volume by default:
  - Volume name: `audiblelight-gibson-cache`
  - Disable cache: `GIBSON_VOLUME=none`
- The container runs in `linux/amd64` mode only.

**Other commands**
```bash
# Use custom config
./scripts/docker-infer.sh /path/to/config.yaml
# Image already downloaded
DOCKER_PULL=false ./scripts/docker-infer.sh
# Use different docker image
DOCKER_IMAGE=ghcr.io/philippxxy/audiblelight-dataset-generator:v1.0.0 ./scripts/docker-infer.sh
```
Only `paths.audio_out` and `paths.meta_out` are used from the host. Foreground data must be in the image.

### Quickstart (Local)

```bash
pip install uv
uv venv
source .venv/bin/activate
uv sync --all-groups
uv run python src/generator.py --config config/config.yaml
```

---

## Generation

This pipeline generates a dataset of simulated acoustic scenes.

For each of `num_scenes`:

* A room scene is sampled from **Gibson**.
* `events_per_scene` **foreground events** are randomly sampled from [`data/esc50`](data/esc50/).

  * Event durations vary.
  * Event properties are randomly modified (e.g., level and other signal characteristics).

For each scene, we generate `num_mics_per_scene` microphone placements:

* Each placement uses an **Eigenmike32** configuration.
* Microphone positions are sampled randomly for the same scene.
* Recordings are stored separately per placement.

### Augmentation and mixing

* Foreground events may be augmented using **AudibleLight** (randomly sampled from the available augmentation options).
* A **background noise** signal is sampled randomly.
* The **signal-to-noise ratio (SNR)** is sampled randomly.

All parameters can be configured in [`config/config.yaml`](config/config.yaml).

### Metadata format

The metadata follows the **DCASE STARRS23** format [[2]](https://arxiv.org/abs/2306.09126). Class labels are set to `0` for all events, since the downstream application does not require class annotations.

---

## ESC-50: Dataset for Environmental Sound Classification

Foreground audio samples are sourced from the **ESC-50** dataset [[3]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT).

Preprocessing applied in this project:

* Resampled to **24 kHz**
* Converted to **mono**

ESC-50 is distributed under the Creative Commons **Attribution-NonCommercial 3.0** license (see [here](http://creativecommons.org/licenses/by-nc/3.0/)).

---

## References

[[1]]() AudibleLight, “AudibleLight: A controllable, end-to-end API for soundscape synthesis across ray-traced & real-world measured acoustics,” GitHub repository. Accessed: Feb. 10, 2026. [Online]. Available: https://github.com/AudibleLight/AudibleLight

[[2]](https://arxiv.org/abs/2306.09126) K. Shimada, A. Politis, P. Sudarsanam, D. Krause, K. Uchida, S. Adavanne, A. Hakala, Y. Koyama, N. Takahashi, S. Takahashi, T. Virtanen, and Y. Mitsufuji, “STARSS23: An Audio-Visual Dataset of Spatial Recordings of Real Scenes with Spatiotemporal Annotations of Sound Events,” arXiv:2306.09126, 2023 (rev. 2023). Accessed: Feb. 10, 2026. [Online]. Available: https://arxiv.org/abs/2306.09126

[[3]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT) K. J. Piczak, “ESC-50: Dataset for Environmental Sound Classification,” Harvard Dataverse. Accessed: Feb. 10, 2026. [Online]. Available: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT
