# AudibleLight Dataset Generator

<div style="display:flex;justify-content:space-evenly;align-items:center;">
  <img src="assets/img/logo_qmul.jpg" alt="QMUL" height="100">
</div>

## Overview

Generate synthetic acoustic datasets with the [AudibleLight](https://github.com/audiblelight/audiblelight) simulation framework.

### Quickstart (Docker)

After cloning the repository, use the provided helper script to launch Docker with the appropriate directory mounts.
```bash
DOCKER_IMAGE=audiblelight-dataset-generator:local DOCKER_PULL=false ./scripts/docker-infer.sh
```

**Docker notes**
- If no external config is provided, the script uses [`config/config.yaml`](config/config.yaml).
- Outputs default to:
  - `~/Downloads/audiblelight-dataset-generator/output/dataset_<timestamp>/em32_dev/dev-train`
  - `~/Downloads/audiblelight-dataset-generator/output/dataset_<timestamp>/metadata_dev/dev-train`
- Gibson meshes are cached in a Docker volume by default:
  - Volume name: `audiblelight-gibson-cache`
  - Disable cache: `GIBSON_VOLUME=none`
- The container runs in `linux/amd64` mode only.

**Custom config**
```bash
./scripts/docker-infer.sh /path/to/config.yaml
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
