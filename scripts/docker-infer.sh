#!/usr/bin/env bash
# =============================================================================
# docker-infer.sh - Wrapper script to run AudibleLight Dataset Generator in Docker
# =============================================================================

set -euo pipefail

# Default image name matches what's published to GHCR.
IMAGE_NAME="${DOCKER_IMAGE:-ghcr.io/philippxxy/audiblelight-dataset-generator:latest}"

# Only support x86_64 Linux.
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
if [[ "$DOCKER_PLATFORM" != "linux/amd64" ]]; then
    echo "[ERROR] Only linux/amd64 is supported."
    exit 1
fi

# Default host paths under Downloads.
DEFAULT_DOWNLOAD_DIR="${HOME}/Downloads"
if [[ ! -d "$DEFAULT_DOWNLOAD_DIR" ]]; then
    DEFAULT_DOWNLOAD_DIR="${HOME}"
fi
DEFAULT_ROOT="${DEFAULT_DOWNLOAD_DIR}/audiblelight-dataset-generator"
DEFAULT_TIMESTAMP="$(date +"%Y-%m-%d-%H-%M-%S")"
DEFAULT_AUDIO_OUT="${DEFAULT_ROOT}/output/dataset_${DEFAULT_TIMESTAMP}/em32_dev/dev-train"
DEFAULT_META_OUT="${DEFAULT_ROOT}/output/dataset_${DEFAULT_TIMESTAMP}/metadata_dev/dev-train"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 [config_file]"
    echo ""
    echo "Arguments:"
    echo "  config_file       Optional path to your config YAML file"
    echo ""
    echo "Environment variables:"
    echo "  DOCKER_IMAGE      Docker image to use (default: ${IMAGE_NAME})"
    echo "  DOCKER_PULL       Auto-pull image before run (default: true)"
    echo "  DOCKER_PLATFORM   Must be linux/amd64"
    echo "  GIBSON_VOLUME     Docker volume for Gibson cache (default: audiblelight-gibson-cache)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 /home/user/my_config.yaml"
    echo ""
    echo "Your config file should contain (under paths:):"
    echo "  - audio_out"
    echo "  - meta_out"
    echo ""
    echo "If audio_out or meta_out are missing, defaults under:"
    echo "  ${DEFAULT_ROOT}/output/dataset_<timestamp>/..."
    exit 1
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

resolve_path() {
    local path_value="$1"
    local base_dir="$2"
    if [[ "$path_value" == "~"* ]]; then
        path_value="${path_value/#\~/$HOME}"
    fi
    if [[ "$path_value" = /* ]]; then
        echo "$path_value"
    else
        echo "${base_dir}/${path_value}"
    fi
}

# Parse a nested YAML key under a section (supports indentation and optional quoted values).
extract_yaml_section_value() {
    local section="$1"
    local key="$2"
    local file="$3"
    awk -v section="$section" -v key="$key" '
      function indent_len(s) { match(s, /^[[:space:]]*/); return RLENGTH }
      $0 ~ "^[[:space:]]*" section ":[[:space:]]*$" {
        in_section = 1
        section_indent = indent_len($0)
        next
      }
      in_section {
        current_indent = indent_len($0)
        if (current_indent <= section_indent) {
          in_section = 0
        } else if ($0 ~ "^[[:space:]]*" key ":[[:space:]]*") {
          line = $0
          sub(/^[^:]+:[[:space:]]*/, "", line)
          sub(/[[:space:]]+#.*/, "", line)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
          gsub(/^"|"$/, "", line)
          print line
          exit
        }
      }
    ' "$file"
}

write_docker_config() {
    local src="$1"
    local dst="$2"
    local audio="${3:-}"
    local meta="${4:-}"
    if [[ -z "$audio" || -z "$meta" ]]; then
        log_error "Internal error: missing audio/meta output paths."
        exit 1
    fi
    awk -v audio="$audio" -v meta="$meta" '
      function indent_len(s) { match(s, /^[[:space:]]*/); return RLENGTH }
      function indent_str(n) { return sprintf("%*s", n, "") }
      BEGIN {
        in_paths = 0
        paths_seen = 0
        audio_found = 0
        meta_found = 0
      }
      $0 ~ /^[[:space:]]*paths:[[:space:]]*$/ {
        in_paths = 1
        paths_seen = 1
        paths_indent = indent_len($0)
        print
        next
      }
      in_paths {
        current_indent = indent_len($0)
        if (current_indent <= paths_indent) {
          indent = indent_str(paths_indent + 2)
          if (!audio_found) print indent "audio_out: \"" audio "\""
          if (!meta_found) print indent "meta_out: \"" meta "\""
          in_paths = 0
        } else if ($0 ~ /^[[:space:]]*audio_out:[[:space:]]*/) {
          print indent_str(indent_len($0)) "audio_out: \"" audio "\""
          audio_found = 1
          next
        } else if ($0 ~ /^[[:space:]]*meta_out:[[:space:]]*/) {
          print indent_str(indent_len($0)) "meta_out: \"" meta "\""
          meta_found = 1
          next
        }
      }
      { print }
      END {
        if (in_paths) {
          indent = indent_str(paths_indent + 2)
          if (!audio_found) print indent "audio_out: \"" audio "\""
          if (!meta_found) print indent "meta_out: \"" meta "\""
        }
        if (!paths_seen) {
          print "paths:"
          print "  audio_out: \"" audio "\""
          print "  meta_out: \"" meta "\""
        }
      }
    ' "$src" > "$dst"
}

if [[ $# -gt 1 ]]; then
    log_error "This wrapper only accepts an optional config file."
    usage
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -eq 1 ]]; then
    CONFIG_FILE="$1"
else
    CONFIG_FILE="${SCRIPT_DIR}/../config/config.yaml"
    log_info "No config provided; using bundled config: $CONFIG_FILE"
fi

# Validate config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Get absolute path of config file
CONFIG_FILE=$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")
CONFIG_DIR=$(cd "$(dirname "$CONFIG_FILE")" && pwd)
log_info "Using config file: $CONFIG_FILE"

# Parse host paths from config file
AUDIO_OUTPUT_PATH=$(extract_yaml_section_value "paths" "audio_out" "$CONFIG_FILE")
META_OUTPUT_PATH=$(extract_yaml_section_value "paths" "meta_out" "$CONFIG_FILE")

# Validate extracted paths
if [[ -z "$AUDIO_OUTPUT_PATH" ]]; then
    log_warn "Could not extract 'paths.audio_out' from config, using default under Downloads."
    AUDIO_OUTPUT_PATH="$DEFAULT_AUDIO_OUT"
fi
if [[ -z "$META_OUTPUT_PATH" ]]; then
    log_warn "Could not extract 'paths.meta_out' from config, using default under Downloads."
    META_OUTPUT_PATH="$DEFAULT_META_OUT"
fi

# Convert to absolute paths if relative.
AUDIO_OUTPUT_PATH=$(resolve_path "$AUDIO_OUTPUT_PATH" "$CONFIG_DIR")
META_OUTPUT_PATH=$(resolve_path "$META_OUTPUT_PATH" "$CONFIG_DIR")

# Create output directories.
mkdir -p "$AUDIO_OUTPUT_PATH" "$META_OUTPUT_PATH"

log_info "Mounting volumes:"
log_info "  Audio Out:  $AUDIO_OUTPUT_PATH"
log_info "  Meta Out:   $META_OUTPUT_PATH"
log_info "  Config:     $CONFIG_FILE -> /config/config.yaml"

# Create a modified config for Docker (with container paths).
DOCKER_CONFIG=$(mktemp)
trap 'rm -f "$DOCKER_CONFIG"' EXIT

# Replace path keys in config for container runtime.
write_docker_config "$CONFIG_FILE" "$DOCKER_CONFIG" \
    "$AUDIO_OUTPUT_PATH" "$META_OUTPUT_PATH"

log_info "Running Docker container..."
log_info "Image: $IMAGE_NAME"

# Auto-pull image unless explicitly disabled.
DOCKER_PULL="${DOCKER_PULL:-true}"
if [[ "$DOCKER_PULL" == "true" ]]; then
    log_info "Pulling latest image from registry..."
    if ! docker pull --platform "$DOCKER_PLATFORM" "$IMAGE_NAME"; then
        log_error "Failed to pull image '$IMAGE_NAME'"
        log_error ""
        log_error "For private repositories, authenticate first:"
        log_error "  docker login ghcr.io"
        log_error "  echo \$GITHUB_TOKEN | docker login ghcr.io --username YOUR_USERNAME --password-stdin"
        log_error ""
        log_error "To skip automatic pull, set:"
        log_error "  DOCKER_PULL=false $0 $CONFIG_FILE"
        exit 1
    fi
elif ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    log_error "Docker image '$IMAGE_NAME' not found locally and DOCKER_PULL is disabled."
    log_error "Either run: docker pull $IMAGE_NAME"
    exit 1
fi

DOCKER_RUN_ARGS=(--rm --platform "$DOCKER_PLATFORM" -it)
GIBSON_VOLUME="${GIBSON_VOLUME:-audiblelight-gibson-cache}"
if [[ -n "$GIBSON_VOLUME" && "$GIBSON_VOLUME" != "none" ]]; then
    docker volume create "$GIBSON_VOLUME" >/dev/null
    DOCKER_RUN_ARGS+=(-v "${GIBSON_VOLUME}:/app/data/gibson")
    log_info "Gibson cache volume: ${GIBSON_VOLUME} -> /app/data/gibson"
else
    log_warn "Gibson cache volume disabled; Gibson will re-download each run."
fi

docker run "${DOCKER_RUN_ARGS[@]}" \
    -v "$AUDIO_OUTPUT_PATH:$AUDIO_OUTPUT_PATH" \
    -v "$META_OUTPUT_PATH:$META_OUTPUT_PATH" \
    -v "$DOCKER_CONFIG:/config/config.yaml:ro" \
    "$IMAGE_NAME" \
    --config /config/config.yaml

log_info "Inference completed!"
log_info "Audio results saved to: $AUDIO_OUTPUT_PATH"
log_info "Metadata results saved to: $META_OUTPUT_PATH"
