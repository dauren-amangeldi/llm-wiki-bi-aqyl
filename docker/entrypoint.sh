#!/bin/bash
set -e

# Ensure data subdirectories exist when the volume is freshly mounted.
# Runs as root (Variant A, MVP mode) so it always succeeds regardless of
# volume ownership. TODO: switch to an init-container pattern for production.
mkdir -p /app/data/raw /app/data/wiki /app/data/chroma

exec "$@"
