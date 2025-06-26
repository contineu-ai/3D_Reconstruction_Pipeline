#!/bin/bash

# Pointcloud Pipeline Runner
# This script builds and runs the pointcloud pipeline in Docker

set -e

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <path_to_data_directory_or_video> <path_to_output_directory> [start_step]"
    echo ""
    echo "Arguments:"
    echo "  path_to_data_directory_or_video: Path to directory containing videos or single video file (.mp4 or .insv)"
    echo "  path_to_output_directory: Path where outputs will be saved"
    echo "  start_step: Optional - 1 for SLAM (default), 2 for Bundle Adjustment only"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/videos /path/to/outputs"
    echo "  $0 /path/to/video.mp4 /path/to/outputs"
    echo "  $0 /path/to/video.mp4 /path/to/outputs 2"
    exit 1
fi

VIDEO_PATH="$1"
OUTPUT_PATH="$2"
START_STEP="${3:-1}"

# Check if paths exist
if [ ! -e "$VIDEO_PATH" ]; then
    echo "Error: Video path '$VIDEO_PATH' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Check if license file exists
if [ ! -f "personal-agisoft.lic" ]; then
    echo "Warning: personal-agisoft.lic not found. Agisoft will run in trial mode (30 days)."
    echo "To use a full license, place your personal-agisoft.lic file in this directory."
    LICENSE_MOUNT=""
else
    LICENSE_MOUNT="-v $(pwd)/personal-agisoft.lic:/root/.agisoft_licenses/metashape.lic:ro"
fi

echo "Building Docker image..."
docker build -f Dockerfile.pointcloud -t pointcloud-optimized:latest .

echo ""
echo "Running pointcloud pipeline..."
echo "Video/Data path: $VIDEO_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Start step: $START_STEP"
echo ""

# Determine mount strategy based on whether it's a file or directory
if [ -f "$VIDEO_PATH" ]; then
    # Single file - mount the parent directory and pass the filename
    VIDEO_DIR=$(dirname "$VIDEO_PATH")
    VIDEO_FILE=$(basename "$VIDEO_PATH")
    DOCKER_VIDEO_PATH="/data/$VIDEO_FILE"
    VIDEO_MOUNT="-v $VIDEO_DIR:/data"
else
    # Directory - mount the entire directory
    DOCKER_VIDEO_PATH="/data"
    VIDEO_MOUNT="-v $VIDEO_PATH:/data"
fi

# Run the Docker container
docker run --gpus all --rm -it \
    $VIDEO_MOUNT \
    -v "$OUTPUT_PATH":/outputs \
    $LICENSE_MOUNT \
    pointcloud-optimized:latest \
    /workspace/run_pointcloud_pipeline_docker.sh $DOCKER_VIDEO_PATH /outputs $START_STEP

echo ""
echo "Pipeline completed! Check your outputs in: $OUTPUT_PATH" 