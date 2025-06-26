# Pointcloud Pipeline

A complete Docker-based pipeline for processing 360° videos into point clouds using SLAM and photogrammetry. THIS README WAS WRITTEN BY CHATGPT SO PLS DON'T JUDGE ME, SLACK MSSG ME (RAGHAV) IF ANY BUILD ISSUES.

## Features

- **SLAM Processing**: Extract camera poses and sparse point clouds from 360° videos using Stella VSLAM
- **Bundle Adjustment**: Refine camera positions using Agisoft Metashape
- **Multiple Input Formats**: Supports .mp4 and .insv video files
- **Self-Contained**: Everything needed is included in this repository

## Prerequisites

- **Docker** with GPU support (nvidia-docker)
- **NVIDIA GPU** with CUDA support
- **Linux** (tested on Ubuntu 22.04)
- **~10GB free disk space** (for Docker build)
- **~10-15 minutes** for first-time Docker build

## Quick Start

### Installation & Usage

1. **Clone this repository:**
   ```bash
   git clone <your-repo-url>
   cd pipeline-clean
   ```

2. **Run the pipeline:**
   ```bash
   ./run.sh /path/to/your/video.mp4 /path/to/output/directory
   ```

   Or for a directory of videos:
   ```bash
   ./run.sh /path/to/video/directory /path/to/output/directory
   ```

That's it! The `run.sh` script will automatically:
- **Build the Docker image** (first time only, ~10-15 minutes)
- **Mount your data** and output directories
- **Mount the Agisoft license** (if present, otherwise uses 30-day trial)
- **Run the full pipeline** inside Docker
- Process your video(s) through SLAM
- Perform bundle adjustment with Agisoft
- Generate point clouds and camera poses

### Advanced Usage

**Start from a specific step:**
```bash
# Start from bundle adjustment only (skip SLAM)
./run.sh /path/to/video.mp4 /path/to/output 2
```

**With Agisoft License:**
If you have an Agisoft Metashape license, place it as `personal-agisoft.lic` in the repository root. The `run.sh` script will automatically mount it into Docker. Otherwise, the pipeline will use the 30-day trial.

**What happens behind the scenes:**
The `run.sh` script handles all Docker complexity for you:
- Builds the Docker image (if not already built)
- Properly mounts your video files and output directory
- Automatically mounts the license file (if present)
- Runs the equivalent of this manual Docker command:
  ```bash
  docker run --gpus all --rm -it \
      -v /your/data:/data \
      -v /your/outputs:/outputs \
      -v $(pwd)/personal-agisoft.lic:/root/.agisoft_licenses/metashape.lic:ro \
      pointcloud-optimized:latest \
      /workspace/run_pointcloud_pipeline_docker.sh /data/video.mp4 /outputs
  ```

### Output Files

The pipeline generates:
- **SLAM Results**: Camera trajectories and sparse point clouds (`*_slam/`)
- **Keyframes**: Sharp frames extracted from videos (`sharp_out/`)
- **Bundle Adjustment**: Refined camera poses and sparse clouds (`agi_out/`)
- **Visualizations**: Trajectory plots and analysis files

### Supported Input Formats

- **MP4 videos**: Standard 360° video files
- **INSV videos**: Insta360 camera files (automatically converted)
- **Video directories**: Process multiple videos together

### Manual Docker Commands

**Most users should use `./run.sh` instead** - it handles everything automatically!

If you need to run Docker commands manually (for debugging or customization):

```bash
# Build the image
docker build -f Dockerfile.pointcloud -t pointcloud-optimized:latest .

# Run the pipeline
docker run --gpus all --rm -it \
    -v /path/to/data:/data \
    -v /path/to/outputs:/outputs \
    -v $(pwd)/personal-agisoft.lic:/root/.agisoft_licenses/metashape.lic:ro \
    pointcloud-optimized:latest \
    /workspace/run_pointcloud_pipeline_docker.sh /data/video.mp4 /outputs
```

## Technical Details

### Architecture
- **SLAM Engine**: Stella VSLAM for visual-inertial odometry
- **Bundle Adjustment**: Agisoft Metashape for photogrammetric refinement
- **Dependencies**: OpenCV, Eigen, g2o, Python 3.x
- **Container**: Ubuntu 22.04 with CUDA support

### Configuration
The pipeline uses optimized settings for 360° video processing:
- Equirectangular projection handling
- Frame scaling to 1440x720 for processing
- Automatic keyframe extraction
- GPS support (if GPX files are available)

## Troubleshooting

**GPU Issues:**
- Ensure nvidia-docker is installed: `sudo apt install nvidia-docker2`
- Check GPU access: `docker run --gpus all nvidia/cuda:11.0-base nvidia-smi`

**Memory Issues:**
- The pipeline requires significant RAM (8GB+ recommended)
- For large videos, consider processing shorter segments

**Build Issues:**
- Ensure sufficient disk space (10GB+ for Docker build)
- Check internet connectivity for dependency downloads

**SLAM Issues:**
- If SLAM binary is missing: The Docker build should create `/workspace/stella_pipeline/stella_vslam_examples/build/run_video_slam`
- Check build logs for compilation errors with g2o or OpenCV dependencies
- If `ghc/filesystem.hpp` errors occur: The `3rd/filesystem` submodule should be properly included
- SLAM may fail on very blurry or low-texture videos - try with sharper footage

**Agisoft License Issues:**
- Trial mode: Works for 30 days without license file
- License mounting: Ensure `personal-agisoft.lic` is in the repository root
- "No license found" error: Check file permissions and Docker mounting



