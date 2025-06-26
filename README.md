# Pointcloud Pipeline

A complete Docker-based pipeline for processing 360° videos into point clouds using SLAM and photogrammetry.

## Features

- **SLAM Processing**: Extract camera poses and sparse point clouds from 360° videos using Stella VSLAM
- **Bundle Adjustment**: Refine camera positions using Agisoft Metashape
- **Multiple Input Formats**: Supports .mp4 and .insv video files

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

That's it! The script will automatically:
- Build the Docker image
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
If you have an Agisoft Metashape license, place it as `personal-agisoft.lic` in the repository root. Otherwise, the pipeline will use the 30-day trial.

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

If you prefer to run Docker commands manually:

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

## License

This project includes:
- Stella VSLAM (BSD License)
- Agisoft Metashape (Commercial - requires license)
- OpenCV (Apache 2.0)

See individual component licenses for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with sample data
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker and GPU setup
3. Open an issue with system details and error logs


