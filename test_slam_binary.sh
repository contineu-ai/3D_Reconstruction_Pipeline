#!/bin/bash

echo "Testing SLAM binary in Docker container..."

# Test if binary exists and is executable
docker run --rm pointcloud-optimized:latest /bin/bash -c "
echo 'Checking stella_vslam_examples build directory:'
ls -la /workspace/stella_vslam_examples/build/ 2>/dev/null || echo 'Build directory does not exist'

echo ''
echo 'Searching for run_video_slam binary:'
find /workspace -name 'run_video_slam' -type f 2>/dev/null || echo 'Binary not found anywhere'

echo ''
echo 'Testing if binary is executable:'
if [ -f '/workspace/stella_vslam_examples/build/run_video_slam' ]; then
    /workspace/stella_vslam_examples/build/run_video_slam --help 2>/dev/null || echo 'Binary exists but may have issues'
else
    echo 'Binary does not exist at expected location'
fi
"

echo "Test complete!"
