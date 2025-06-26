# 3D RECONSTRUCTION PIPELINE


## EXPLANATION

This 3D reconstruction pipeline is orchestrated by the bash script 'run_pointcloud_pipeline_docker.sh'. The bash script is run inside the docker container 'Dockerfile.pointcloud'

## STEPS IN BASH SCRIPT

1) Convert the .mp4 video to 1440x720 resolution.

2) SLAM: (A) Run stella_Vslam, (B) Extract pointcloud data from .msg output file

3) AGISOFT: 

(A) Run select_sharp.py: Select sharp frames in a sliding window around all keyframes, and save the sharp frames in a folder beside SLAM data.

(B) Run run_agisoft.py: Script runs agisoft's bundle adjustment

## HOW TO RUN

1) Clone the repo, and go to the virtual-tour folder.

2) Run: ```docker build -f Dockerfile.pointcloud -t pointcloud-optimized:latest .```

3) [For running the repo:] 

(A) [Directly run:]

After docker is built: ```docker run --gpus all --rm -it -v /DATA_DIRECTORY_CONTAINING_VIDEO:/data -v /OUTPUT_DIRECTORY_FOR_RESULTS:/outputs -v /VIRTUAL_TOUR_LOCATION/personal-agisoft.lic:/root/.agisoft_licenses/metashape.lic:ro pointcloud-optimized:latest /workspace/run_pointcloud_pipeline_docker.sh /data/VIDEO_NAME.mp4 /outputs```

(B) [For running from inside docker:] 

After docker is built: ```docker run --gpus all --rm -it -v /DATA_DIRECTORY_CONTAINING_VIDEO:/data -v /OUTPUT_DIRECTORY_FOR_RESULTS:/outputs -v /VIRTUAL_TOUR_LOCATION/personal-agisoft.lic:/root/.agisoft_licenses/metashape.lic:ro pointcloud-optimized:latest```

Inside virtual docker: ```/workspace/run_pointcloud_pipeline_docker.sh /data/VIDEO_NAME.mp4 /outputs```

## OUTPUT

Output Directory Structure:


