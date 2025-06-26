#!/bin/bash
set -e

# LICENSE MOUNTING: To mount Agisoft license, use this Docker run command:
# docker run --gpus all --rm -it \
#     -v /path/to/your/data:/data \
#     -v /path/to/your/outputs:/outputs \
#     -v $(pwd)/personal-agisoft.lic:/root/.agisoft_licenses/metashape.lic:ro \
#     pointcloud-optimized:latest /workspace/run_pointcloud_pipeline_docker.sh /data/your_video.mp4 /outputs

VideoDirectoryPath=$1
OutputDirectory=$2
# Get the start step number from command-line argument (1=SLAM, 2=AGISOFT, etc.)
start_step=${3:-1}

CONFIG_FILE=/workspace/configs/settings.json
DATASET_FILE=/workspace/configs/dataset.json

# Create minimal config files if they don't exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating minimal config file..."
    mkdir -p /workspace/configs
    cat <<EOF > "$CONFIG_FILE"
{
    "dataset_type": "wtp",
    "segmentation": false,
    "bim": false,
    "bim_obj": false,
    "is_360": true,
    "poisson": false,
    "showcase": false,
    "debug": true,
    "gps": false,
    "report": false
}
EOF
fi

if [ ! -f "$DATASET_FILE" ]; then
    echo "Creating minimal dataset file..."
    cat <<EOF > "$DATASET_FILE"
{
    "outputs_selected": [
        {"type": "POINTCLOUD"}
    ]
}
EOF
fi

# Parse the JSON file for settings
dataset_type=$(jq -r '.dataset_type' $CONFIG_FILE)
segmentation=$(jq -r '.segmentation' $CONFIG_FILE)
bim=$(jq -r '.bim' $CONFIG_FILE)
bim_obj=$(jq -r '.bim_obj' $CONFIG_FILE)
is_360=$(jq -r '.is_360' $CONFIG_FILE)
poisson=$(jq -r '.poisson' $CONFIG_FILE)
showcase=$(jq -r '.showcase' $CONFIG_FILE)
debug=$(jq -r '.debug' $CONFIG_FILE)
gps=$(jq -r '.gps' $CONFIG_FILE)
report=$(jq -r '.report' $CONFIG_FILE)

# Check if outputs are selected using jq
POINTCLOUD=$(jq '.outputs_selected[] | select(.type == "POINTCLOUD")' "$DATASET_FILE")

# Helper functions
get_frame_count() {
    ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "$1"
}

convert_insv_to_mp4() {
    local input_file="$1"
    local output_file="$2"
    
    echo "Converting .insv file to .mp4: $input_file -> $output_file"
    ffmpeg -y -i "$input_file" -vf scale=1440:720 -c:v libx264 -preset fast -crf 23 "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "Conversion successful: $output_file"
        return 0
    else
        echo "Conversion failed for: $input_file"
        return 1
    fi
}

# Getting the current time in the format HHMMSS
timestamp=$(date +"%d-%m-%Y-%H-%M")

mkdir -p $OutputDirectory
mkdir -p $OutputDirectory/sharp_out
mkdir -p $OutputDirectory/agi_out

# Process input video files
sorted_files=()

if [ -f "$VideoDirectoryPath" ]; then
    if [[ "$VideoDirectoryPath" == *.mp4 ]]; then
        echo "$VideoDirectoryPath is a single .mp4 file."
        sorted_files=("$VideoDirectoryPath")
    elif [[ "$VideoDirectoryPath" == *.insv ]]; then
        echo "$VideoDirectoryPath is a single .insv file. Converting to .mp4..."
        temp_converted_dir="$OutputDirectory/converted_videos"
        mkdir -p "$temp_converted_dir"
        
        filename=$(basename "$VideoDirectoryPath")
        converted_file="$temp_converted_dir/${filename%.*}.mp4"
        
        if convert_insv_to_mp4 "$VideoDirectoryPath" "$converted_file"; then
            sorted_files=("$converted_file")
        else
            echo "Error: Failed to convert .insv file."
            exit 1
        fi
    else
        echo "Error: $VideoDirectoryPath is not a supported video file (.mp4 or .insv)."
        exit 1
    fi
elif [ -d "$VideoDirectoryPath" ]; then
    echo "Processing directory: $VideoDirectoryPath"
    
    mp4_files=($(find "$VideoDirectoryPath" -maxdepth 1 -name "*.mp4" -type f))
    insv_files=($(find "$VideoDirectoryPath" -maxdepth 1 -name "*.insv" -type f))
    
    if [ ${#insv_files[@]} -gt 0 ]; then
        temp_converted_dir="$OutputDirectory/converted_videos"
        mkdir -p "$temp_converted_dir"
        
        echo "Converting ${#insv_files[@]} .insv file(s) to .mp4..."
        for insv_file in "${insv_files[@]}"; do
            filename=$(basename "$insv_file")
            converted_file="$temp_converted_dir/${filename%.*}.mp4"
            
            if convert_insv_to_mp4 "$insv_file" "$converted_file"; then
                mp4_files+=("$converted_file")
            else
                echo "Warning: Failed to convert $insv_file, skipping..."
            fi
        done
    fi
    
    if [ ${#mp4_files[@]} -eq 0 ]; then
        echo "Error: No .mp4 or .insv files found in the directory."
        exit 1
    fi
    
    echo "Sorting ${#mp4_files[@]} video file(s) by frame count..."
    sorted_files=($(for file in "${mp4_files[@]}"; do
        if [ -f "$file" ]; then
            frame_count=$(get_frame_count "$file")
            echo "$frame_count $file"
        fi
    done | sort -rn | awk '{print $2}'))
else
    echo "Error: Please provide path to a directory containing videos or a single video file."
    exit 1
fi

echo "Found ${#sorted_files[@]} video file(s) to process"

# SLAM function
slam() {
    echo "Starting SLAM processing..."
    
    for VideoPath in "${sorted_files[@]}"; do
        filename=$(basename "$VideoPath")
        echo "Processing video: $filename"

        output_msg="${filename%.*}_${timestamp}.msg"
        output_scaled_video="${filename%.*}_scaled.mp4"
        output_eval_dir="${filename%.*}_slam"
        
        mkdir -p $OutputDirectory/$output_eval_dir

        echo "[SCALING VIDEO] -> 1440x720..."
        time ffmpeg -y -i "$VideoPath" -vf scale="1440:720" "$OutputDirectory/$output_eval_dir/$output_scaled_video"

        echo "[RUNNING STELLA VSLAM]..."
        /workspace/stella_pipeline/stella_vslam_examples/build/run_video_slam \
            -v /workspace/stella_pipeline/orb_vocab.fbow \
            -m "$OutputDirectory/$output_eval_dir/$output_scaled_video" \
            -c /workspace/stella_pipeline/configs/wtp/equirectangular_wtp.yaml \
            --frame-skip 1 \
            --no-sleep \
            --map-db-out "$OutputDirectory/$output_eval_dir/$output_msg" \
            --eval-log-dir "$OutputDirectory/$output_eval_dir" \
            --auto-term > "$OutputDirectory/$output_eval_dir/output_log.txt"

        echo "SLAM processing complete for $filename"
    done
    
    echo "SLAM complete for all videos"
    
    # Extract PLY pointclouds and trajectories from SLAM data for immediate analysis
    echo "EXTRACTING PLY FILES FROM SLAM DATA FOR ANALYSIS..."
    for VideoPath in "${sorted_files[@]}"; do
        filename=$(basename "$VideoPath")
        output_eval_dir="${filename%.*}_slam"
        
        echo "Extracting PLY files from SLAM data: $output_eval_dir"
        
        # Create dedicated directory for .msg extraction results
        msg_extraction_dir="$OutputDirectory/$output_eval_dir/msg_extraction"
        mkdir -p "$msg_extraction_dir"
        
        # Extract PLY files using the fixed script
        python3 /workspace/scripts/extract_slam_pointcloud.py \
            --slam-dir "$OutputDirectory/$output_eval_dir" \
            --output-dir "$msg_extraction_dir"
        
        if [ $? -eq 0 ]; then
            echo "PLY extraction complete for $filename"
            echo "PLY files available at: $msg_extraction_dir"
            echo "   - Landmarks: ${filename%.*}_${timestamp}_landmarks.ply"
            echo "   - Trajectory: ${filename%.*}_${timestamp}_trajectory.ply"
            echo "   - Trajectory only: ${filename%.*}_${timestamp}_trajectory_visualization.png"
            echo "   - Combined view: ${filename%.*}_${timestamp}_combined_overview.png"
        else
            echo "PLY extraction failed for $filename"
        fi
    done
    
    echo "PLY extraction complete for all videos"
}

# Bundle Adjustment function (using Agisoft - same as full pipeline)
agisoft() {
    echo "[BUNDLE ADJUSTING] w/ Agisoft..."
    
    num_files=${#sorted_files[@]}
    
    # Extract keyframes from each video using select_sharp.py
    for ((i = 0; i < num_files; i++)); do
        video_path="${sorted_files[i]}"
        filename=$(basename "$video_path")
        eval_dir="${filename%.*}_slam"
        map_db="${filename%.*}_${timestamp}.msg"
        let "vid_id=$i+1"
        vid_id=$(printf "%03d" "$vid_id")
        vid_tag="${vid_id}_${vid_id}"

        echo "Extracting keyframes from $video_path tagged with $vid_tag"
        
        if [ "$gps" == "true" ]; then
            echo "GPS enabled - looking for GPX file"
            dirpath=$(dirname "$video_path")
            gpx_file="$dirpath/${filename%.*}.gpx"
            
            if [ -f "$gpx_file" ]; then
                python3 /workspace/scripts/select_sharp.py \
                    --input "$video_path" \
                    --output $OutputDirectory/sharp_out/ \
                    --tag $vid_tag \
                    --frame_traj $OutputDirectory/$eval_dir/frame_trajectory.txt \
                    --key_traj $OutputDirectory/$eval_dir/keyframe_trajectory.txt \
                    --msg_file $OutputDirectory/$eval_dir/ \
                    --gpx "$gpx_file" \
                    --frame_skip 1
            else
                echo "Warning: GPX file not found at $gpx_file, proceeding without GPS"
                python3 /workspace/scripts/select_sharp.py \
                    --input "$video_path" \
                    --output $OutputDirectory/sharp_out/ \
                    --tag $vid_tag \
                    --frame_traj $OutputDirectory/$eval_dir/frame_trajectory.txt \
                    --key_traj $OutputDirectory/$eval_dir/keyframe_trajectory.txt \
                    --msg_file $OutputDirectory/$eval_dir/ \
                    --frame_skip 1
            fi
        else
            echo "GPS disabled"
            python3 /workspace/scripts/select_sharp.py \
                --input "$video_path" \
                --output $OutputDirectory/sharp_out/ \
                --tag $vid_tag \
                --frame_traj $OutputDirectory/$eval_dir/frame_trajectory.txt \
                --key_traj $OutputDirectory/$eval_dir/keyframe_trajectory.txt \
                --msg_file $OutputDirectory/$eval_dir/ \
                --frame_skip 1
        fi
    done
    
    echo "Keyframe extraction complete"
    
    # Create properly named JSON files for Agisoft
    echo "Creating properly named JSON files for Agisoft..."
    for ((i = 0; i < num_files; i++)); do
        let "vid_id=$i+1"
        vid_id=$(printf "%03d" "$vid_id")
        
        # Copy video.json to the expected 001_001.json format
        if [ -f "$OutputDirectory/sharp_out/video.json" ]; then
            cp "$OutputDirectory/sharp_out/video.json" "$OutputDirectory/sharp_out/pairs/${vid_id}_${vid_id}.json"
            echo "Created $OutputDirectory/sharp_out/pairs/${vid_id}_${vid_id}.json"
        fi
    done
    
    # # Run image relocation only if there are multiple videos
    # if [ "$num_files" -ne 1 ]; then
    #     echo "Running hloc on sharp images for multi-video relocation"
    #     python3 /workspace/scripts/reloc.py \
    #             --image_dir $OutputDirectory/sharp_out/ \
    #             --hloc_dir $OutputDirectory/hloc_out/ \
    #             --json_dir $OutputDirectory/sharp_out/pairs/ \
    #             --match_thresh 0.45 \
    #             --min_pairs 30
    # fi
    
    # Run Agisoft bundle adjustment - same as full pipeline
    echo "Running Agisoft bundle adjustment..."
    cd /workspace/
    
    # Convert gps boolean for Python script
    if [ "$gps" == "true" ]; then
        gps_flag=1
    else
        gps_flag=0
    fi
    
    # Run Agisoft bundle adjustment
    python3 /workspace/agility_360/run_agisoft.py \
        --images_dir $OutputDirectory/sharp_out/ \
        --output_dir $OutputDirectory/agi_out \
        --vidcount $num_files \
        --gps $gps_flag \
        --stills 0

    # Check if bundle adjustment was successful
    if [ -f "$OutputDirectory/agi_out/project.psx" ]; then
        echo "Bundle adjustment complete!"
        echo "Results available in:"
        echo "- Agisoft project: $OutputDirectory/agi_out/project.psx"
        echo "- Camera poses: $OutputDirectory/agi_out/cams.xml"
        
        # Generate sparse point cloud from Agisoft project if possible
        if [ -f "$OutputDirectory/agi_out/cams.xml" ]; then
            echo "- Camera calibration: $OutputDirectory/agi_out/cams.xml"
        fi
        
        # Check if sparse cloud was generated
        if [ -f "$OutputDirectory/agi_out/sparse_cloud.ply" ]; then
            echo "- Sparse point cloud: $OutputDirectory/agi_out/sparse_cloud.ply"
        fi
    else
        echo "Warning: Bundle adjustment may have failed - no project file generated"
        echo "Check image quality and feature matching"
    fi
}

# Main execution based on start step
case $start_step in
1)
    slam
    ;& # Fallthrough to the next case
2)
    agisoft
    ;;
*)
    echo "Invalid start step. Please enter 1 (SLAM) or 2 (BUNDLE_ADJUSTMENT)."
    exit 1
    ;;
esac

echo "Pointcloud pipeline completed successfully!"
echo "Output directory: $OutputDirectory"
echo ""
echo "Key outputs:"
echo "- SLAM results: $OutputDirectory/*_slam/"
echo "- MSG extractions: $OutputDirectory/*_slam/msg_extraction/"
echo "- Keyframes: $OutputDirectory/sharp_out/"
echo "- Bundle adjustment: $OutputDirectory/agi_out/"
if [ -f "$OutputDirectory/agi_out/sparse_cloud.ply" ]; then
    echo "- Point cloud: $OutputDirectory/agi_out/sparse_cloud.ply"
fi
echo ""
echo "IMMEDIATE ANALYSIS FILES (from .msg):"
echo "- SLAM landmarks: $OutputDirectory/*_slam/msg_extraction/*_landmarks.ply"
echo "- SLAM trajectory: $OutputDirectory/*_slam/msg_extraction/*_trajectory.ply"
echo "- Trajectory only: $OutputDirectory/*_slam/msg_extraction/*_trajectory_visualization.png"
echo "- Combined overview: $OutputDirectory/*_slam/msg_extraction/*_combined_overview.png"
echo "- SLAM summary: $OutputDirectory/*_slam/msg_extraction/*_slam_summary.txt"
echo "Pipeline completed successfully!" 