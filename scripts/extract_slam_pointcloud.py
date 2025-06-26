#!/usr/bin/env python3
"""
Convert Stella VSLAM .msg file to viewable point cloud formats
This script extracts 3D landmarks from the SLAM map database and exports them
as PLY, PCD, or other formats for visualization.
"""

import sys
import msgpack
import numpy as np
import argparse
from pathlib import Path
import cv2
import json

def load_msg_file(msg_path):
    """Load and parse the SLAM .msg file"""
    try:
        with open(msg_path, 'rb') as f:
            data = msgpack.unpack(f, raw=False, strict_map_key=False)
        print(f"Successfully loaded {msg_path}")
        return data
    except Exception as e:
        print(f"Error loading {msg_path}: {e}")
        return None

def extract_landmarks(data):
    """Extract 3D landmarks from the SLAM data"""
    landmarks = []
    
    if 'landmarks' in data:
        landmarks_data = data['landmarks']
        print(f"Found {len(landmarks_data)} landmarks in the map")
        
        for landmark_id, landmark_info in landmarks_data.items():
            if 'pos_w' in landmark_info:
                pos = landmark_info['pos_w']
                if len(pos) >= 3:
                    landmarks.append([pos[0], pos[1], pos[2]])
                    
    elif 'map_points' in data:
        map_points = data['map_points']
        print(f"Found {len(map_points)} map points in the map")
        
        for point_id, point_info in map_points.items():
            if 'pos_w' in point_info:
                pos = point_info['pos_w']
                if len(pos) >= 3:
                    landmarks.append([pos[0], pos[1], pos[2]])
    
    return np.array(landmarks)

def extract_keyframes(data):
    """Extract keyframe poses from the SLAM data"""
    poses = []
    
    if 'keyframes' in data:
        keyframes_data = data['keyframes']
        print(f"Found {len(keyframes_data)} keyframes in the map")
        
        # Create list of keyframes with timestamps for proper temporal ordering
        keyframe_list = []
        for kf_id, kf_info in keyframes_data.items():
            # Check for trans_wc (world-to-camera) first, then fallback to trans_cw
            if 'trans_wc' in kf_info:
                trans = kf_info['trans_wc']
                timestamp = kf_info.get('ts', 0)  # Get timestamp, default to 0 if missing
                if len(trans) >= 3:
                    keyframe_list.append((timestamp, trans, kf_id))
            elif 'trans_cw' in kf_info and 'rot_cw' in kf_info:
                trans = kf_info['trans_cw']
                rot = kf_info['rot_cw']
                timestamp = kf_info.get('ts', 0)
                if len(trans) >= 3:
                    # Convert from camera-to-world to world coordinates
                    keyframe_list.append((timestamp, trans, kf_id))
        
        # Sort keyframes by timestamp to ensure proper temporal order
        keyframe_list.sort(key=lambda x: x[0])
        
        # Extract poses in correct temporal order
        for timestamp, trans, kf_id in keyframe_list:
            poses.append([trans[0], trans[1], trans[2]])
    
    return np.array(poses)

def create_trajectory_visualization(poses, output_file):
    """Create a simple trajectory visualization using OpenCV"""
    if len(poses) < 2:
        print("WARNING: Need at least 2 camera poses for trajectory visualization")
        return
        
    points = np.array(poses)
    
    # Create 2D projection (top-down view)
    x_vals = points[:, 0]
    z_vals = points[:, 2]  # Use Z for depth
    
    # Normalize coordinates to image space
    x_min, x_max = x_vals.min(), x_vals.max()
    z_min, z_max = z_vals.min(), z_vals.max()
    
    canvas_size = 800
    margin = 50
    
    if x_max != x_min and z_max != z_min:
        # Create white canvas
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        # Map coordinates to canvas
        for i, (x, z) in enumerate(zip(x_vals, z_vals)):
            x_norm = int((x - x_min) / (x_max - x_min) * (canvas_size - 2*margin)) + margin
            z_norm = int((z - z_min) / (z_max - z_min) * (canvas_size - 2*margin)) + margin
            z_norm = canvas_size - z_norm  # Flip Y axis
            
            # Draw trajectory point
            cv2.circle(canvas, (x_norm, z_norm), 3, (0, 0, 255), -1)
            
            # Draw trajectory line
            if i > 0:
                prev_x = int((x_vals[i-1] - x_min) / (x_max - x_min) * (canvas_size - 2*margin)) + margin
                prev_z = int((z_vals[i-1] - z_min) / (z_max - z_min) * (canvas_size - 2*margin)) + margin
                prev_z = canvas_size - prev_z
                cv2.line(canvas, (prev_x, prev_z), (x_norm, z_norm), (255, 0, 0), 2)
        
        # Add start/end markers
        if len(poses) > 0:
            start_x = int((x_vals[0] - x_min) / (x_max - x_min) * (canvas_size - 2*margin)) + margin
            start_z = canvas_size - (int((z_vals[0] - z_min) / (z_max - z_min) * (canvas_size - 2*margin)) + margin)
            cv2.circle(canvas, (start_x, start_z), 8, (0, 255, 0), -1)  # Green start
            
            end_x = int((x_vals[-1] - x_min) / (x_max - x_min) * (canvas_size - 2*margin)) + margin
            end_z = canvas_size - (int((z_vals[-1] - z_min) / (z_max - z_min) * (canvas_size - 2*margin)) + margin)
            cv2.circle(canvas, (end_x, end_z), 8, (0, 0, 255), -1)  # Red end
        
        # Save visualization
        cv2.imwrite(str(output_file), canvas)
        print(f"Trajectory visualization saved to: {output_file}")
    else:
        print("WARNING: Cannot create trajectory visualization - insufficient coordinate variation")

def create_combined_pointcloud_trajectory_visualization(landmarks, poses, output_file):
    """Create a combined visualization showing both 3D landmarks and camera trajectory"""
    if len(landmarks) == 0 and len(poses) == 0:
        print("WARNING: No landmarks or trajectory data for combined visualization")
        return
    
    # Combine all points to determine overall bounds
    all_points = []
    if len(landmarks) > 0:
        all_points.extend(landmarks.tolist())
    if len(poses) > 0:
        all_points.extend(poses.tolist())
    
    if len(all_points) == 0:
        return
        
    all_points = np.array(all_points)
    
    # Create 2D projection (top-down view) 
    x_vals = all_points[:, 0]
    z_vals = all_points[:, 2]  # Use Z for depth
    
    # Normalize coordinates to image space
    x_min, x_max = x_vals.min(), x_vals.max()
    z_min, z_max = z_vals.min(), z_vals.max()
    
    canvas_size = 1200  # Larger canvas for combined view
    margin = 80
    
    if x_max != x_min and z_max != z_min:
        # Create dark canvas for better contrast
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 20
        
        # Draw landmarks as small dots
        if len(landmarks) > 0:
            print(f"Drawing {len(landmarks)} landmark points...")
            for point in landmarks:
                x, z = point[0], point[2]
                x_norm = int((x - x_min) / (x_max - x_min) * (canvas_size - 2*margin)) + margin
                z_norm = int((z - z_min) / (z_max - z_min) * (canvas_size - 2*margin)) + margin
                z_norm = canvas_size - z_norm  # Flip Y axis
                
                # Draw landmark point in cyan
                cv2.circle(canvas, (x_norm, z_norm), 1, (255, 255, 0), -1)
        
        # Draw camera trajectory
        if len(poses) > 0:
            print(f"Drawing camera trajectory with {len(poses)} poses...")
            traj_points = np.array(poses)
            traj_x = traj_points[:, 0]
            traj_z = traj_points[:, 2]
            
            for i, (x, z) in enumerate(zip(traj_x, traj_z)):
                x_norm = int((x - x_min) / (x_max - x_min) * (canvas_size - 2*margin)) + margin
                z_norm = int((z - z_min) / (z_max - z_min) * (canvas_size - 2*margin)) + margin
                z_norm = canvas_size - z_norm  # Flip Y axis
                
                # Draw trajectory point in red
                cv2.circle(canvas, (x_norm, z_norm), 4, (0, 0, 255), -1)
                
                # Draw trajectory line
                if i > 0:
                    prev_x = int((traj_x[i-1] - x_min) / (x_max - x_min) * (canvas_size - 2*margin)) + margin
                    prev_z = int((traj_z[i-1] - z_min) / (z_max - z_min) * (canvas_size - 2*margin)) + margin
                    prev_z = canvas_size - prev_z
                    cv2.line(canvas, (prev_x, prev_z), (x_norm, z_norm), (255, 0, 0), 2)
            
            # Add start/end markers for camera path
            if len(poses) > 0:
                start_x = int((traj_x[0] - x_min) / (x_max - x_min) * (canvas_size - 2*margin)) + margin
                start_z = canvas_size - (int((traj_z[0] - z_min) / (z_max - z_min) * (canvas_size - 2*margin)) + margin)
                cv2.circle(canvas, (start_x, start_z), 10, (0, 255, 0), -1)  # Green start
                
                end_x = int((traj_x[-1] - x_min) / (x_max - x_min) * (canvas_size - 2*margin)) + margin
                end_z = canvas_size - (int((traj_z[-1] - z_min) / (z_max - z_min) * (canvas_size - 2*margin)) + margin)
                cv2.circle(canvas, (end_x, end_z), 10, (255, 0, 255), -1)  # Magenta end
        
        # Add legend
        legend_y = 30
        cv2.putText(canvas, "SLAM Reconstruction Overview", (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        legend_y += 40
        if len(landmarks) > 0:
            cv2.circle(canvas, (30, legend_y), 2, (255, 255, 0), -1)
            cv2.putText(canvas, f"3D Landmarks ({len(landmarks)} points)", (50, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            legend_y += 30
        if len(poses) > 0:
            cv2.circle(canvas, (30, legend_y), 4, (0, 0, 255), -1)
            cv2.putText(canvas, f"Camera Path ({len(poses)} poses)", (50, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            legend_y += 30
            cv2.circle(canvas, (30, legend_y), 6, (0, 255, 0), -1)
            cv2.putText(canvas, "Start Position", (50, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            legend_y += 30
            cv2.circle(canvas, (30, legend_y), 6, (255, 0, 255), -1)
            cv2.putText(canvas, "End Position", (50, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save visualization
        cv2.imwrite(str(output_file), canvas)
        print(f"Combined pointcloud + trajectory visualization saved to: {output_file}")
    else:
        print("WARNING: Cannot create combined visualization - insufficient coordinate variation")

def save_ply(points, output_path, colors=None):
    """Save points as PLY format"""
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
"""
    
    if colors is not None:
        header += """property uchar red
property uchar green
property uchar blue
"""
    
    header += "end_header\n"
    
    with open(output_path, 'w') as f:
        f.write(header)
        
        for i, point in enumerate(points):
            if colors is not None:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
            else:
                f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"Saved PLY file: {output_path}")

def save_pcd(points, output_path):
    """Save points as PCD format"""
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(points)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(points)}
DATA ascii
"""
    
    with open(output_path, 'w') as f:
        f.write(header)
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"Saved PCD file: {output_path}")

def save_xyz(points, output_path):
    """Save points as simple XYZ format"""
    with open(output_path, 'w') as f:
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    
    print(f"Saved XYZ file: {output_path}")

def print_statistics(data, landmarks, poses):
    """Print statistics about the SLAM data"""
    print("\n" + "="*50)
    print("SLAM MAP STATISTICS")
    print("="*50)
    
    if landmarks is not None and len(landmarks) > 0:
        print(f"   3D Landmarks: {len(landmarks)}")
        print(f"   Min coordinates: [{landmarks.min(axis=0)[0]:.2f}, {landmarks.min(axis=0)[1]:.2f}, {landmarks.min(axis=0)[2]:.2f}]")
        print(f"   Max coordinates: [{landmarks.max(axis=0)[0]:.2f}, {landmarks.max(axis=0)[1]:.2f}, {landmarks.max(axis=0)[2]:.2f}]")
        print(f"   Center: [{landmarks.mean(axis=0)[0]:.2f}, {landmarks.mean(axis=0)[1]:.2f}, {landmarks.mean(axis=0)[2]:.2f}]")
    
    if poses is not None and len(poses) > 0:
        print(f"   Camera Poses: {len(poses)}")
        print(f"   Trajectory length: {len(poses)} keyframes")
    
    print(f"  Other data keys: {list(data.keys())}")
    print("="*50)

def create_summary_report(data, landmarks, poses, output_dir, base_name):
    """Create a summary report of the SLAM extraction"""
    summary_file = output_dir / f"{base_name}_slam_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"[SLAM Extraction Summary]\n")
        f.write(f"Dataset: {base_name}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        f.write(f"SLAM Data Contents:\n")
        f.write(f"- Data Keys: {list(data.keys())}\n")
        f.write(f"- 3D Landmarks: {len(landmarks)} points\n")
        f.write(f"- Camera Poses: {len(poses)} poses\n")
        
        if len(landmarks) > 0:
            f.write(f"\nLandmark Statistics:\n")
            f.write(f"- Min coordinates: [{landmarks.min(axis=0)[0]:.2f}, {landmarks.min(axis=0)[1]:.2f}, {landmarks.min(axis=0)[2]:.2f}]\n")
            f.write(f"- Max coordinates: [{landmarks.max(axis=0)[0]:.2f}, {landmarks.max(axis=0)[1]:.2f}, {landmarks.max(axis=0)[2]:.2f}]\n")
            f.write(f"- Center: [{landmarks.mean(axis=0)[0]:.2f}, {landmarks.mean(axis=0)[1]:.2f}, {landmarks.mean(axis=0)[2]:.2f}]\n")
        
        f.write(f"\nFiles Generated:\n")
        if len(landmarks) > 0:
            f.write(f"- {base_name}_landmarks.ply (PLY point cloud)\n")
            f.write(f"- {base_name}_landmarks.pcd (PCD point cloud)\n")
            f.write(f"- {base_name}_landmarks.xyz (XYZ coordinates)\n")
        if len(poses) > 0:
            f.write(f"- {base_name}_poses.ply (Camera trajectory PLY)\n")
            f.write(f"- {base_name}_poses.pcd (Camera trajectory PCD)\n")
            f.write(f"- {base_name}_poses.xyz (Camera trajectory XYZ)\n")
            f.write(f"- {base_name}_trajectory.png (Trajectory visualization)\n")
        if len(landmarks) > 0 and len(poses) > 0:
            f.write(f"- {base_name}_combined.ply (Combined landmarks + trajectory)\n")
            f.write(f"- {base_name}_combined.pcd (Combined PCD)\n")
            f.write(f"- {base_name}_combined.xyz (Combined XYZ)\n")
        if len(landmarks) > 0 or len(poses) > 0:
            f.write(f"- {base_name}_combined_overview.png (Full SLAM visualization)\n")
        f.write(f"- {base_name}_slam_summary.txt (This summary report)\n")
    
    print(f"Summary report saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert Stella VSLAM .msg file to point cloud formats')
    
    # Support both interfaces: direct .msg file and pipeline approach
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('msg_file', nargs='?', help='Path to the .msg file (direct approach)')
    group.add_argument('--slam-dir', help='Directory containing SLAM outputs (pipeline approach)')
    
    parser.add_argument('-o', '--output', '--output-dir', dest='output', help='Output directory (default: same as input file)')
    parser.add_argument('--format', choices=['ply', 'pcd', 'xyz', 'all'], default='all',
                       help='Output format (default: all)')
    parser.add_argument('--landmarks-only', action='store_true', 
                       help='Export only landmarks (no camera poses)')
    parser.add_argument('--poses-only', action='store_true',
                       help='Export only camera poses (no landmarks)')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip creating PNG visualizations (only generate PLY/PCD/XYZ files)')
    
    args = parser.parse_args()
    
    # Handle pipeline approach (--slam-dir)
    if args.slam_dir:
        slam_dir = Path(args.slam_dir)
        if not slam_dir.exists():
            print(f"ERROR: SLAM directory not found: {slam_dir}")
            return 1
        
        # Look for .msg file in slam_dir
        msg_files = list(slam_dir.glob("*.msg"))
        if not msg_files:
            print(f"ERROR: No .msg file found in SLAM directory: {slam_dir}")
            return 1
        
        msg_file_path = msg_files[0]  # Use first .msg file found
        dataset_name = msg_file_path.stem
        
        # Set output directory for pipeline approach
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = slam_dir / "msg_extraction"
    
    # Handle direct .msg file approach  
    else:
        msg_file_path = Path(args.msg_file)
        if not msg_file_path.exists():
            print(f"ERROR: .msg file not found: {msg_file_path}")
            return 1
        
        dataset_name = msg_file_path.stem
        
        # Set output directory for direct approach
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = msg_file_path.parent
    
    # Load the .msg file
    data = load_msg_file(msg_file_path)
    if data is None:
        return 1
    
    # Extract landmarks and poses
    landmarks = extract_landmarks(data) if not args.poses_only else None
    poses = extract_keyframes(data) if not args.landmarks_only else None
    
    # Print statistics
    print_statistics(data, landmarks, poses)
    
    output_dir.mkdir(exist_ok=True)
    base_name = dataset_name
    
    # Save landmarks
    if landmarks is not None and len(landmarks) > 0:
        print(f"\n  Saving {len(landmarks)} landmarks...")
        
        if args.format in ['ply', 'all']:
            # Landmarks in blue
            colors = np.array([[100, 149, 237]] * len(landmarks))  # Cornflower blue
            save_ply(landmarks, output_dir / f"{base_name}_landmarks.ply", colors)
        
        if args.format in ['pcd', 'all']:
            save_pcd(landmarks, output_dir / f"{base_name}_landmarks.pcd")
        
        if args.format in ['xyz', 'all']:
            save_xyz(landmarks, output_dir / f"{base_name}_landmarks.xyz")
    
    # Save camera poses
    if poses is not None and len(poses) > 0:
        print(f"\n  Saving {len(poses)} camera poses...")
        
        if args.format in ['ply', 'all']:
            # Camera poses in red
            colors = np.array([[255, 69, 0]] * len(poses))  # Red-orange
            save_ply(poses, output_dir / f"{base_name}_poses.ply", colors)
        
        if args.format in ['pcd', 'all']:
            save_pcd(poses, output_dir / f"{base_name}_poses.pcd")
        
        if args.format in ['xyz', 'all']:
            save_xyz(poses, output_dir / f"{base_name}_poses.xyz")
    
    # Save combined point cloud
    if landmarks is not None and poses is not None and len(landmarks) > 0 and len(poses) > 0:
        print(f"\n  Saving combined point cloud...")
        combined_points = np.vstack([landmarks, poses])
        
        if args.format in ['ply', 'all']:
            # Combine colors: blue for landmarks, red for poses
            landmark_colors = np.array([[100, 149, 237]] * len(landmarks))
            pose_colors = np.array([[255, 69, 0]] * len(poses))
            combined_colors = np.vstack([landmark_colors, pose_colors])
            save_ply(combined_points, output_dir / f"{base_name}_combined.ply", combined_colors)
        
        if args.format in ['pcd', 'all']:
            save_pcd(combined_points, output_dir / f"{base_name}_combined.pcd")
        
        if args.format in ['xyz', 'all']:
            save_xyz(combined_points, output_dir / f"{base_name}_combined.xyz")
    
    # Create visualizations (unless disabled)
    if not args.no_visualizations:
        print(f"\n  Creating visualizations...")
        
        # Create trajectory visualization
        if poses is not None and len(poses) > 0:
            create_trajectory_visualization(poses, output_dir / f"{base_name}_trajectory.png")
        
        # Create combined pointcloud + trajectory visualization
        if landmarks is not None or poses is not None:
            create_combined_pointcloud_trajectory_visualization(landmarks, poses, output_dir / f"{base_name}_combined_overview.png")
    
    # Create summary report
    create_summary_report(data, landmarks if landmarks is not None else np.array([]), 
                         poses if poses is not None else np.array([]), output_dir, base_name)
    
    print(f"\nConversion completed! Files saved in: {output_dir}")
    print("\nGenerated Files:")
    print("   • PLY files: Open with MeshLab, CloudCompare, or Blender")
    print("   • PCD files: Open with PCL Viewer or CloudCompare")
    print("   • XYZ files: Open with any text editor or import into MATLAB/Python")
    if not args.no_visualizations:
        print("   • PNG files: Trajectory and combined overview visualizations")
    print(f"   • TXT file: Complete summary report ({base_name}_slam_summary.txt)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 