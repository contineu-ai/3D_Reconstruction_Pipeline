import cv2
import gpxpy
import pandas as pd
from geopy.distance import geodesic
import bisect
import numpy as np
import argparse
from tqdm import tqdm 
import torch
import time
import msgpack
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import os
import multiprocessing as mp
import json
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

class UltraOptimizedSharpSelector:
    def __init__(
            self, 
            frame_traj_path: Path, 
            keyframe_traj_path: Path, 
            video_path: Path,
            gpx_path: Path = None,
            start_frame: int = 0,
            end_frame: int = None,
            frame_skip: int = 1,
        ):
        """
        Ultra-optimized Sharp Selector with proper GPU acceleration and intelligent caching
        """
        self.frame_trajectory = np.loadtxt(str(frame_traj_path))
        self.keyframe_trajectory = np.loadtxt(str(keyframe_traj_path))
        self.video_path = video_path
        self.num_frames = self._get_video_frame_count()
        self.fps = self._get_video_fps()
        self.start_frame = start_frame
        if end_frame is None:
            self.end_frame = self.num_frames
        else:
            self.end_frame = end_frame
        if gpx_path is None:
            self.gps_enabled = False
            self.gps_data = None
        else:
            self.gps_enabled = True
            self.gps_data = self.load_gpx(str(gpx_path))
        self.frame_skip = frame_skip

        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize GPU kernels once
        if torch.cuda.is_available():
            self._init_gpu_kernels()
        
        print(f"Using device: {self.device}")
        
        # Fixed keyframe detection
        self.keyframes = self.get_keyframes_fixed()
        
        # Pre-compute keyframe windows for parallel processing
        self.keyframe_windows = self._precompute_windows()
        
        # Smart caching analysis
        self._analyze_frame_clusters()
        
        print(f"Video: {self.num_frames} frames, {self.fps:.2f} FPS")
        print(f"Trajectories: {len(self.frame_trajectory)} frames, {len(self.keyframe_trajectory)} keyframes")
        print(f"Detected keyframes: {len(self.keyframes)}")
        if len(self.keyframes) > 0:
            print(f"Keyframe range: {self.keyframes[0]} to {self.keyframes[-1]}")

    def _init_gpu_kernels(self):
        """Pre-initialize GPU kernels for faster processing"""
        try:
            # Pre-compile GPU kernels
            self.blur_kernel = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False).to(self.device)
            self.blur_kernel.weight.data = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]).float().to(self.device) / 16.0
            
            self.laplace_kernel = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False).to(self.device)
            self.laplace_kernel.weight.data = torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]]).float().to(self.device)
            
            # Warm up GPU
            dummy_input = torch.randn(1, 1, 700, 1400, device=self.device)
            with torch.no_grad():
                _ = self.laplace_kernel(self.blur_kernel(dummy_input))
            
            print("GPU kernels initialized and warmed up")
        except Exception as e:
            print(f"WARNING: GPU kernel initialization failed: {e}")

    def _analyze_frame_clusters(self):
        """Analyze frame clusters to optimize reading patterns"""
        # Group nearby keyframes to optimize video reading
        self.frame_clusters = []
        current_cluster = []
        
        for i, frame_idx in enumerate(self.keyframes):
            window_start, window_end, _ = self.keyframe_windows[i]
            
            if not current_cluster:
                current_cluster = [(i, frame_idx, window_start, window_end)]
            else:
                # Check if this keyframe's window overlaps with current cluster
                last_window_end = current_cluster[-1][3]
                if window_start <= last_window_end + 50:  # 50 frame gap tolerance
                    current_cluster.append((i, frame_idx, window_start, window_end))
                else:
                    # Finalize current cluster
                    self.frame_clusters.append(current_cluster)
                    current_cluster = [(i, frame_idx, window_start, window_end)]
        
        if current_cluster:
            self.frame_clusters.append(current_cluster)
        
        print(f"Optimized into {len(self.frame_clusters)} frame clusters")

    def _get_video_frame_count(self):
        video = cv2.VideoCapture(str(self.video_path))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return frame_count
    
    def _get_video_fps(self):
        video = cv2.VideoCapture(str(self.video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps

    def get_keyframes_fixed(self):
        """Vectorized keyframe detection using numpy broadcasting"""
        keyframe_timestamps = self.keyframe_trajectory[:, 0]
        frame_timestamps = self.frame_trajectory[:, 0]
        
        # Vectorized closest frame finding
        diff_matrix = np.abs(frame_timestamps[:, np.newaxis] - keyframe_timestamps)
        closest_indices = np.argmin(diff_matrix, axis=0)
        
        keyframes = (closest_indices * self.frame_skip).tolist()
        return sorted(keyframes)

    def _precompute_windows(self, window=5):
        """Pre-compute window bounds for all keyframes"""
        windows = []
        for i, keyframe_pos in enumerate(self.keyframes):
            lb = max(keyframe_pos - window, 0)
            ub = min(keyframe_pos + window + 1, self.num_frames)
            
            # Adjust for adjacent keyframes
            if i > 0:
                prev_keyframe = self.keyframes[i - 1]
                lb = max(lb, prev_keyframe + window + 1)
            if i < len(self.keyframes) - 1:
                next_keyframe = self.keyframes[i + 1]
                ub = min(ub, next_keyframe - window)
            
            # Ensure at least one frame
            if lb >= ub:
                lb = keyframe_pos
                ub = keyframe_pos + 1
                
            windows.append((lb, ub, list(range(lb, ub))))
        
        return windows

    def load_gpx(self, gpx_path):
        """Optimized GPX loading with vectorized operations"""
        with open(gpx_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
        
        data = []
        start_time = None
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    if start_time is None:
                        start_time = point.time
                    elapsed_time = (point.time - start_time).total_seconds()
                    satellites = getattr(point, 'satellites', 20)
                    if satellites is None:
                        satellites = 20
                    if satellites >= 19:
                        data.append([point.time, point.latitude, point.longitude, point.elevation, satellites, elapsed_time])
        
        return pd.DataFrame(data, columns=['timestamp', 'latitude', 'longitude', 'elevation', 'satellites', 'elapsed_time'])

    @torch.no_grad()
    def calculate_sharpness_gpu_ultra(self, frames_batch):
        """
        Ultra-optimized GPU batch sharpness calculation with pre-compiled kernels
        """
        if len(frames_batch) == 0:
            return []
            
        try:
            # Filter out None frames
            valid_frames = []
            valid_indices = []
            for i, frame in enumerate(frames_batch):
                if frame is not None:
                    valid_frames.append(cv2.resize(frame, (1400, 700)))
                    valid_indices.append(i)
            
            if len(valid_frames) == 0:
                return [0.0] * len(frames_batch)
                
            # Convert to tensor batch - optimized
            frames_np = np.stack(valid_frames, axis=0)
            frames_t = torch.from_numpy(frames_np).to(self.device, dtype=torch.float32)
            frames_t = frames_t.permute(0, 3, 1, 2)
            frames_t = torch.mean(frames_t, dim=1, keepdim=True)  # RGB to grayscale
            
            # Apply pre-compiled kernels
            blurred = self.blur_kernel(frames_t)
            laplacian = self.laplace_kernel(blurred)
            
            # Calculate variance (sharpness) - optimized
            sharpness_batch = torch.var(laplacian.view(laplacian.size(0), -1), dim=1).cpu().numpy()
            
            # Map back to original indices
            result = [0.0] * len(frames_batch)
            for i, valid_idx in enumerate(valid_indices):
                result[valid_idx] = float(sharpness_batch[i])
            
            return result
            
        except Exception as e:
            print(f"WARNING: GPU processing error: {e}, falling back to CPU")
            return self._calculate_sharpness_cpu_batch(frames_batch)

    def _calculate_sharpness_cpu_batch(self, frames_batch):
        """Optimized CPU fallback with vectorized operations"""
        results = []
        valid_frames = []
        
        for frame in frames_batch:
            if frame is not None:
                resized = cv2.resize(frame, (1400, 700))
                valid_frames.append(resized)
            else:
                valid_frames.append(None)
        
        # Batch process valid frames
        for frame in valid_frames:
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                sharpness = cv2.Laplacian(blurred, cv2.CV_64F).var()
                results.append(sharpness)
            else:
                results.append(0.0)
        
        return results

    def read_frame_cluster_smart(self, video, cluster_info):
        """
        Smart cluster-based frame reading with intelligent caching
        """
        # Extract all frame indices needed for this cluster
        all_frame_indices = set()
        keyframe_frame_map = {}
        
        for keyframe_idx, keyframe_pos, window_start, window_end in cluster_info:
            window_frames = list(range(window_start, window_end))
            all_frame_indices.update(window_frames)
            keyframe_frame_map[keyframe_idx] = window_frames
        
        # Read frames sequentially for optimal disk access
        sorted_frames = sorted(all_frame_indices)
        frame_cache = {}
        
        # Smart seeking - only seek when necessary
        current_pos = -1
        for frame_idx in sorted_frames:
            try:
                if current_pos != frame_idx:
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                ret, frame = video.read()
                if ret and frame is not None:
                    frame_cache[frame_idx] = frame
                    current_pos = frame_idx + 1
                else:
                    frame_cache[frame_idx] = None
                    current_pos = -1
                    
            except Exception as e:
                print(f"ERROR: Error reading frame {frame_idx}: {e}")
                frame_cache[frame_idx] = None
                current_pos = -1
        
        return frame_cache, keyframe_frame_map

    def process_frame_cluster(self, cluster_info):
        """
        Process a cluster of frames with shared video context
        """
        video = cv2.VideoCapture(str(self.video_path))
        if not video.isOpened():
            return []
        
        results = []
        try:
            # Read all frames for this cluster
            frame_cache, keyframe_frame_map = self.read_frame_cluster_smart(video, cluster_info)
            
            # Process each keyframe in the cluster
            for keyframe_idx, keyframe_pos, window_start, window_end in cluster_info:
                window_frames = keyframe_frame_map[keyframe_idx]
                
                # Get frames for this keyframe
                keyframe_frames = [frame_cache.get(idx) for idx in window_frames]
                
                # Calculate sharpness using GPU
                if keyframe_frames:
                    if torch.cuda.is_available():
                        sharpness_values = self.calculate_sharpness_gpu_ultra(keyframe_frames)
                    else:
                        sharpness_values = self._calculate_sharpness_cpu_batch(keyframe_frames)
                    
                    # Find best frame
                    if sharpness_values:
                        best_idx = np.argmax(sharpness_values)
                        best_frame_idx = window_frames[best_idx]
                        best_sharpness = sharpness_values[best_idx]
                        best_frame = keyframe_frames[best_idx]
                        
                        results.append({
                            'keyframe_idx': keyframe_idx,
                            'best_frame_idx': best_frame_idx,
                            'best_sharpness': best_sharpness,
                            'best_frame': best_frame,
                            'window_bounds': (window_start, window_end)
                        })
        
        finally:
            video.release()
        
        return results

    def save_sharp_ultra_optimized(
            self, 
            window: int,
            out_dir: Path,
            prefix: str,
            num_threads: int = 8  # Use threads instead of processes for GPU access
        ):
        """
        Ultra-optimized sharp selection with intelligent clustering and GPU acceleration
        """
        print(f"Starting ULTRA-OPTIMIZED sharp selection:")
        print(f"   - {len(self.keyframes)} keyframes")
        print(f"   - {len(self.frame_clusters)} frame clusters")
        print(f"   - Window size: {window}")
        print(f"   - Threads: {num_threads}")
        print(f"   - Device: {self.device}")
        
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pairs").mkdir(parents=True, exist_ok=True)
        
        video_json = {"keyframes": {}}
        selected_idxs = set()
        gps_file_data = [] if self.gps_enabled else None
        
        # Create a progress bar for individual keyframes
        keyframe_progress = tqdm(total=len(self.keyframes), desc="Processing keyframes", unit="kf")
        processed_keyframes = set()
        
        # Process clusters in parallel using threads (not processes) for GPU access
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all clusters
            future_to_cluster = {
                executor.submit(self.process_frame_cluster, cluster): i 
                for i, cluster in enumerate(self.frame_clusters)
            }
            
            # Collect results (no progress bar on clusters since there are only 2-3)
            for future in as_completed(future_to_cluster):
                try:
                    cluster_results = future.result(timeout=300)  # 5 minute timeout
                    
                    # Save results from this cluster
                    for result in cluster_results:
                        keyframe_idx = result['keyframe_idx']
                        best_frame_idx = result['best_frame_idx']
                        best_sharpness = result['best_sharpness']
                        best_frame = result['best_frame']
                        
                        if best_frame is not None and keyframe_idx not in processed_keyframes:
                            # Save image
                            cv2.imwrite(str(out_dir / f"{prefix}_{keyframe_idx}.jpg"), best_frame)
                            
                            # Track selection
                            if best_frame_idx not in selected_idxs:
                                selected_idxs.add(best_frame_idx)
                            
                            # GPS data
                            if self.gps_enabled:
                                gps_row = self.find_closest_gpx_track_point(best_frame_idx, 30, 10)
                                if gps_row is not None:
                                    gps_file_data.append(f"{prefix}_{keyframe_idx}.jpg {gps_row.latitude} {gps_row.longitude} {gps_row.elevation} 20.0 20.0 35.0\n")
                            
                            # Metadata
                            video_json["keyframes"][str(keyframe_idx)] = {
                                "frame_idx": best_frame_idx,
                                "name": f"{prefix}_{keyframe_idx}",
                                "matches": []
                            }
                            
                            # Update progress
                            processed_keyframes.add(keyframe_idx)
                            keyframe_progress.update(1)
                            keyframe_progress.set_postfix({
                                'current': keyframe_idx,
                                'sharpness': f"{best_sharpness:.2f}"
                            })
                
                except Exception as e:
                    cluster_idx = future_to_cluster[future]
                    print(f"ERROR: Error processing cluster {cluster_idx}: {e}")
                    continue
        
        # Close progress bar
        keyframe_progress.close()
        success_count = len(processed_keyframes)
        
        # Save GPS data
        if self.gps_enabled and gps_file_data:
            with open(str(out_dir / "pairs" / f"gpx_{prefix}.txt"), 'w') as f:
                f.writelines(gps_file_data)
        
        print(f"ULTRA-OPTIMIZED processing completed!")
        print(f"Success: {success_count} out of {len(self.keyframes)} keyframes")
        
        return video_json

    def find_closest_gpx_track_point(self, frame, frame_rate=30, min_frame_distance=15):
        """Optimized GPX lookup with binary search"""
        if not self.gps_enabled:
            return None
            
        elapsed_times = self.gps_data['elapsed_time'].values
        elapsed_frames = elapsed_times * frame_rate
        
        pos = bisect.bisect_left(elapsed_frames, frame)
        closest_index = -1
        min_distance = float('inf')
        
        if pos < len(elapsed_frames):
            if abs(elapsed_frames[pos] - frame) <= min_frame_distance:
                closest_index = pos
                min_distance = abs(elapsed_frames[pos] - frame)
        
        if pos > 0:
            if abs(elapsed_frames[pos - 1] - frame) <= min_frame_distance:
                if abs(elapsed_frames[pos - 1] - frame) < min_distance:
                    closest_index = pos - 1
        
        return self.gps_data.iloc[closest_index] if closest_index != -1 else None

# All the support functions remain the same...
def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def is_close_pair(pair, camera_centers, max_distance):
    cam1, cam2 = pair
    dist = calculate_distance(np.array(camera_centers[cam1]["w2c"])[:3,3], np.array(camera_centers[cam2]["w2c"])[:3,3])
    return dist <= max_distance

def get_match_matrix(camera_centers, max_distance):
    keys = list(camera_centers.keys())
    num_keys = len(keys)
    match_matrix = np.zeros((num_keys, num_keys), dtype=bool)

    pairs = list(combinations(range(num_keys), 2))
    args = [((keys[i], keys[j]), camera_centers, max_distance) for i, j in pairs]

    with mp.Pool(mp.cpu_count()) as pool:
        close_results = pool.starmap(is_close_pair_wrapper, args)

    for (i, j), is_close in zip(pairs, close_results):
        match_matrix[i, j] = is_close
        match_matrix[j, i] = is_close

    return match_matrix

def get_avg_temporally_adjacent_camera_distance(camera_centers):
    keys = list(camera_centers.keys())
    n = len(keys)
    cam_dists = []
    for i in range(n-1):
        dist = calculate_distance(np.array(camera_centers[keys[i]]["w2c"])[:3,3], np.array(camera_centers[keys[i + 1]]["w2c"])[:3,3])
        cam_dists.append(dist)
    avg_dist = sum(cam_dists)/len(cam_dists)
    return avg_dist

def is_close_pair_wrapper(pair, camera_centers, max_distance):
    return is_close_pair(pair, camera_centers, max_distance)

def worker(keyframes_map, data, key_1, results):
    lm_ids_1 = data["keyframes"][key_1]['lm_ids']
    for key_2, keyframe_2 in data["keyframes"].items():
        if key_1 == key_2:
            continue
        lm_ids_2 = keyframe_2['lm_ids']
        for lm_id in lm_ids_1:
            if lm_id == -1:
                continue
            elif lm_id in lm_ids_2:
                with results.get_lock():
                    index = keyframes_map[key_1] * len(data["keyframes"]) + keyframes_map[key_2]
                    results[index] += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, help="Output directory name")
    parser.add_argument("--frame_traj", type=Path, help="frame_trajectory.txt file")
    parser.add_argument("--key_traj", type=Path, help="keyframe_trajectory.txt directory")
    parser.add_argument("--gpx", type=Path, default=None, help=".gpx file path if it exists")
    parser.add_argument("--frame_skip", type=int, default=1, help="Frame skip used in stella")
    parser.add_argument("--window", type=int, default=5, help="Window size for sharpness evaluation")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index")
    parser.add_argument("--msg_file", type=str, help="msg_db file from stella_vslam")
    parser.add_argument("--tag", type=str, help="tag for the sharp outs")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads for parallel processing")

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pairs").mkdir(parents=True, exist_ok=True)

    # Use ultra-optimized selector
    sharp_selector = UltraOptimizedSharpSelector(
        args.frame_traj, 
        args.key_traj, 
        args.input,
        args.gpx,
        args.start_frame,
        args.end_frame,
        args.frame_skip
    )

    video_json = sharp_selector.save_sharp_ultra_optimized(
        args.window, 
        args.output, 
        args.tag,
        args.num_threads
    )

    # Continue with original pipeline processing...
    add_spatial_pairs = True
    remove_non_spatial_stella = False

    data_files = os.listdir(args.msg_file)
    msg_file = [file for file in data_files if file.endswith('.msg')][0]
    with open(os.path.join(args.msg_file, msg_file), "rb") as f:
        msg = msgpack.unpackb(f.read(), use_list=False, raw=False)

    keyframes_map = {}
    num_slam_keyframes = len(msg["keyframes"])
    num_processed_keyframes = len(video_json["keyframes"])
    
    print(f"INFO: SLAM keyframes: {num_slam_keyframes}")
    print(f"INFO: Processed keyframes: {num_processed_keyframes}")
    
    # CRITICAL FIX: Sort keyframes by timestamp, not by ID!
    # The keyframe IDs have gaps and don't correspond to temporal order
    keyframe_items = []
    for key, value in msg['keyframes'].items():
        timestamp = value.get('ts', 0)
        keyframe_items.append((timestamp, key, value))
    
    # Sort by timestamp to ensure proper temporal order
    keyframe_items.sort(key=lambda x: x[0])
    
    for ptcnt, (timestamp, key, value) in enumerate(keyframe_items):
        keyframes_map[key] = ptcnt
        
        # Only add camera pose if this keyframe exists in video_json
        if str(ptcnt) in video_json["keyframes"]:
            # Handle both coordinate system formats that SLAM might output
            if "trans_wc" in value:
                # World-to-camera format (newer SLAM versions)
                t = value["trans_wc"]
                r = R.from_quat(value["rot_wc"])
                T = np.eye(4)
                T[:3,:3] = r.as_matrix()
                T[:3, 3] = np.asarray(t).copy()
                video_json["keyframes"][str(ptcnt)]["w2c"] = T.tolist()
            elif "trans_cw" in value:
                # Camera-to-world format (older SLAM versions) - need to invert
                t_cw = value["trans_cw"]
                r_cw = R.from_quat(value["rot_cw"])
                T_cw = np.eye(4)
                T_cw[:3,:3] = r_cw.as_matrix()
                T_cw[:3, 3] = np.asarray(t_cw)
                T_wc = np.linalg.inv(T_cw)
                video_json["keyframes"][str(ptcnt)]["w2c"] = T_wc.tolist()
            else:
                print(f"ERROR: Keyframe {ptcnt} has unknown coordinate format!")
        else:
            print(f"WARNING: SLAM keyframe {ptcnt} (ID {key}, ts {timestamp:.3f}) not found in processed keyframes, skipping camera pose")

    # Use the number of keyframes that actually have camera poses
    num_keyframes = len([kf for kf in video_json["keyframes"].values() if "w2c" in kf])
    print(f"INFO: Keyframes with camera poses: {num_keyframes}")
    video_json["keyframe_count"] = num_keyframes
    video_json["reloc_keyframes"] = []

    # Use SLAM keyframes count for pair matrix (this is correct)
    slam_keyframes_count = len(msg["keyframes"])
    pair_mat = mp.Array('i', slam_keyframes_count * slam_keyframes_count)

    processes = []
    for key_1 in msg["keyframes"]:
        p = mp.Process(target=worker, args=(keyframes_map, msg, key_1, pair_mat))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    pair_mat_np = np.frombuffer(pair_mat.get_obj(), dtype='int32').reshape((slam_keyframes_count, slam_keyframes_count))
    np.save(os.path.join(os.path.join(args.output, "pairs"), args.tag + '_array.npy'), pair_mat_np)

    # Only process matches for keyframes that have camera poses
    keyframes_with_poses = [int(k) for k, v in video_json["keyframes"].items() if "w2c" in v]
    print(f"INFO: Processing matches for {len(keyframes_with_poses)} keyframes with camera poses")
    
    if len(keyframes_with_poses) > 0:
        avg_cam_dist = get_avg_temporally_adjacent_camera_distance(video_json["keyframes"])
        spatial_pair_mat = get_match_matrix(video_json["keyframes"], 3*avg_cam_dist)
        match_dict = {}
        
        # Only process matches for keyframes that exist and have camera poses
        max_keyframe_idx = max(keyframes_with_poses)
        pair_mat_size = min(slam_keyframes_count, pair_mat_np.shape[0], pair_mat_np.shape[1], max_keyframe_idx + 1)
        
        for i in keyframes_with_poses:
            if i < pair_mat_size and i < spatial_pair_mat.shape[0]:
                matches = [(j, pair_mat_np[i, j]) for j in range(i + 1, pair_mat_size) if j in keyframes_with_poses]
                spatial_matches = [j for j in range(i + 1, min(spatial_pair_mat.shape[1], pair_mat_size)) 
                                 if j in keyframes_with_poses and spatial_pair_mat[i, j]]
                valid_matches = [match for match in matches if match[1] >= 30]
                valid_matches.sort(key=lambda x: x[1], reverse=True)
                valid_indices = [match[0] for match in valid_matches]

                valid_indices = set(valid_indices)
                spatial_matches = set(spatial_matches)
                if remove_non_spatial_stella:
                    valid_indices = valid_indices - spatial_matches
                if add_spatial_pairs:
                    valid_indices = valid_indices.union(spatial_matches)
                valid_indices = list(valid_indices)

                match_dict[i] = valid_indices
            else:
                match_dict[i] = []
    else:
        match_dict = {}

    # Assign matches to individual keyframes (not as global matches)
    for keyframe_idx, match_list in match_dict.items():
        if str(keyframe_idx) in video_json["keyframes"]:
            # Convert match indices to actual keyframe names
            match_names = []
            for match_idx in match_list:
                if str(match_idx) in video_json["keyframes"]:
                    match_names.append(video_json["keyframes"][str(match_idx)]["name"])
            video_json["keyframes"][str(keyframe_idx)]["matches"] = match_names

    with open(os.path.join(args.output, "video.json"), 'w') as f:
        json.dump(video_json, f, indent=4)

    print("ULTRA-OPTIMIZED Sharp frame selection completed successfully!") 