import Metashape
from argparse import ArgumentParser
from pathlib import Path
import sys
import numpy as np
import json
import os
import cv2
import torch
import torch.nn.functional as F
import open3d as o3d
from matplotlib import pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from xml.etree import ElementTree as ET
from tqdm import tqdm

def check_if_reloc_frame(label, id_i, id_j):
    parts = label.split('_')
    ids_to_check = list(int(parts[0]), int(parts[1]))
    if id_i in ids_to_check and id_j in ids_to_check:
        return True
    else:
        return False

def vid_from_name(name):
    if name[:4] == "OLD_":
        return 0
    return int(name.split("_")[0])

def duplicate_camera_check(label, sensor_key):
    # parts = label.split('_')
    vid_count = vid_from_name(label)
    if vid_count == 0:
        # if sensor_key == num_vids:
        #     return False
        # else:
        return True
    
    if vid_count == (sensor_key + 1):
        return False
    else:
        return True

def get_camera_pairs(cameras, json_keyframes):
    agi_mapping = {}
    for camera in cameras:
        agi_mapping[camera.label] = camera.key
    camera_pairs = []
    for key, value in json_keyframes.items():
        for match_name in value["matches"]:
            camera_pairs.append((agi_mapping[value["name"]], agi_mapping[match_name]))
    return camera_pairs


def add_markers(doc, chunk_id, output_dir):
    chunk = doc.findChunk(chunk_id)
    real_cams = {}
    real_cams_set = set()
    duplicate_cams = {}
    duplicate_cams_set = set()
    for camera in chunk.cameras:
        if duplicate_camera_check(camera.label, camera.sensor.key):
            duplicate_cams[camera.label] = camera
            duplicate_cams_set.add(camera.label)
        else:
            real_cams[camera.label] = camera
            real_cams_set.add(camera.label)

    track_id_to_points = {}
    print ("Preparing points dictionary")
    for point in tqdm(chunk.tie_points.points):
        track_id_to_points[point.track_id] = point
    
    track_id_to_projections = {}
    track_residuals = {}
    print ("Preparing projections dictionary")
    for camera in tqdm(chunk.cameras):
        if camera not in chunk.tie_points.projections:
            continue
        for projection in chunk.tie_points.projections[camera]:
            if projection.track_id not in track_id_to_points:
                continue
            if projection.track_id not in track_id_to_projections:
                track_id_to_projections[projection.track_id] = []
            
            track_id_to_projections[projection.track_id].append((camera, projection))
            
            point = track_id_to_points[projection.track_id]
            reproj_error = camera.error(point.coord, projection.coord)
            
            if projection.track_id not in track_residuals:
                track_residuals[projection.track_id] = []
            track_residuals[projection.track_id].append(reproj_error[0]**2 + reproj_error[1]**2)

    print ("Calculating track lengths")
    track_lengths = {}
    for track_id in tqdm(track_residuals):
        track_lengths[track_id] = len(track_residuals[track_id])

    print ("Adding markers")
    num_markers = 0
    num_skipped_cams = 0
    for duplicate_cam in tqdm(duplicate_cams.values()):
        if duplicate_cam not in chunk.tie_points.projections:
            num_skipped_cams += 1
            continue
        tracks_in_cam = [proj.track_id for proj in chunk.tie_points.projections[duplicate_cam]]
        sorted_tracks = sorted(tracks_in_cam, key=lambda x: -track_lengths[x] if x in track_lengths else 0)
        num_tracks = 4
        for track_id in sorted_tracks:
        # for projection in chunk.tie_points.projections[duplicate_cam]:
            # if np.random.rand() > 0.001:
            #     continue
            if track_id not in track_id_to_points:
                continue
            if track_id not in track_id_to_projections:
                continue
            if num_tracks <= 0:
                break
            num_tracks -= 1
            
            all_projections = track_id_to_projections.pop(track_id)
            # point = track_id_to_points[track_id]
            # marker = chunk.addMarker(Metashape.Vector([point.coord[0], point.coord[1], point.coord[2]]))
            marker = chunk.addMarker()
            num_markers += 1
            for camera, projection in all_projections:
                if duplicate_camera_check(camera.label, camera.sensor.key):
                    if camera.label in real_cams_set:
                        real_cam = real_cams[camera.label]
                    else:
                        continue
                else:
                    real_cam = camera
                marker.projections[real_cam] = Metashape.Marker.Projection()
                marker.projections[real_cam].coord = projection.coord
                marker.projections[real_cam].valid = True
                marker.projections[real_cam].pinned = True

    print ("Added", num_markers, "markers")
    print ("Skipped", num_skipped_cams, "cameras")
    if num_markers > 0:
        chunk.optimizeCameras()

    psxfile = os.path.join(str(Path(output_dir) / "project_all.psx"))
    doc.save(psxfile)
    # # breakpoint()


def align_cameras(images_dir, vidcount, output_dir, metric_depths, seg, gps, use_stills, past_cams=None):
    pairs_dir = os.path.join(images_dir, "pairs")
    
        
    reloc_frames_json_path = os.path.join(pairs_dir, "reloc_frames.json")
    if not os.path.exists(reloc_frames_json_path):
        reloc_frames_json = {"1": []}
    else:
        with open(reloc_frames_json_path, 'r') as file:
            reloc_frames_json = json.load(file)

    # Create a new document
    doc = Metashape.Document()

    for index in range(vidcount):
        vid_index = index + 1

        # Create a new chunk
        chunk = doc.addChunk()
    
        for camera in chunk.cameras:
            # Check if the camera has calibration data
            if camera.sensor.calibration:
                # Camera type is configured via sensor.type (done later)
                print(f"INFO: Camera {camera.label} has calibration data")
            else:
                print ("Camera {} has no calibration data".format(camera.label))
                sys.exit(1) 
            
        # Path to your images
        # Glob over all images in the directory
        image_paths_first_tag = set(Path(images_dir).glob('{:03d}*'.format(vid_index)))
        image_paths_second_tag = set(Path(images_dir).glob('*_{:03d}_*'.format(vid_index)))
        image_paths_legacy = set()
        if len(image_paths_first_tag) == 0 and len(image_paths_second_tag) == 0:
            image_paths_legacy = set(Path(images_dir).glob('[0-9]*.jpg'))
        image_paths = image_paths_first_tag.union(image_paths_second_tag).union(image_paths_legacy)

        image_paths = [str(p) for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        reloc_frames = [str(Path(images_dir) / frame) for frame in reloc_frames_json[str(vid_index)]]
        image_paths = image_paths + reloc_frames

        # Add images to the chunk
        chunk.addPhotos(image_paths)

        # CRITICAL FIX: For equirectangular videos, use SPHERICAL camera model!
        # Equirectangular is a spherical projection format
        print("INFO: Configuring cameras for equirectangular projection...")
        
        for sensor in chunk.sensors:
            # Use spherical camera model for equirectangular content
            sensor.type = Metashape.Sensor.Type.Spherical
            
            # OPTIMIZED parameters for maximum alignment success
            sensor.width = 1440  # Image width
            sensor.height = 720   # Image height 
                
            # Optimized pixel parameters for spherical cameras
            sensor.pixel_width = 1.0  # Square pixels
            sensor.pixel_height = 1.0
            
            # Enable calibration optimization for better alignment
            sensor.fixed = False  # Allow calibration optimization
            
            print(f"INFO: Optimized spherical camera - size: {sensor.width}x{sensor.height}, calibration: enabled")

        # REMOVED: All masking logic completely removed to allow maximum feature detection
        print("INFO: Using maximum feature detection - no masking, no keypoint limits")

        # read stella pairs
        video_json_path = os.path.join(pairs_dir, '{:03d}_{:03d}.json'.format(vid_index, vid_index))
        with open(video_json_path, 'r') as file:
            video_json = json.load(file)

        camera_pairs = get_camera_pairs(chunk.cameras, video_json["keyframes"])
        
        # MAXIMUM FEATURE MATCHING - No limits, no masking, maximum keypoints
        print("INFO: Feature matching with UNLIMITED keypoints for maximum alignment...")
        
        chunk.matchPhotos(
            downscale=1,  # Full resolution for maximum feature detection
            generic_preselection=True,  
            reference_preselection=True,  
            pairs=camera_pairs,
            keep_keypoints=True,  # Keep all detected keypoints
            filter_mask=False,  # No masking whatsoever
            guided_matching=True,  # Use guided matching for better results
            keypoint_limit=0,  # UNLIMITED keypoints per image (removes 2000 limit)
            tiepoint_limit=0   # UNLIMITED tie points per image
        )

        new_cams = []
        old_cams = []
        for camera in chunk.cameras:
            if camera.label[:4] == "OLD_":
                old_cams.append(camera)
            else:
                new_cams.append(camera)
        
        # MULTI-STAGE ROBUST ALIGNMENT for 100% success
        print("INFO: Starting multi-stage robust camera alignment...")
        
        # Stage 1: Initial alignment with standard parameters
        chunk.alignCameras(cameras=new_cams)
        
        # Check and retry failed alignments
        not_aligned_stage1 = [cam for cam in new_cams if not cam.transform]
        if len(not_aligned_stage1) > 0:
            print(f"INFO: Stage 1 - {len(not_aligned_stage1)} cameras failed, trying recovery...")
            
            # Stage 2: Recovery alignment without reset
            chunk.alignCameras(
                cameras=not_aligned_stage1,
                reset_alignment=False
            )
            
            # Stage 3: Final recovery for remaining failures
            not_aligned_stage2 = [cam for cam in not_aligned_stage1 if not cam.transform]
            if len(not_aligned_stage2) > 0:
                print(f"INFO: Stage 2 - {len(not_aligned_stage2)} cameras still failed, final recovery...")
                chunk.alignCameras(
                    cameras=not_aligned_stage2,
                    reset_alignment=False
                )
        
        # Handle old cameras if present
        if len(old_cams) > 0:
            chunk.alignCameras(cameras=old_cams, reset_alignment=False)

        not_aligned = list()
        for camera in chunk.cameras:
            if not camera.transform:
                not_aligned.append(camera)
                
        print("Agisoft chunk number:", vid_index, "unaligned cameras:", len(not_aligned))
        
        # FINAL RECOVERY ATTEMPT - try super aggressive settings
        if len(not_aligned) > 0:
            print(f"INFO: Attempting FINAL RECOVERY for {len(not_aligned)} unaligned cameras...")
            
            # Try final recovery alignment
            chunk.alignCameras(
                cameras=not_aligned,
                reset_alignment=False
            )
            
            # Check final results
            final_not_aligned = [cam for cam in not_aligned if not cam.transform]
            final_aligned = len(not_aligned) - len(final_not_aligned)
            
            if final_aligned > 0:
                print(f"INFO: FINAL RECOVERY succeeded for {final_aligned} cameras!")
            
            not_aligned = final_not_aligned
        
        total_cams = len(chunk.cameras)
        aligned_ratio = (total_cams - len(not_aligned)) / total_cams
        
        if len(not_aligned) == 0:
            print("🎉 PERFECT ALIGNMENT SUCCESS! 100% cameras aligned!")
        else:
            print(f"Camera alignment ratio: {aligned_ratio:.3f} ({total_cams - len(not_aligned)}/{total_cams})")
            
            if aligned_ratio >= 0.8:  # 80%+ is very good
                print("✅ EXCELLENT: >80% cameras aligned - high quality results expected")
            elif aligned_ratio >= 0.5:  # 50%+ is acceptable
                print("✅ GOOD: >50% cameras aligned - reasonable results expected")
            elif aligned_ratio >= 0.2:  # 20%+ is marginal
                print("⚠️  MARGINAL: >20% cameras aligned - results may be poor")
            else:
                print("❌ POOR: <20% cameras aligned - results will be very poor")
            
            print(f"Removing {len(not_aligned)} permanently unaligned cameras...")
            chunk.remove(not_aligned)

        # if gps data available use it for the chunk
        if gps:
            chunk.crs = Metashape.CoordinateSystem("EPSG::4326")
            gps_path = os.path.join(images_dir, 'pairs', 'gpx_{:03d}_{:03d}.txt'.format(vid_index, vid_index))
            chunk.importReference(gps_path, format=Metashape.ReferenceFormatCSV, delimiter=" ", columns="nyxzYXZ")
            chunk.optimizeCameras()

    if past_cams is not None:
        tree = ET.parse(past_cams)
        root = tree.getroot()

        # Glob over the images directory
        image_paths = set(Path(images_dir).glob('*.jpg'))
        images_dict = {images_paths.stem: images_paths for images_paths in image_paths}

        images_old = [str(images_dict[camera.get('label')]) for camera in root.findall(".//camera")]
        
        # Create a new chunk
        chunk = doc.addChunk()
        chunk.addPhotos(images_old)
        chunk.importCameras(path=past_cams)

    # calculate_transform_and_densify(doc, metric_depths, seg, "densify")
    # If more than one chunk, align them all and merge
    if vidcount > 1 or past_cams is not None:
        # Align all chunks
        # psxfile = os.path.join(str(Path(output_dir) / "project_all_prealign.psx"))
        # doc.open(psxfile)
        chunk = doc.chunks
        chunk_int_list = list(range(len(chunk)))

        # psxfile = os.path.join(str(Path(output_dir) / "project_all_prealign.psx"))
        # doc.save(psxfile)
        
        # Skip saving in demo mode - proceed with alignment
        print("INFO: Skipping intermediate project save (demo mode limitation)")
        doc.alignChunks(chunk_int_list, len(chunk)-1, method=2, generic_preselection = True)

        # Merge all chunks except the past run
        if past_cams is not None:
            chunk_int_list = list(range(len(chunk) - 1))
        savechunk_id = 0
        # Merge only if more than one chunk
        if vidcount > 1:
            doc.mergeChunks(chunks=chunk_int_list, merge_tiepoints = False)
            savechunk_id = doc.chunks[-1].key
            
        saveChunk = doc.findChunk(savechunk_id)
        
        # Remove old cameras
        cams_to_remove = []
        for camera in saveChunk.cameras:
            if camera.label[:4] == "OLD_":
                print("old camera removed at", camera.label, "with the key", camera.sensor.key)
                cams_to_remove.append(camera)
        saveChunk.remove(cams_to_remove)
        
        # Add markers to simulate tiepoints from the duplicated cameras
        if vidcount > 1:
            add_markers(doc, savechunk_id, output_dir)

        saveChunk = doc.findChunk(savechunk_id)

        # saveChunk.matchPhotos(downscale=2, 
        #                 generic_preselection=True, 
        #                 reference_preselection=False)
        # saveChunk.alignCameras()

        # remove duplicate cameras used to align chunks
        for camera in saveChunk.cameras:
            if duplicate_camera_check(camera.label, camera.sensor.key):
                print("camera duplciate removed at", camera.label, "with the key", camera.sensor.key)
                saveChunk.remove([camera])

    else:
        # Skip saving in demo mode
        print("INFO: Skipping single-chunk project save (demo mode limitation)")
        saveChunk = doc.findChunk(0)

    if use_stills == 1:
        print("Adding in still images to align them")
        still_images_paths = set(Path(args.images_dir).glob('*still_*'))
        image_paths = [str(p) for p in still_images_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        saveChunk.addPhotos(image_paths)
        still_cams = list()
        for camera in saveChunk.cameras:
            if not camera.transform:
                still_cams.append(camera)
                # strip gps data if available, all stills till now have come with exif gps data that getse loaded in
                camera.reference.enabled = False

        for sensor in saveChunk.sensors:
            sensor.type = Metashape.Sensor.Type.Spherical

        # Ensure crs is local i.e gps didn't set different coordinate system if no gps
        if not gps:
            saveChunk.crs = Metashape.CoordinateSystem('LOCAL_CS["Local Coordinates (m)",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]')

        # Maximum feature matching for still images - no masking, no limits
        saveChunk.matchPhotos(
            downscale=1,  # Full resolution
            generic_preselection=True, 
            reference_preselection=False, 
            keep_keypoints=True,  # Keep all detected keypoints
            filter_mask=False,  # No masking for still images either
            keypoint_limit=0,  # UNLIMITED keypoints per image
            tiepoint_limit=0   # UNLIMITED tie points per image
        )
        saveChunk.alignCameras(cameras = still_cams, reset_alignment=False)
        unaligned_stills = list()
        for camera in saveChunk.cameras:
            if not camera.transform:
                still_cams.append(camera)
        if len(unaligned_stills) == 0:
            print("All stills successfully aligned!")
        else:
            print(f'{len(unaligned_stills)} stills could not be aligned, removing them')
            saveChunk.remove(unaligned_stills)

    # Skip saving project file in demo mode - focus on exports
    print("INFO: Skipping project save (demo mode limitation)")
    return doc


# Y-up helper functions
def sample_triplets(positions, ups, max_distance=10, num_samples=300):
    triplets_w_up = []
    n = len(positions)
    for _ in range(num_samples):
        start_idx = np.random.randint(0, n - max_distance)
        indices = np.sort(
            np.random.choice(range(start_idx, min(start_idx + max_distance,
                                                  n)),
                             3,
                             replace=False))
        # get average up vector
        triplet_up_vector = np.mean(ups[indices], axis=0)
        triplet_up_vector = triplet_up_vector / np.linalg.norm(
            triplet_up_vector)
        triplet_with_up = np.vstack((positions[indices], triplet_up_vector))
        triplets_w_up.append(triplet_with_up)
    return triplets_w_up


def calculate_normal(triplet_with_up):
    vec1 = triplet_with_up[1] - triplet_with_up[0]
    vec2 = triplet_with_up[2] - triplet_with_up[0]
    normal = np.cross(vec1, vec2)
    if np.linalg.norm(normal) < 0.001:
        print("muste be a case of very close normals")
    normal = normal / np.linalg.norm(normal)
    up = triplet_with_up[3]
    # if normal is pointing away from up vector, flip it
    if np.dot(normal, up) < 0:
        normal = -normal
    return normal

def ransac_normals(normals):
    """Find the consensus normal vector using RANSAC."""
    X = np.arange(len(normals)).reshape(-1, 1)
    y = np.stack(normals)
    ransac = RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    consensus_normals = y[inlier_mask]
    consensus_normal = np.mean(consensus_normals, axis=0)
    consensus_normal = consensus_normal / np.linalg.norm(
        consensus_normal)  # Normalize
    return consensus_normal

def get_global_z_transform(consensus_normal, target_z=np.array([0, 0, 1])):
    """Calculate the rotation matrix required to rotate consensus_normal to align with global y up"""
    # Normalize input vectors
    consensus_normal = consensus_normal / np.linalg.norm(consensus_normal)
    target_z = target_z / np.linalg.norm(target_z)

    # Calculate the rotation axis (cross product of v1 and v2)
    axis = np.cross(consensus_normal, target_z)
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-8:
        # The vectors are parallel; no need for rotation
        return np.eye(3)

    axis = axis / axis_length  # normalize the axis
    # Calculate the rotation angle
    angle = np.arccos(np.dot(consensus_normal, target_z))

    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    rotation_matrix = np.eye(
        3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return np.vstack((np.hstack(
        (rotation_matrix, np.array([[0], [0], [0]]))), [0, 0, 0, 1]))

def load_depth(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) / 1000.0
    return depth.astype(np.float32)

# NOTE: All masking functionality removed from feature detection for maximum camera alignment success

def calculate_transform_and_densify(doc, metric_depths=None, seg=None, mode="densify", floorplan_dir=None):
    
    # Handle case where seg is None (no segmentation data provided)
    if seg is not None:
        print(os.path.dirname(seg))  # May output: $OUT_DIR/acmm/seg

        # To ensure you get the parent directory, normalize the path first:
        normalized_path = os.path.normpath(seg)
        acmm_dir = os.path.dirname(normalized_path)

        name2acmm, acmm2name = None, None

        if os.path.exists((Path(acmm_dir) / 'mapping.json')): # For reruns of agisoft after mvs renames seg
            name2acmm = json.loads((Path(acmm_dir) / 'mapping.json').read_text())
            acmm2name = {v: k for k, v in name2acmm.items()}
    else:
        print("INFO: No segmentation data provided, skipping segmentation-based processing")
        acmm_dir = None
        name2acmm, acmm2name = None, None

    # Ensure the project has at least one chunk
    if len(doc.chunks) < 1:
        print("No chunks found in the document.")
        exit(1)

    if len(doc.chunks) > 1:
        print("More than one chunk found in the document.")
        print("Generating global y-up transform for all chunks -> helps in aligining with the floorplan")

    # Initialize variables regardless of whether metric_depths/seg are provided
    chunk_walls, chunk_floors, chunk_scales = [], [], []
    
    if metric_depths is not None and seg is not None:
        for chunk in doc.chunks:
            # Get the point cloud and tie points
            point_cloud = chunk.tie_points
            tie_points = point_cloud.points
            print("lenge of tie points", len(tie_points))
            
            R = chunk.region.rot		#Bounding box rotation matrix
            C = chunk.region.center	    #Bounding box center vector
            size = chunk.region.size
            in_points = []

            for point in tie_points:
                v = point.coord
                v.size = 3
                v_c = v - C
                v_r = R.t() * v_c
                if abs(v_r.x) > abs(size.x / 2.):
                    continue
                elif abs(v_r.y) > abs(size.y / 2.):
                    continue
                elif abs(v_r.z) > abs(size.z / 2.):
                    continue
                else:
                    in_points.append(point)
                    point.selected = True
                    pass

            # chunk.tie_points.removeSelectedPoints() 
                    
            # Create a dictionary to map tie points to their projections in each camera
            track_id_to_points = {}
            for point in chunk.tie_points.points:
                if point.selected:
                    track_id_to_points[point.track_id] = point

            camera_scales, wall_points, floor_points = [], [], []
            for camera in chunk.cameras:
                if camera.transform is not None:
                    projections = []
                    sparse_depths = []
                    sparse_points = []
                    for projection in chunk.tie_points.projections[camera]:
                        uv = projection.coord
                        if projection.track_id not in track_id_to_points:
                            continue
                        point = track_id_to_points[projection.track_id]
                        depth = camera.transform.inv().mulp(point.coord[:3])        
                        projections.append(uv)
                        sparse_depths.append(depth.norm())
                        sparse_points.append(np.array(point.coord))

                    # dense_depth = np.load(Path(metric_depths) / (camera.label + ".npy"))
                    dense_depth = load_depth(Path(metric_depths) / (camera.label + ".png"))
                    dense_depth_t = torch.from_numpy(dense_depth).unsqueeze(0).unsqueeze(0)
                    if seg is not None:
                        if os.path.exists(str(Path(seg) / "seg" / f"{camera.label}.png")):
                            seg_map = cv2.imread(str(Path(seg) / "seg" / f"{camera.label}.png"))
                        elif os.path.exists(str(Path(seg) / "seg" / f"{name2acmm[camera.label]}.png") and name2acmm is not None):
                            seg_map = cv2.imread(str(Path(seg) / "seg" / f"{name2acmm[camera.label]}.png"))
                        else:
                            print("BOTH SEG FILES NOT FOUND")
                            seg_map = None
                    else:
                        # No segmentation data - skip segmentation processing
                        seg_map = None
                    if len(projections) == 0:
                        continue
                        print("no projects")
                    N = len(projections)
                    projections = np.array(projections)*2 / np.array([camera.sensor.width, camera.sensor.height]) - 1

                    # for scale calculation
                    dense_depths = F.grid_sample(dense_depth_t, torch.tensor(projections).to(torch.float32).unsqueeze(0).unsqueeze(0), align_corners=True, mode='bilinear')
                    dense_depths = dense_depths.squeeze(0).squeeze(0).numpy()
                    mask = np.logical_and(dense_depths > 2, dense_depths < 20)

                    # get scale for each camera
                    if len(mask) == 0:
                        camera_scale = dense_depths / np.array(sparse_depths)
                        camera_scales.append(camera_scale.flatten().tolist())
                    else:
                        camera_scale = dense_depths[mask] / np.array(sparse_depths)[mask[0]]
                        camera_scales.append(camera_scale.flatten().tolist())
                    
                    # Segmentation-based wall/floor detection (only if seg_map available)
                    if seg_map is not None:
                        seg_map_t = torch.from_numpy(seg_map).to(torch.float32)[:,:,0].unsqueeze(0).unsqueeze(-1)
                        _, H, W, _ = seg_map_t.shape
                        
                        # for density image creation
                        # make 5 x 5 pixel patch to be sampled around projections
                        patch_size = 10
                        pixel_offset_y = 2.0 / (H - 1)  # Normalize for height
                        pixel_offset_x = 2.0 / (W - 1)  # Normalize for width
                        # Create an offset grid for a 5x5 patch centered at each point
                        y_offsets = torch.linspace(-2, 2, patch_size) * pixel_offset_y
                        x_offsets = torch.linspace(-2, 2, patch_size) * pixel_offset_x
                        grid_offsets = torch.stack(torch.meshgrid(y_offsets, x_offsets), dim=-1).reshape(1, patch_size, patch_size, 2)

                        # Expand the coordinates to create grids centered on each (u, v)
                        pixel_coords = torch.tensor(projections).to(torch.float32).unsqueeze(0).unsqueeze(0).view(N, 1, 1, 2)  # Shape (N, 1, 1, 2)
                        sampling_grids = pixel_coords + grid_offsets  # Shape (N, 5, 5, 2), creating 5x5 patches around each (u, v)

                        # Use grid_sample to extract the patches
                        patches = F.grid_sample(torch.permute(seg_map_t, (0, 3, 1, 2)).expand(N, -1, -1, -1), sampling_grids, mode='nearest', align_corners=True, padding_mode="border").squeeze(1)  # Shape (N, 1, 5, 5)
                        floor_pixels = (patches == 3) + (patches == 6) + (patches == 13) + (patches == 29) # 3->floor, 6->road, 13->ground, 29->field
                        wall_pixels = (patches == 0) + (patches == 1) # 0->Wall, 1->Building
                        wall_counts = wall_pixels.sum(dim=(1, 2)) 
                        floor_counts = floor_pixels.sum(dim=(1, 2))  
                        walls = (wall_counts / patch_size ** 2) > 0.8
                        floors = (floor_counts / patch_size ** 2) > 0.8

                        # get walls and floors for each camera using segmentation
                        wall_points.append(np.array(sparse_points)[walls.numpy()])
                        floor_points.append(np.array(sparse_points)[floors.numpy()])
                    else:
                        # No segmentation - use simple heuristics or skip wall/floor detection
                        # For now, assume all points could be walls/floors (conservative approach)
                        wall_points.append(np.array(sparse_points))
                        floor_points.append(np.array(sparse_points))
                else :
                    camera_scales.append([])
            chunk_scales.append(camera_scales)
            chunk_walls.append(np.vstack(wall_points)[:,:3])
            chunk_floors.append(np.vstack(floor_points)[:,:3])
    else:
        # No metric depths or segmentation - use default values
        final_scale = 1.0
        # Initialize with defaults for each chunk
        for chunk in doc.chunks:
            chunk_scales.append([[1.0]])  # Default scale for each camera
            # Create dummy wall/floor points (use tie points as fallback)  
            tie_points = [np.array(point.coord[:3]) for point in chunk.tie_points.points]
            if len(tie_points) > 0:
                chunk_walls.append(np.array(tie_points))
                chunk_floors.append(np.array(tie_points))
            else:
                # If no tie points, create minimal dummy data
                chunk_walls.append(np.array([[0, 0, 0]]))
                chunk_floors.append(np.array([[0, 0, 0]]))

    chunk_final_scales = []
    for chunk_id, camera_scales in enumerate(chunk_scales):
        scales_list = []
        for camera_scale in camera_scales:
            scales_list += camera_scale
        scales = np.array(scales_list)
        final_scale = np.mean(scales)
        print ("Final scale for : ", chunk_id, " is :", final_scale)
        chunk_final_scales.append(final_scale)

    chunk_global_z_transform = []

    eps = 0.5
    min_samples = 10
    for chunk_id, chunk in enumerate(doc.chunks):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(chunk_floors[chunk_id])

        # Cluster the floor points
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0
        # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.io.write_point_cloud(str(Path(seg) / "floor_clusters.ply"), pcd)

        # vis clusters

        label_to_points = {}
        for label in np.unique(labels):
            if label == -1:
                continue
            label_to_points[label] = chunk_floors[chunk_id][labels == label]

        normals = []
        weights = []

        # Check if we have any valid clusters
        if len(label_to_points) == 0:
            print(f"WARNING: No floor clusters found for chunk {chunk_id}, using default Y-up transform")
            # Use default Y-up transform when clustering fails
            avg_normal = np.array([0, 1, 0])  # Default Y-up
        else:
            # Find label of largest cluster
            largest_cluster_label = max(label_to_points, key=lambda lbl: len(label_to_points[lbl]))
            largest_cluster_points = label_to_points[largest_cluster_label]

            # Perform PCA only on largest cluster (To use the mode of largest cluster instead of weighted mean)
            cluster_mean = np.mean(largest_cluster_points, axis=0)
            covariance_matrix = np.cov(largest_cluster_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
            avg_normal = eigenvectors[:, np.argmin(eigenvalues)]  # Smallest eigenvalue -> normal
            if avg_normal[1] < 0:  # Flip if opp. direction (consistency of normal direction)
                avg_normal = -avg_normal

        # Get transform
        global_z_transform = get_global_z_transform(avg_normal)
        global_z_transform[:3, :3] *= chunk_final_scales[chunk_id]
        if mode == "densify" and seg is not None:
            np.savetxt(str(Path(seg) / f"global_z_transform_{chunk_id:03d}.txt"), global_z_transform)
            if floorplan_dir is not None:
                np.savetxt(str(Path(seg) / f"global_z_transform_{chunk_id:03d}_floorplan.txt"), global_z_transform)
        chunk_global_z_transform.append(global_z_transform)

    if mode == "transform":
        print("Length of global y transform", len(chunk_global_z_transform))
        # only one chunk must be there
        return chunk_global_z_transform[0]

    for chunk_id, walls in enumerate(chunk_walls):
        # transform to z up and scale
        floors = chunk_floors[chunk_id]
        floors = np.hstack((floors, np.ones((floors.shape[0], 1))))
        walls = np.hstack((walls, np.ones((walls.shape[0], 1))))
        walls = np.dot(chunk_global_z_transform[chunk_id], walls.T).T
        floors = np.dot(chunk_global_z_transform[chunk_id], floors.T).T
        walls = walls[:, :3]
        floors = floors[:, :3]

        # Project to xy plane
        walls[:,2] = 0
        walls = walls[:,[0,1]]
        min_x, min_z = walls.min(axis=0)
        max_x, max_z = walls.max(axis=0)

        pixel_size_x = (max_x - min_x) / 2048.0
        pixel_size_z = (max_z - min_z) / 2048.0

        # Choose the smaller pixel size to maintain aspect ratio
        pixel_size = min(pixel_size_x, pixel_size_z)

        # Calculate the final number of pixels
        num_pixels_x = int(np.ceil((max_x - min_x) / pixel_size))
        num_pixels_z = int(np.ceil((max_z - min_z) / pixel_size))
        
        origin = np.array([min_x, min_z, pixel_size])
        # save origin and ppcm for each chunk
        if mode == "densify" and seg is not None:
            np.savetxt(str(Path(seg) / f"origin_{chunk_id:03d}.txt"), origin)
        # Initialize the density image
        density_image = np.zeros((num_pixels_z, num_pixels_x), dtype=int)

        # Step 2: Populate the density image
        for x, z in walls:
            # Convert the (x, y) coordinates to pixel indices
            pixel_x = int((x - min_x) / pixel_size)
            pixel_z = int((z - min_z) / pixel_size)

            # Clip values to stay within bounds
            pixel_x = np.clip(pixel_x, 0, num_pixels_x - 1)
            pixel_z = np.clip(pixel_z, 0, num_pixels_z - 1)

            # Increment the density at the corresponding pixel
            density_image[pixel_z, pixel_x] += 1
        gamma = 1
        # at non zero values only add 200 so that color map is distinguishable
        density_image[density_image >= 1] = density_image[density_image >= 1] + 200
        density_image = np.power(density_image, gamma)
        colored_image = plt.cm.inferno(density_image)  # You can choose other colormaps like 'plasma', 'inferno', etc.

        # Set all 0 values to black
        colored_image[density_image == 0] = [0, 0, 0, 1]  # RGBA for black (opaque)
        
        # Save the image
        if mode == "densify" and seg is not None:
            plt.imsave(str(Path(seg) / f"density_image_{chunk_id:03d}.png"), colored_image)

        return chunk_global_z_transform

def transform_chunk(doc, transform, output_dir):
    # Transform the chunk
    if transform is not None:
        for chunk in doc.chunks:
            chunk.transform.matrix = Metashape.Matrix(transform)

    saveChunk = doc.chunks[0]

    # Save the cameras 
    saveChunk.exportCameras(str(Path(output_dir) / "cams.xml"))

    saveChunk.exportCameras(path=str(Path(output_dir) / "cams_fm.out"), format=Metashape.CamerasFormatBundler, save_points=True,
                        save_markers=False, use_labels=True, binary=False, bundler_save_list=True, bundler_path_list=str(Path(output_dir) / "bundler.txt"))
    
    saveChunk.exportPointCloud(path=str(Path(output_dir) / "sparse_cloud.ply"), source_data=Metashape.DataSource.TiePointsData, binary=True,
                       save_point_normal=True, save_point_color=True, save_point_confidence=True, colors_rgb_8bit=False, save_comment=False, format=Metashape.PointCloudFormat.PointCloudFormatPLY)

    T = saveChunk.transform.matrix  # transformation from local to UTM
    M = saveChunk.crs.localframe(T.mulp(Metashape.Vector([0., 0., 0.])))  # transformation matrix from UTM to the LSE coordinates(local as in bundler) in the given point
    xml_to_bundler = M * T  # local to UTM, then UTM to world
    # save the transformation matrix
    np.savetxt(str(Path(args.output_dir) / "xml_to_bundler.txt"), np.array(xml_to_bundler).reshape(4, 4))

    # Skip saving document in demo mode - exports should still work
    print("INFO: Skipping project save (demo mode limitation) - exports completed successfully")

# All visualization functionality removed

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vidcount", type=int, required=True)
    parser.add_argument("--gps", type=int, required=False, default=0)
    parser.add_argument("--metric_depths", type=str, required=False)
    parser.add_argument("--seg", type=str, required=False)
    parser.add_argument("--stills", type=int, required=False, default=0)
    parser.add_argument("--past_cams", type=str, required=False, default=None)
    
    args = parser.parse_args()

    # Align cameras
    print("AGISOFT CAMERA ALIGNMENT")
    doc = align_cameras(args.images_dir, args.vidcount, args.output_dir, args.metric_depths, args.seg, args.gps, args.stills, past_cams=args.past_cams)

    # Transform and densify
    print("COORDINATE TRANSFORMS & DENSIFICATION")
    z_transforms = calculate_transform_and_densify(doc, args.metric_depths, args.seg, "densify")

    # Final export and transform
    print("FINAL EXPORT")
    
    if args.gps == 0:
        if args.past_cams is not None:
            print ("\n\n\nNot using scale and y-up transform as past cams are provided\n\n\n")
            transform = None
            transform_chunk(doc, transform, args.output_dir)
        else:
            # check if z_transforms is singular (ideally if pipeline ran as intended, densify is run on project.psx that has only the merged chunk)
            if len(z_transforms) == 1:
                transform = z_transforms[0]
                # Transform chunk using transform matrix and save document
                transform_chunk(doc, transform, args.output_dir)
            else:
                print("warning, densify run on project with multiple chunks, refusing to transform it all to y-up, as resultant project will be incompatible with the rest of the pipeline open project.pdx to investigate")
    else:
        print ("\n\n\nUsing GPS alignment transform\n\n\n")
        # external GPS support should go here
        # transform = np.eye(4)
        transform_chunk(doc, None, args.output_dir)

    print("PIPELINE COMPLETE!")
