from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
from pathlib import Path
import numpy as np

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.utils.io import get_keypoints, get_matches
import pairs_from_retrieval_with_score

import json
import argparse


feature_conf = extract_features.confs["disk"]
matcher_conf = match_features.confs["disk+lightglue"]
retrieval_conf = extract_features.confs["netvlad"]

def get_avg_temporally_adjacent_camera_distance(camera_centers):
    keys = list(camera_centers.keys())
    n = len(keys)
    cam_dists = []
    for i in range(n-1):
        dist = np.linalg.norm(np.array(camera_centers[keys[i]]["w2c"])[:3,3] - np.array(camera_centers[keys[i + 1]]["w2c"])[:3,3])
        cam_dists.append(dist)
    avg_dist = sum(cam_dists)/len(cam_dists)
    return avg_dist

def remove_non_hloc_matches(video_json, reloc_frames, vid_id):
    for key_idx in list(video_json["keyframes"].keys()):
        video_json["keyframes"][key_idx]["matches"] = [name for name in video_json["keyframes"][key_idx]["matches"] 
                                                       if (name + ".jpg") in reloc_frames or vid_from_name(name) == vid_id]

def estimate_scale(C1, C2):
    distances_C1 = np.linalg.norm(C1[1:] - C1[:-1], axis=1)
    distances_C2 = np.linalg.norm(C2[1:] - C2[:-1], axis=1)

    # Estimate the scale as the ratio of the average distances between cameras
    scale_estimate = np.mean(distances_C2) / np.mean(distances_C1)
    
    return scale_estimate

def ransac_inlier_selection(C1, C2, normalized_threshold=0.1):
    """
    Apply RANSAC to remove outlier camera pairs.
    """   
    scale_estimate = estimate_scale(C1, C2)
    
    # Normalize the points by scaling C1
    C1_normalized = C1 * scale_estimate
    
    # RANSAC regressor based on the normalized points
    ransac = RANSACRegressor(residual_threshold=normalized_threshold, max_trials=1000)
    
    # Fit RANSAC on the normalized translation vectors between the two sets of points
    ransac.fit(C1_normalized, C2)
    print("Ransac on centr", C1_normalized.shape, C2.shape)

    # Get inliers mask
    inliers_mask = ransac.inlier_mask_

    # Filter the inlier camera centers
    inlier_C1 = C1[inliers_mask]
    inlier_C2 = C2[inliers_mask]
    
    return inlier_C1, inlier_C2, scale_estimate

def compute_similarity_transform(C1, C2):
    """
    Compute the similarity transformation (R, T, s) that aligns C1 to C2.
    This includes scale, rotation, and translation.
    """
    # Compute centroids of both sets
    centroid_C1 = np.mean(C1, axis=0)
    centroid_C2 = np.mean(C2, axis=0)
    
    # Center the points
    C1_centered = C1 - centroid_C1
    C2_centered = C2 - centroid_C2
    
    # Compute the covariance matrix
    # H = C1_centered.T @ C2_centered
    H = (C1_centered.T @ C2_centered) / C1_centered.shape[0]
    
    # Compute the Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(H)

    # identity matrix used for fixing reflections
    E = np.eye(C2_centered.shape[1]).repeat(1, 1)
    R_test = U @ V.T
    E[-1, -1] = np.linalg.det(R_test)
    
    # Compute the optimal rotation
    # R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))
    # R = (U @ E) @ V.T
    R = V.T @ U.T
    print("R", R)
    # R = (U @ E) @ V.T
    trace_ES = (np.diagonal(np.eye(3)) * S).sum()
    Xcov = (C1_centered * C1_centered).sum((0,1)) / C1_centered.shape[0]
    s = trace_ES / Xcov
    print("s", s)
    # the scaling component

    scale_num = np.sum(np.linalg.norm(C2_centered, axis=1) ** 2)
    scale_den = np.sum(np.linalg.norm(C1_centered, axis=1) ** 2)
    scale = np.sqrt(scale_num / scale_den)
    print("scale", scale)

    # Compute the optimal translation
    T = centroid_C2 - s *(R @ centroid_C1)
    # Create the transform to align C1 with C2
    # R = R.T
    # T = -(T @ R)
    # Create the 4x4 transformation matrix
    print("R", R)
    transform = np.eye(4)
    transform[:3, :3] = s * R  # Apply the scale to the rotation matrix
    transform[:3, 3] = T
    
    return transform, s

def apply_similarity_transform(similarity_transform, camera_centers):
    """
    Apply the similarity transform to a list of camera centers.
    Args:
    - similarity_transform: 4x4 similarity transformation matrix.
    - camera_centers: List of 3D camera centers.

    Returns:
    - transformed_centers: List of transformed 3D camera centers.
    """
    transformed_centers = []
    
    for center in camera_centers:
        # Convert to homogeneous coordinates
        homogeneous_center = np.append(center, 1)  # [x, y, z, 1]
        
        # Apply the similarity transformation
        transformed_center_homogeneous = similarity_transform @ homogeneous_center
        
        # Extract the transformed 3D coordinates
        transformed_center = transformed_center_homogeneous[:3]
        
        transformed_centers.append(transformed_center)

    return np.array(transformed_centers)
    
def prep_img_list(image_list):
    image_dir = Path(image_list[0]).parent
    image_list = [str(Path(image).relative_to(image_dir)) for image in image_list]
    return image_list, image_dir

def get_image_lists(image_dir):
    image_dir = Path(image_dir)
    image_lists = []

    image_list = list(image_dir.glob(f"OLD_*.jpg"))
    if len(image_list) > 0:
        image_lists.append(image_list)
    
    vid_num = 1
    while True:
        prefix = f"{vid_num:03d}_{vid_num:03d}"
        image_list = list(image_dir.glob(f"{prefix}*.jpg"))
        if len(image_list) == 0:
            break
        image_lists.append(image_list)
        vid_num += 1
    return image_lists

def save_features(image_list, output_dir):
    image_list, image_dir = prep_img_list(image_list)
    feature_conf = extract_features.confs["disk"]
    features = output_dir / "features.h5"
    retrieval_path = extract_features.main(retrieval_conf, image_dir, output_dir)
    return retrieval_path

def relocalize(image1_list, image2_list, retrieval_path, output_dir):
    image1_list, images1_path = prep_img_list(image1_list)
    image2_list, images2_path = prep_img_list(image2_list)

    reloc_pairs = output_dir / "pairs-reloc.txt"
    features = output_dir / "features.h5"
    matches = output_dir / "matches.h5"

    # 1. Compute the pairs
    pairs_from_retrieval_with_score.main(retrieval_path, reloc_pairs, num_matched=10, query_list=image1_list, db_list=image2_list)

def vid_from_name(name):
    if name[:4] == "OLD_":
        return 0
    
    return int(name.split("_")[0])

def save_reloc_pairs(match_thresh, min_pairs, hloc_dir, json_dir):
    reloc_pairs = hloc_dir / "pairs-reloc.txt"
    with open(str(reloc_pairs), "r") as f:
        pairs = [p.split() for p in f.readlines()]
    reloc_scores = hloc_dir / "pairs-reloc.score"
    with open(reloc_scores, "r") as f:
        scores = [float(s) for s in f.readlines()]

    match_dict = {}

    vid_match_dicts = {}
    for pair, score in zip(pairs, scores):
        name0, name1 = pair
        vid0 = vid_from_name(name0)
        vid1 = vid_from_name(name1)
        if (vid0, vid1) not in vid_match_dicts:
            vid_match_dicts[vid0, vid1] = {}
        vid_match_dicts[vid0, vid1][name0, name1] = score

    # Filter matches into reloc_frames to be added into agisoft
    reloc_frames = {}
    # if file exists read in the old reloc_frames
    if (json_dir / "reloc_frames.json").exists():
        with open(json_dir / "reloc_frames.json", "r") as f:
            reloc_frames = json.load(f)
        reloc_frames = {int(k): set(v) for k, v in reloc_frames.items()}     
    vid_match_dicts_filtered = {}
    vid_max_matches_dict = {}
    for (vid0, vid1) in vid_match_dicts:
        match_dict = sorted(vid_match_dicts[vid0, vid1].items(), key=lambda x: -x[1])
        max_matches_dict = {}
        match_dict_filtered = {}
        for idx, (pair, num_matches) in enumerate(match_dict):
            name0, name1 = pair
            if vid0 != 0 and vid1 != 0:
                if (num_matches > match_thresh) or (idx < min_pairs):
                    if name0 not in max_matches_dict:
                        max_matches_dict[name0] = (name1, num_matches)
                    else:
                        if num_matches > max_matches_dict[name0][1]:
                            max_matches_dict[name0] = (name1, num_matches)

            if (num_matches > match_thresh) or (idx < min_pairs):
                match_dict_filtered[name0, name1] = num_matches
                if vid0 not in reloc_frames:
                    reloc_frames[vid0] = set()
                if vid1 not in reloc_frames:
                    reloc_frames[vid1] = set()
                # add in old frames as they are not gonna be filtered
                if vid0 == 0 or vid1 == 0:
                    reloc_frames[vid1].add(name0)
                    reloc_frames[vid0].add(name1)
                
        vid_match_dicts_filtered[vid0, vid1] = match_dict_filtered
        if vid0 != 0 and vid1 != 0:
            vid_max_matches_dict[vid0, vid1] = max_matches_dict
    
    # Prepare jsons and name to idx matches
    name_to_idx = {}
    vid_jsons = {}
    past_frames = set()
    for vid in reloc_frames:
        past_frames.update(set(reloc_frames[vid]))
    past_frames = list(past_frames)
    past_frames = [frame for frame in past_frames if frame[:4] == "OLD_"]

    # if len(past_frames) != 0:
    #     old_vid_json = {"keyframes": {}}
    #     for idx, frame in enumerate(past_frames):
    #         old_vid_json["keyframes"][str(idx)] = {"name": Path(frame).stem, "matches": []}

    # Create video json for old vid or read video json for new vids
    for vid in reloc_frames:
        if vid == 0: # OLD video
            vid_json = {"keyframes": {}}
            for idx, frame in enumerate(past_frames):
                vid_json["keyframes"][str(idx)] = {"name": Path(frame).stem, "matches": []}

            name_to_idx[vid] = {vid_json["keyframes"][idx]["name"]: idx for idx in vid_json["keyframes"]}
            vid_jsons[vid] = vid_json
            
        else:
            vid_json_path = json_dir / f"{vid:03d}_{vid:03d}.json"
            with open(vid_json_path, "r") as f:
                vid_json = json.load(f)
            name_to_idx[vid] = {vid_json["keyframes"][idx]["name"]: idx for idx in vid_json["keyframes"]}
            vid_jsons[vid] = vid_json

    # Add the matches to the json files after running RANSAC
    for (vid0, vid1) in vid_max_matches_dict:
        C0, C1 = [], []
        used_idx = set()
        vid0_json_path = json_dir / f"{vid0:03d}_{vid0:03d}.json"
        vid1_json_path = json_dir / f"{vid1:03d}_{vid1:03d}.json"
        for name0, (name1, _ ) in vid_max_matches_dict[vid0, vid1].items():
            idx0 = name_to_idx[vid0][Path(name0).stem]
            idx1 = name_to_idx[vid1][Path(name1).stem]
            if idx1 in used_idx:
                continue
            used_idx.add(idx1)
            C0.append(np.array(vid_jsons[vid0]["keyframes"][idx0]["w2c"])[:3,3])
            C1.append(np.array(vid_jsons[vid1]["keyframes"][idx1]["w2c"])[:3,3])
        if len(C0) < 9 and len(C1) < 9:
            print("Not enough points to run RANSAC")
            print("got", len(C0), len(C1))
            transform_matrix = None
            continue
        inlier_C0, inlier_C1, scale_estimate = ransac_inlier_selection(np.array(C0), np.array(C1), normalized_threshold=0.05)
        print("ransac scale estimate", scale_estimate)
        transform_matrix, scale = compute_similarity_transform(inlier_C0, inlier_C1)
        print("Stella aligned", vid0, vid1)
        print("Similarity Transformation Matrix:\n", transform_matrix)
        print("Scale Factor:", scale)
        # save transform matrix as npy
        np.save(json_dir / f"{vid0:03d}_{vid1:03d}_transform.npy", transform_matrix)

    for (vid0, vid1) in vid_match_dicts_filtered:
        match_dict_filtered = vid_match_dicts_filtered[vid0, vid1]
        if vid0 != 0 and vid1 != 0:
            if transform_matrix is not None:
                avg_temporal_cam_dist = get_avg_temporally_adjacent_camera_distance(vid_jsons[vid1]["keyframes"])
                for pair in match_dict_filtered:
                    name0, name1 = pair
                    stem0 = Path(name0).stem
                    stem1 = Path(name1).stem
                    idx0 = name_to_idx[vid0][stem0]
                    idx1 = name_to_idx[vid1][stem1]
                    C0 = np.array(vid_jsons[vid0]["keyframes"][idx0]["w2c"])
                    C1 = np.array(vid_jsons[vid1]["keyframes"][idx1]["w2c"])
                    C0_shifted = transform_matrix @ C0
                    cam_dist = np.linalg.norm(C0_shifted[:3,3] - C1[:3,3])
                    if cam_dist < 2*avg_temporal_cam_dist:
                        reloc_frames[vid0].add(name1)
                        reloc_frames[vid1].add(name0)
                        if stem1 not in vid_jsons[vid0]["keyframes"][idx0]["matches"]:
                            vid_jsons[vid0]["keyframes"][idx0]["matches"].append(stem1)
                        if stem0 not in vid_jsons[vid1]["keyframes"][idx1]["matches"]:
                            vid_jsons[vid1]["keyframes"][idx1]["matches"].append(stem0)
                        print("allowed the cameras", name0, name1)
                    else:
                        print("disallowed", name0, name1)
        else:
            for pair in match_dict_filtered:
                name0, name1 = pair
                stem0 = Path(name0).stem
                stem1 = Path(name1).stem
                idx0 = name_to_idx[vid0][stem0]
                idx1 = name_to_idx[vid1][stem1]
                reloc_frames[vid0].add(name1)
                reloc_frames[vid1].add(name0)
                vid_jsons[vid0]["keyframes"][idx0]["matches"].append(stem1)
                vid_jsons[vid1]["keyframes"][idx1]["matches"].append(stem0)

    for vid in reloc_frames:
        reloc_frames[vid] = list(reloc_frames[vid])
    
    with open(json_dir / "reloc_frames.json", "w") as f:
        json.dump(reloc_frames, f, indent=4)
    
    # check matches so that reruns don't leave in older matches before saving
    for vid, vid_json in vid_jsons.items():
        if vid != 0:
            remove_non_hloc_matches(vid_json, reloc_frames[vid], vid)
        with open(json_dir / f"{vid:03d}_{vid:03d}.json", "w") as f:
            json.dump(vid_json, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--hloc_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--match_thresh", type=float, default=0.45)
    parser.add_argument("--min_pairs", type=int, default=20)

    args = parser.parse_args()
    print (args)

    image_dir = Path(args.image_dir)
    hloc_dir = Path(args.hloc_dir)
    json_dir = Path(args.json_dir)

    image_lists = get_image_lists(image_dir)

    for image_list in image_lists:
        retrieval_path = save_features(image_list, hloc_dir)
    
    for i in range(len(image_lists) - 1):
        for j in range(i + 1, len(image_lists)):
            relocalize(image_lists[i], image_lists[j], retrieval_path, hloc_dir)
            # out_dir.mkdir(exist_ok=True, parents=True)
            save_reloc_pairs(args.match_thresh, args.min_pairs, hloc_dir, json_dir)