import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render, integrate
import random
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
from PIL import Image
import cv2
import time



@torch.no_grad()
def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=False):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    if return_color:
        final_color = torch.ones((points.shape[0], 3), dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        i = 0
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
                ret = integrate(points, view, gaussians, pipeline, background, kernel_size=kernel_size)
                alpha_integrated = ret["alpha_integrated"]
                if return_color:
                    color_integrated = ret["color_integrated"]    
                    final_color = torch.where((alpha_integrated < final_alpha).reshape(-1, 1), color_integrated, final_color)
                final_alpha = torch.min(final_alpha, alpha_integrated)

        alpha = 1 - final_alpha
    if return_color:
        return alpha, final_color
    return alpha


@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, filter_mesh : bool, texture_mesh : bool):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "fusion")

    makedirs(render_path, exist_ok=True)
    
    # generate tetra points here
    points, points_scale = gaussians.get_tetra_points()
   
    # load cell if exists
    if os.path.exists(os.path.join(render_path, "cells.pt")):
        print("load existing cells")
        cells = torch.load(os.path.join(render_path, "cells.pt"))
    else:
        # create cell and save cells
        print("create cells and save")
        start_time = time.time()
        cells = cpp.triangulate(points)
        end_time = time.time()

        elapsed_time = end_time - start_time
        print("ELAPSED TIME: ", elapsed_time)
        # we should filter the cell if it is larger than the gaussians
        torch.save(cells, os.path.join(render_path, "cells.pt"))
    
    # evaluate alpha
    alpha = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size)

    vertices = points.cuda()[None]
    tets = cells.cuda().long()


    def alpha_to_sdf(alpha):    
        sdf = alpha - 0.5
        sdf = sdf[None]
        return sdf
    
    sdf = alpha_to_sdf(alpha)
    
    torch.cuda.empty_cache()
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf, points_scale[None])
    torch.cuda.empty_cache()
    
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    
    faces=faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
        
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale
        
    n_binary_steps = 10

    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        alpha = evaluage_alpha(mid_points, views, gaussians, pipeline, background, kernel_size)
        mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)
        
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
    
        points = (left_points + right_points) / 2
         
    if texture_mesh:
        _, color = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, return_color=True)
        vertex_colors=(color.cpu().numpy() * 255).astype(np.uint8)
    else:
        vertex_colors=None

    mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, vertex_colors=vertex_colors, process=False)
        
    # filter
    if filter_mesh:
        mask = (distance <= scale).cpu().numpy()
        face_mask = mask[faces].all(axis=1)
        mesh.update_vertices(mask)
        mesh.update_faces(face_mask)
        
    mesh.export(os.path.join(render_path, f"mesh_binary_search_{n_binary_steps}.ply"))

    # linear interpolation
    # right_sdf *= -1
    # points = (left_points * left_sdf + right_points * right_sdf) / (left_sdf + right_sdf)
    # mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces)
    # mesh.export(os.path.join(render_path, f"mesh_binary_search_interp.ply"))
    

def calculate_iou(image1, image2):
    binary_image1 = (image1 > 0).astype(np.uint8)
    binary_image2 = (image2 > 0).astype(np.uint8)

    intersection = np.sum(binary_image1 & binary_image2)
    union = np.sum(binary_image1 | binary_image2)

    iou = intersection / union if union != 0 else 0
    return iou

def align_images(img1, img2, matcher):
    keypoints1, descriptors1 = matcher.detectAndCompute(img1, None)
    keypoints2, descriptors2 = matcher.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # extracting keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # computing homography
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

    # applying transformation
    aligned_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
    
    return aligned_img



def get_most_influential_images(input_folder, matcher, iou_threshold=0.5):
    image_files = sorted(os.listdir(input_folder))
    most_influential_images = []

    i = 0
    while i < len(image_files): 
        ref_image_path = os.path.join(input_folder, image_files[i])
        ref_image = Image.open(ref_image_path).convert('L')  # Converte in scala di grigi
        ref_array = np.array(ref_image)

        min_iou = float('inf')
        min_iou_image_name = None

        
        for j in range(i + 1, len(image_files)):
            current_image_path = os.path.join(input_folder, image_files[j])
            current_image = Image.open(current_image_path).convert('L')  # Converte in scala di grigi
            current_array = np.array(current_image)

            aligned_image = align_images(ref_array, current_array, matcher)
            iou = calculate_iou(ref_array, aligned_image)

            if iou <= iou_threshold:
                min_iou_image_name = image_files[j]
                min_iou = iou
                i = j + 1  
                break

        if min_iou_image_name is None:
            break

        most_influential_images.append(min_iou_image_name)

    return most_influential_images


def extract_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams, filter_mesh : bool, texture_mesh : bool, feature_matcher : str, colmap_path : str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        
        cams = scene.getTrainCameras()

        if feature_matcher is not None and colmap_path is not None:
            most_important_images = get_most_influential_images(colmap_path, feature_matcher)
            most_important_images = [os.path.splitext(image)[0] for image in most_important_images]
            most_important_cams = []

            for cam in cams:
                image = cam.image_name
                if image in most_important_images:
                        most_important_cams.append(cam)

            cams = most_important_cams
        
        marching_tetrahedra_with_binary_search(dataset.model_path, "test", iteration, cams, gaussians, pipeline, background, kernel_size, filter_mesh, texture_mesh)


def create_feature_matcher(feature_matcher_name):
    if feature_matcher_name == "akaze":
        return cv2.AKAZE_create()
    elif feature_matcher_name == "orb":
        return cv2.ORB_create(nfeatures=1000) 
    elif feature_matcher_name == "brisk":
        return cv2.BRISK_create()
    else:
        raise ValueError(f"Matcher not supported: {feature_matcher_name}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    
    parser.add_argument(
        "--feature_matcher",
        choices=["akaze", "orb", "brisk", "sift"],
        default=None,
        help="Specify the type of feature matcher to use (akaze, orb, brisk)"
    )

    parser.add_argument(
        "--colmap_path",
        default=None,
        type=str,
        help="Path to the COLMAP dataset"
    )
    
    args = get_combined_args(parser)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    feature_matcher = None

    if args.feature_matcher is not None:
        feature_matcher = create_feature_matcher(args.feature_matcher)

    extract_mesh(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.filter_mesh,
        args.texture_mesh,
        feature_matcher,
        args.colmap_path # could be None
    )
