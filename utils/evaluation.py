import argparse
import os
import glob
import pprint
import pickle
from pathlib import Path
import io
import typing
import copy

import numpy as np
import torch
import pytorch3d
import pytorch3d.loss
import open3d
open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

from utils import general_utils


def normalize_open3d_pc(open3d_pc, diameter):
    """Normalizes an open3D pointcloud to be within a specific diameter sphere.

    Args:
        open3d_pc (open3d.geometry.PointCloud): Pointcloud to normalize.
        diameter (float): Diameter of sphere.

    Returns:
        open3d.geometry.PointCloud: Normalized pointcloud.
    """
    normalized_pc = copy.deepcopy(open3d_pc)
    max_width = np.amax(normalized_pc.get_max_bound()-normalized_pc.get_min_bound())
    normalized_pc = normalized_pc.scale((1/max_width)*diameter, np.array([[0.0,0.0,0.0]]).T)
    return normalized_pc


def get_open3d_pc_from_mesh_path(mesh_path, n_points_sample):
    """Obtains the open3d pointcloud from a path pointing to a mesh.

    Args:
        mesh_path (str): Filepath of mesh.
        n_points_sample (int): number of points to sample

    Returns:
        open3d.geometry.PointCloud: Open3D pointcloud object.
    """
    mesh = open3d.io.read_triangle_mesh(mesh_path)
    if np.asarray(mesh.vertices).shape[0] == 0:
        return None
    else:
        mesh_pc = mesh.sample_points_uniformly(number_of_points=n_points_sample)
        return mesh_pc


def compute_f_score(pr, gt, th=0.05):
    """Computes the f-score between two pointclouds

    Args:
        pr (open3d.geometry.PointCloud): first pointcloud
        gt (open3d.geometry.PointCloud): second pointcloud
        th (float, optional): f-score threshold. Defaults to 0.05.

    Returns:
        float: the f-score
    """
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))
        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore


def compute_chamfer_L2(rec_pc_torch, gt_pc_torch):
    """Computes the chamfer-L2 distance between two pointclouds

    Args:
        rec_pc_torch (torch.tensor): first pointcloud
        gt_pc_torch (torch.tensor): second pointcloud

    Returns:
        float: the chamfer distance
    """
    chamfer_dist_pytorch = pytorch3d.loss.chamfer_distance(rec_pc_torch, gt_pc_torch)
    chamfer_dist_pytorch = chamfer_dist_pytorch[0].item() * 1000
    return chamfer_dist_pytorch


def evaluate(rec_obj_path, gt_obj_path, device):
    """Evaluates two reconstructions, to obtain the f-score and chamfer-l2 distance.

    Args:
        rec_obj_path (str): Filepath to the first .obj mesh reconstruction.
        gt_obj_path (str): Filepath to the second .obj mesh reconstruction.
        device (torch.device): PyTorch device to perform computations on.

    Returns:
        dict: Dictionary with results for the f-score and chamfer-l2 distance.
    """
    open3d_n_points_sample = 10000
    rec_pc_open3d = get_open3d_pc_from_mesh_path(rec_obj_path, open3d_n_points_sample)
    gt_pc_open3d = get_open3d_pc_from_mesh_path(gt_obj_path, open3d_n_points_sample)
    rec_pc_open3d_dia_2 = normalize_open3d_pc(rec_pc_open3d, 2)
    rec_pc_torch_dia_2 = torch.tensor(np.asarray(rec_pc_open3d_dia_2.points), dtype=torch.float).unsqueeze(0).to(device)
    gt_pc_open3d_dia_2 = normalize_open3d_pc(gt_pc_open3d, 2)
    gt_pc_torch_dia_2 = torch.tensor(np.asarray(gt_pc_open3d_dia_2.points), dtype=torch.float).unsqueeze(0).to(device)

    instance_record = {}
    f_score = compute_f_score(rec_pc_open3d_dia_2, gt_pc_open3d_dia_2, th=0.05)
    instance_record["f_score"] = f_score
    chamfer_L2 = compute_chamfer_L2(rec_pc_torch_dia_2, gt_pc_torch_dia_2)
    instance_record["chamfer_L2"] = chamfer_L2

    return instance_record
