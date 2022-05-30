import yaml
import glob
import io
import os
import copy
import pprint
import random

import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    HardFlatShader,
    BlendParams,
    softmax_rgb_blend
)
from tqdm.autonotebook import tqdm
from PIL import Image
import trimesh
from sklearn.neighbors import NearestNeighbors


def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool, optional): Whether to use default path. Defaults to None.
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def load_untextured_mesh(mesh_path, device):
    """Loads a mesh from a file path, into a Pytorch3D mesh with no texture.
       Based on https://github.com/facebookresearch/pytorch3d/issues/51

    Args:
        mesh_path (str): mesh path location
        device (torch.device): pytorch device

    Returns:
        Mesh: pytorch3d mesh object.
    """
    mesh = load_objs_as_meshes([mesh_path], device=device, load_textures=False)
    verts, faces_idx, _ = load_obj(mesh_path)
    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None] # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(device))
    mesh_no_texture = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
        )
    return mesh_no_texture


# for rendering a single image
def render_mesh(mesh, R, T, device, img_size=512, silhouette=False, custom_lights="",
                differentiable=True, return_renderer_only=False, black_bg=False):
    """Renders images from a mesh.

    Args:
        mesh (Mesh): pytorch3d mesh object to render
        R (torch.tensor): rotation matrix for camera when rendering
        T (torch.tensor): translation matrix for camera when rendering
        device (torch.device): pytorch device
        img_size (int, optional): size of image to render. Defaults to 512.
        silhouette (bool, optional): Render sillhouete only. Defaults to False.
        custom_lights (str, optional): lights to use. Defaults to "".
        differentiable (bool, optional): If rendering should be differentiable. Defaults to True.
        return_renderer_only (bool, optional): If only the renderer should be returned. Defaults to False.
        black_bg (bool, optional): If a black background should be used. Defaults to False.

    Returns:
        torch.tensor: rendered image.
    """
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    if silhouette:
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        raster_settings = RasterizationSettings(
            image_size=img_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100, 
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
    else:
        raster_settings = RasterizationSettings(
            image_size=img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        if custom_lights == "":
            lights = PointLights(device=device, location=[[0.0, 5.0, -10.0]])
        elif type(custom_lights) == str and custom_lights == "ambient":
            ambient = 0.5
            diffuse = 0
            specular = 0
            lights = PointLights(device=device, ambient_color=((ambient, ambient, ambient), ), diffuse_color=((diffuse, diffuse, diffuse), ),
                                 specular_color=((specular, specular, specular), ), location=[[0.0, 5.0, -10.0]])
        else:
            lights = custom_lights

        if differentiable:
            if black_bg:
                blend_params=BlendParams(background_color = (0,0,0))
            else:
                blend_params=None

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=device, 
                    cameras=cameras,
                    lights=lights,
                    blend_params=blend_params
                )
            )
        else:
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings
                ),
                shader=HardPhongShader(
                    device=device, 
                    cameras=cameras,
                    lights=lights
                )
            )
    if return_renderer_only:
        return renderer
    else:
        rendered_images = renderer(mesh, cameras=cameras)
        return rendered_images


def rotate_verts(RT, verts):
    """Rotates vertices.

    Args:
        RT (torch.tensor): (N, 4, 4) array of extrinsic matrices
        verts (torch.tensor): (N, V, 3) array of vertex positions

    Returns:
        torch.tensor: batch of rotated vertices
    """
    singleton = False
    if RT.dim() == 2:
        assert verts.dim() == 2
        RT, verts = RT[None], verts[None]
        singleton = True

    if isinstance(verts, list):
        verts_rot = []
        for i, v in enumerate(verts):
            verts_rot.append(rotate_verts(RT[i], v))
        return verts_rot

    R = RT[:, :3, :3]
    verts_rot = verts.bmm(R.transpose(1, 2))
    if singleton:
        verts_rot = verts_rot[0]
    return verts_rot


def reflect_batch(batch, sym_plane, device):
    """Reflects a batch of vertices across a plane of symmetry

    Args:
        batch (torch.tensor): batch of vertices
        sym_plane (torch.tensor): vector orthogonal to the plane of symmetry
        device (torch.device): pytorch device

    Returns:
        torch.tensor: batch of reflected vertices
    """
    N = np.array([sym_plane])
    reflect_matrix = torch.tensor(np.eye(3) - 2*N.T@N, dtype=torch.float).to(device)
    for i in range(batch.shape[0]):
        batch[i] = batch[i] @ reflect_matrix
    return batch


def align_and_normalize_verts_original(verts, R, T, device):
    """Aligns and normalizes vertices

    Args:
        verts (torch.tensor): vertices
        R (torch.tensor): rotation matrix
        T (torch.tensor): translation matrix
        device (torch.device): gpu to run on

    Returns:
        torch.tensor: normalized vertices
    """
    # creating batch of 4x4 extrinsic matrices from rotation matrices and translation vectors
    temp = torch.tensor([0,0,0,1]).to(device)
    temp = temp.repeat(R.shape[0],1).unsqueeze(1)
    T = T.unsqueeze(2)
    P = torch.cat([R,T], 2)
    P = torch.cat([P,temp], 1)

    # changing vertices from world coordinates to camera coordinates
    P_inv = torch.inverse(P).to(device)
    aligned_verts = rotate_verts(P_inv, verts)
    aligned_verts = aligned_verts * 2
    aligned_verts = reflect_batch(aligned_verts, [1,0,0], device)
    aligned_verts = reflect_batch(aligned_verts, [0,1,0], device)

    return aligned_verts
