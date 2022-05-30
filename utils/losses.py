import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.nn import functional as F
import pytorch3d
from pytorch3d.renderer import look_at_view_transform, TexturesVertex, Textures, PointLights

from utils import general_utils


def vertex_symmetry_loss(mesh, sym_plane, device, asym_conf_scores=False, sym_bias=0.005):
    """Vertex based symmetry loss.

    Args:
        mesh (Mesh): pytorch3d mesh object
        sym_plane (torch.tensor): vector orthogonal to the plane of symmetry
        device (torch.device): pytorch device
        asym_conf_scores (bool, optional): If asymmetry confidence scores should be used. Defaults to False.
        sym_bias (float, optional): Bias term for symmetry. Defaults to 0.005.

    Raises:
        ValueError: Symmetry plane vector needs to be a unit normal vector.

    Returns:
        torch.tensor: loss value
    """
    N = np.array([sym_plane])
    if np.linalg.norm(N) != 1:
        raise ValueError("sym_plane needs to be a unit normal")

    reflect_matrix = torch.tensor(np.eye(3) - 2*N.T@N, dtype=torch.float).to(device)
    mesh_verts = mesh.verts_packed()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(mesh_verts.detach().cpu())
    sym_points = mesh_verts @ reflect_matrix
    distances, indices = nbrs.kneighbors(sym_points.detach().cpu())
    nn_dists = torch.unsqueeze(torch.sum(F.mse_loss(sym_points, torch.squeeze(mesh_verts[indices],1), reduction='none'), 1),1)

    if asym_conf_scores is not None:
        avg_sym_loss = torch.mean(torch.log(torch.div(1,asym_conf_scores))*sym_bias + ((asym_conf_scores)*nn_dists))
    else:
        avg_sym_loss = torch.mean(nn_dists)
    
    return avg_sym_loss


def vertex_symmetry_loss_batched(meshes, sym_plane, device, asym_conf_scores=None, sym_bias=0.005):
    """Batched version of the vertex symmetry loss.

    Args:
        meshes (Mesh): batch of pytorch3d mesh object
        sym_plane (torch.tensor): vector orthogonal to the plane of symmetry
        device (torch.device): pytorch device
        asym_conf_scores (bool, optional): If asymmetry confidence scores should be used. Defaults to False.
        sym_bias (float, optional): Bias term for symmetry. Defaults to 0.005.

    Returns:
        torch.tensor: loss value
    """
    total_vtx_sym_loss = 0

    for mesh in meshes:
        curr_sym_loss = vertex_symmetry_loss(mesh, sym_plane, device, asym_conf_scores=asym_conf_scores, sym_bias=sym_bias)
        total_vtx_sym_loss += curr_sym_loss

    avg_vtx_sym_loss = total_vtx_sym_loss / len(meshes)
    return avg_vtx_sym_loss


def image_symmetry_loss(mesh, sym_plane, num_azim, device, asym_conf_scores=None, sym_bias=0.005, dist=1.9):
    """Computes the image-based symmetry loss.

    Args:
        mesh (Mesh): pytorch3d mesh object
        sym_plane (torch.tensor): vector orthogonal to the plane of symmetry
        num_azim (int): number of azimuths to use when rendering
        device (torch.device): pytorch device
        asym_conf_scores (torch.tensor, optional): tensor of asymmetry confidence scores. Defaults to None.
        sym_bias (float, optional): Symmetry bias term to use for loss. Defaults to 0.005.
        dist (float, optional): Distance to use when rendering. Defaults to 1.9.

    Raises:
        ValueError: Symmetry plane vector needs to be a unit normal vector.

    Returns:
        torch.tensor: loss value
        torch.tensor: renders for image symmetry
    """
    N = np.array([sym_plane])
    if np.linalg.norm(N) != 1:
        raise ValueError("sym_plane needs to be a unit normal")

    # camera positions for one half of the sphere. Offset allows for better angle even when num_azim = 1
    num_views_on_half = num_azim * 2
    offset = 15
    azims = torch.linspace(0+offset,90-offset,num_azim).repeat(2)
    elevs = torch.Tensor([-45 for i in range(num_azim)] + [45 for i in range(num_azim)] )
    dists = torch.ones(num_views_on_half) * dist
    R_half_1, T_half_1 = look_at_view_transform(dists, elevs, azims)
    R = [R_half_1]

    # compute the other half of camera positions according to plane of symmetry
    reflect_matrix = torch.tensor(np.eye(3) - 2*N.T@N, dtype=torch.float)
    for i in range(num_views_on_half):
        camera_position = pytorch3d.renderer.cameras.camera_position_from_spherical_angles(dists[i], elevs[i], azims[i])
        R_sym = pytorch3d.renderer.cameras.look_at_rotation(camera_position@reflect_matrix)
        R.append(R_sym)
    R = torch.cat(R)
    T = torch.cat([T_half_1, T_half_1])
    meshes = mesh.extend(num_views_on_half*2)
    if asym_conf_scores is not None:
        meshes_asym_conf_scores = asym_conf_scores.unsqueeze(0).repeat(num_views_on_half*2,1,3)
        meshes.textures = TexturesVertex(verts_features=meshes_asym_conf_scores)

    # rendering silhouettes
    renders = general_utils.render_mesh(meshes, R, T, device, img_size=224, silhouette=True)[...,3]

    # rendering sym conf. map
    if asym_conf_scores is not None:
        no_lights = PointLights(device=device, ambient_color=((1.0,1.0,1.0),), diffuse_color=((0.0,0.0,0.0),), specular_color=((0.0,0.0,0.0),))
        sym_conf_maps = general_utils.render_mesh(meshes[num_views_on_half:], R[num_views_on_half:], T[num_views_on_half:],
                                                  device, img_size=224, silhouette=False, custom_lights=no_lights, black_bg=False)

    # a sym_triple is [R1, R1_flipped, R2, vertex texture map using asym conf at R2]
    sym_triples = []
    for i in range(num_views_on_half):
        if asym_conf_scores is None:
            sym_triples.append([renders[i], torch.flip(renders[i], [1]), renders[i+num_views_on_half]])
        else:
            sym_conf_map = sym_conf_maps[i][...,0]
            sym_conf_map_mask = torch.ones(sym_conf_map.shape).to(device)
            sym_conf_map_mask[sym_conf_map==1] = 0
            sym_triples.append([renders[i], torch.flip(renders[i], [1]), renders[i+num_views_on_half], sym_conf_map, sym_conf_map*sym_conf_map_mask])
    
    # calculating loss
    sym_loss = 0
    for sym_triple in sym_triples:
        if asym_conf_scores is None:
            sym_loss += F.mse_loss(sym_triple[1], sym_triple[2])
        else:
            sym_loss += torch.mean((torch.log(torch.div(1,sym_triple[3]))*sym_bias)) + torch.mean((F.mse_loss(sym_triple[1], sym_triple[2], reduction="none") * sym_triple[4]))

    sym_loss = sym_loss / num_views_on_half

    return sym_loss, sym_triples
    

def image_symmetry_loss_batched(meshes, sym_plane, num_azim, device, asym_conf_scores=None, sym_bias=0.005):
    """Computes the image-based symmetry loss for a batch.

    Args:
        meshes (Mesh): batch of pytorch3d mesh object
        sym_plane (torch.tensor): vector orthogonal to the plane of symmetry
        num_azim (int): number of azimuths to use when rendering
        device (torch.device): pytorch device
        asym_conf_scores (torch.tensor, optional): tensor of asymmetry confidence scores. Defaults to None.
        sym_bias (float, optional): Symmetry bias term to use for loss. Defaults to 0.005.

    Returns:
        torch.tensor: loss values
        torch.tensor: renders for image symmetry
    """
    total_img_sym_loss = 0
    sym_img_sets = []

    for mesh in meshes:
        curr_sym_loss, curr_sym_img_set = image_symmetry_loss(mesh, sym_plane, num_azim, device, asym_conf_scores=asym_conf_scores, sym_bias=sym_bias)
        total_img_sym_loss += curr_sym_loss
        sym_img_sets.append(curr_sym_img_set)

    avg_img_sym_loss = total_img_sym_loss / len(meshes)
    return avg_img_sym_loss, sym_img_sets
