
import torch
from torch.nn import functional as F
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency

from utils import general_utils
from utils import losses as def_losses


def batched_forward_pass(cfg, device, deform_net, input_batch):
    forward_pass_info = {}

    # deforming mesh
    if cfg["training"]["vertex_asym"]:
        deformation_output, asym_conf_scores = deform_net(input_batch)
        forward_pass_info["asym_conf_scores"] = asym_conf_scores
    else:
        deformation_output = deform_net(input_batch)
        asym_conf_scores = None
    deformation_output = deformation_output.reshape((-1,3))
    mesh_batch = input_batch["mesh"].to(device)
    deformed_meshes = mesh_batch.offset_verts(deformation_output)

    # computing network's losses
    loss_dict = {}
    if cfg["training"]["sil_lam"] > 0:
        R = input_batch["R"]
        T = input_batch["T"]
        deformed_renders = general_utils.render_mesh(deformed_meshes, R, T, device, img_size=224, silhouette=True)
        deformed_silhouettes = deformed_renders[:, :, :, 3]
        mask_batch = input_batch["mask"].to(device)
        loss_dict["sil_loss"] = F.binary_cross_entropy(deformed_silhouettes, mask_batch)
    else:
        loss_dict["sil_loss"] = torch.tensor(0).to(device)

    if cfg["training"]["l2_lam"] > 0:
        num_vertices = deformed_meshes.verts_packed().shape[0]
        zero_deformation_tensor = torch.zeros((num_vertices, 3)).to(device)
        loss_dict["l2_loss"] = F.mse_loss(deformation_output, zero_deformation_tensor)
    else:
        loss_dict["l2_loss"] = torch.tensor(0).to(device)

    if cfg["training"]["lap_smoothness_lam"] > 0:
        loss_dict["lap_smoothness_loss"] = mesh_laplacian_smoothing(deformed_meshes)
    else:
        loss_dict["lap_smoothness_loss"] = torch.tensor(0).to(device)

    if cfg["training"]["normal_consistency_lam"] > 0:
        loss_dict["normal_consistency_loss"] = mesh_normal_consistency(deformed_meshes)
    else:
        loss_dict["normal_consistency_loss"] = torch.tensor(0).to(device)

    if cfg["training"]["img_sym_lam"] > 0:
        loss_dict["img_sym_loss"], sym_img_sets = def_losses.image_symmetry_loss_batched(deformed_meshes, [0,0,1], cfg["training"]["img_sym_num_azim"], device,
                                                                                         asym_conf_scores, cfg["training"]["img_sym_bias"])
        forward_pass_info["img_sym_loss_debug_imgs"] = sym_img_sets
    else:
        loss_dict["img_sym_loss"] = torch.tensor(0).to(device)

    if cfg["training"]["vertex_sym_lam"] > 0:
        loss_dict["vertex_sym_loss"] = def_losses.vertex_symmetry_loss_batched(deformed_meshes, [0,0,1], device,
                                                                               asym_conf_scores, cfg["training"]["vertex_sym_bias"])
    else:
        loss_dict["vertex_sym_loss"] = torch.tensor(0).to(device)

    return loss_dict, deformed_meshes, forward_pass_info

