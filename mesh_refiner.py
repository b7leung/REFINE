
import torch
import torch.optim as optim
from pytorch3d.renderer import look_at_view_transform
from tqdm.autonotebook import tqdm
import pandas as pd

from deformation_net_graph_convolutional_lite import DeformationNetworkGraphConvolutionalLite
from forward_pass import batched_forward_pass


class MeshRefiner():

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.num_iterations = self.cfg["refinement"]["num_iterations"]
        self.lr = self.cfg["refinement"]["learning_rate"]
        self.img_sym_num_azim = self.cfg["training"]["img_sym_num_azim"]
        self.sil_lam = self.cfg["training"]["sil_lam"]
        self.l2_lam = self.cfg["training"]["l2_lam"]
        self.lap_lam = self.cfg["training"]["lap_smoothness_lam"]
        self.normals_lam = self.cfg["training"]["normal_consistency_lam"]
        self.img_sym_lam = self.cfg["training"]["img_sym_lam"]
        self.vertex_sym_lam = self.cfg["training"]["vertex_sym_lam"]


    def refine_mesh(self, mesh, rgba_image, R, T, record_debug=False):

        # prep inputs used during training
        image = rgba_image[:,:,:3]
        image_in = torch.unsqueeze(torch.tensor(image/255, dtype=torch.float).permute(2,0,1),0).to(self.device)
        mask = rgba_image[:,:,3] > 0
        mask_gt = torch.unsqueeze(torch.tensor(mask, dtype=torch.float), 0).to(self.device)
        verts_in = torch.unsqueeze(mesh.verts_packed(),0).to(self.device)
        R = R.to(self.device)
        T = T.to(self.device)
        deform_net_input = {"mesh_verts": verts_in, "image":image_in, "R": R, "T":T, "mesh": mesh, "mask": mask_gt}

        # setting up
        deform_net = DeformationNetworkGraphConvolutionalLite(self.cfg, self.device)
        deform_net.to(self.device)
        optimizer = optim.Adam(deform_net.parameters(), lr=self.lr)
        loss_info = pd.DataFrame()
        lowest_loss = None
        best_deformed_mesh = None
        best_refinement_info = {}

        # starting REFINEment
        for i in tqdm(range(self.num_iterations)):

            # forward pass
            deform_net.train()
            optimizer.zero_grad()
            loss_dict, deformed_mesh, forward_pass_info = batched_forward_pass(self.cfg, self.device, deform_net, deform_net_input)

            # optimization step on weighted losses
            total_loss = sum([loss_dict[loss_name] * self.cfg['training'][loss_name.replace("loss", "lam")] for loss_name in loss_dict])
            total_loss.backward()
            optimizer.step()

            # saving info
            curr_train_info = {"iteration": i, "total_loss": total_loss.item()}
            curr_train_info = {**curr_train_info, **{loss_name:loss_dict[loss_name].item() for loss_name in loss_dict}}
            loss_info = loss_info.append(curr_train_info, ignore_index=True)
            if lowest_loss is None or total_loss.item() < lowest_loss:
                lowest_loss = total_loss.item()
                best_deformed_mesh = deformed_mesh
                if record_debug:
                    best_refinement_info = forward_pass_info

        best_refinement_info["loss_info"] = loss_info

        # moving refinement info to cpu
        if "asym_conf_scores" in best_refinement_info:
            best_refinement_info["asym_conf_scores"] = best_refinement_info["asym_conf_scores"].detach().cpu()
        if "img_sym_loss_debug_imgs" in best_refinement_info:
            for i in range(len(best_refinement_info["img_sym_loss_debug_imgs"])):
                for j in range(len(best_refinement_info["img_sym_loss_debug_imgs"][i])):
                    for k in range(len(best_refinement_info["img_sym_loss_debug_imgs"][i][j])):
                        best_refinement_info["img_sym_loss_debug_imgs"][i][j][k] = best_refinement_info["img_sym_loss_debug_imgs"][i][j][k].detach().cpu()

        return best_deformed_mesh, best_refinement_info
