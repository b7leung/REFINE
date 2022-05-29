
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch3d

from resnet_backbone import build_backbone
import general_utils


class DeformationNetworkGraphConvolutionalLite(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()

        self.device = device
        self.asym = cfg["training"]["vertex_asym"]
        hidden_dim = 128

        self.backbone, self.feat_dims = build_backbone("resnet18", num_conv=2)
        img_feat_dim = sum(self.feat_dims)
        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)

        self.gconvs = nn.ModuleList()
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=3+hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))

        self.vert_offset = nn.Linear(hidden_dim, 3)

        self.asym_conf = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1))


    def forward(self, input_batch):
        
        images = input_batch["image"].to(self.device)
        mesh_batch = input_batch["mesh"].to(self.device)
        R = input_batch["R"]
        T = input_batch["T"]

        feat_maps = self.backbone(images)

        # aligning and normalizing vertex positions so (-1,-1) is the top left, (1,1) is the bottom right relative to the feature map
        verts_padded = mesh_batch.verts_padded()
        aligned_verts_padded = general_utils.align_and_normalize_verts_original(verts_padded, R, T, self.device)

        # computing vert_align features
        vert_align_feats = pytorch3d.ops.vert_align(feat_maps, aligned_verts_padded, return_packed=True)
        vert_align_feats = F.relu(self.bottleneck(vert_align_feats))

        # appending original cordinates to vert_align features
        batch_vertex_features = torch.cat([vert_align_feats, mesh_batch.verts_packed()], dim=1)
        for i in range(len(self.gconvs)):
            batch_vertex_features = F.relu(self.gconvs[i](batch_vertex_features, mesh_batch.edges_packed()))

        delta_v = self.vert_offset(batch_vertex_features)

        if self.asym:
            asym_conf_scores = torch.sigmoid(self.asym_conf(batch_vertex_features))
            return [delta_v, asym_conf_scores]
        else:
            return delta_v
