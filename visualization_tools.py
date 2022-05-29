import os

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    HardPhongShader
)

def render_mesh_HQ(mesh, R, T, device, img_size=512, aa_factor=10):
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=img_size*aa_factor, 
        blur_radius=0.000, 
        faces_per_pixel=1, 
        cull_backfaces=True
    )
    ambient = 0.6
    diffuse = 0.35
    specular = 0.4
    lights = PointLights(device=device, ambient_color=((ambient, ambient, ambient), ), diffuse_color=((diffuse, diffuse, diffuse), ),
                         specular_color=((specular, specular, specular), ), location=[[1.0, 8.0, 1.0]])
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

    images = renderer(mesh, cameras=cameras)
    images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
    images = F.avg_pool2d(images, kernel_size=aa_factor, stride=aa_factor)
    images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC

    return images 


def rot_x(theta, degrees=True):
    if degrees:
        theta = theta * (np.pi/180)
    rot_matrix = np.array([[1, 0, 0],
                           [0, np.cos(theta), -np.sin(theta)],
                           [0, np.sin(theta), np.cos(theta)]
                          ])
    return rot_matrix

def rot_y(theta, degrees=True):
    if degrees:
        theta = theta * (np.pi/180)
    rot_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                           [0,1,0],
                           [-np.sin(theta), 0, np.cos(theta)]
                          ])
    return rot_matrix

def rot_z(theta, degrees=True):
    if degrees:
        theta = theta * (np.pi/180)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0,0,1]
                          ])
    return rot_matrix


def show_refinement_results(input_image, mesh_original, mesh_processed, R, T, device, num_novel_view=3, img_size=224, pred_dist=None, pred_elev=None, pred_azim=None):

    mesh_original_render = render_mesh_HQ(mesh_original, R, T, device, img_size=img_size)
    mesh_processed_render = render_mesh_HQ(mesh_processed, R, T, device, img_size=img_size)
    
    # rendering processed mesh at poses other than the predicted pose
    novel_view_renders = []
    for i in range(num_novel_view):
        if pred_dist is None:
            rot = np.expand_dims(rot_y(25),0)
            R = R @ rot
        else:
            R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim + ((i+1)*45))
        novel_view_renders.append(render_mesh_HQ(mesh_processed, R, T, device, img_size=img_size))

    # visualizing
    num_columns = 3 + num_novel_view
    fig, ax = plt.subplots(nrows=1, ncols=num_columns, squeeze=False, figsize=(16,3))

    col_i = 0
    ax[0][col_i].imshow(input_image)
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)
    ax[0][col_i].set_title("Input Image")

    col_i += 1
    ax[0][col_i].imshow(np.clip(mesh_original_render[0, ..., :3].detach().cpu().numpy(),0,1))
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)
    ax[0][col_i].set_title("Original Mesh")

    col_i += 1
    ax[0][col_i].imshow(np.clip(mesh_processed_render[0, ..., :3].detach().cpu().numpy(),0,1))
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)
    ax[0][col_i].set_title("Refined Mesh")
    
    col_i += 1
    for i in range(num_novel_view):
        ax[0][col_i+i].imshow(np.clip(novel_view_renders[i][0, ..., :3].detach().cpu().numpy(),0,1))
        ax[0][col_i+i].xaxis.set_visible(False)
        ax[0][col_i+i].yaxis.set_visible(False)
        ax[0][col_i+i].set_title("Refined Mesh")
    plt.pause(0.05)

    fig.tight_layout(pad=0.5)
    
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot
