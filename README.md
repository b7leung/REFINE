# Black-Box Test-Time Shape REFINEment for Single View 3D Reconstruction

![image](https://user-images.githubusercontent.com/20059131/170858764-9ba69aa1-98f4-4408-8166-9ea7360653f4.png)


# Intro

This repo contains the code used for the [CVPRW 2022 paper](http://www.svcl.ucsd.edu/projects/OOWL/CVPRW2022_REFINE/REFINE.pdf), "Black-Box Test-Time Shape REFINEment for Single View 3D Reconstruction". For more details, please refer to the [project website](http://www.svcl.ucsd.edu/projects/OOWL/CVPRW2022_REFINE.html).

The Jupyter Notebook [Simulate_Full_Pipeline.ipynb](https://github.com/b7leung/REFINE/blob/main/Simulate_Full_Pipeline.ipynb) is useful for getting started, and provides code for refining several examples. 

## Dependencies

The following python packages are required:

- tqdm
- PIL
- numpy
- matplotlib
- trimesh
- pandas
- torch
- pytorch3d
- torchvision
- scipy
- openCV
- sklearn

For a precise list of the configuration used in the paper, refer to packages listed in the [refine_env.yml](https://github.com/b7leung/REFINE/blob/main/refine_env.yml) file. You can also create a conda environment using it with:

`
conda env create --file refine_env.yml
`

## Dataset

![image](https://user-images.githubusercontent.com/20059131/170858967-d7cae941-5428-4358-847d-e7c32e223156.png)

The 3D Object Domain Dataset Suite (3D-ODDS) is a hierarchical multiview, multidomain image dataset with 3D meshes that was created for rigoriously evaluting the effectinveness of REFINE. To download, use one of the links below (zip archive password is cvpr2659).

* [Download](https://3dodds.s3.us-west-1.amazonaws.com/3D-ODDS.zip)
* [Download Mirror](https://drive.google.com/file/d/1_u9Gp9luKeuTLVBw_qFJl1JxQ3jKON26/view?usp=sharing)

## Related Work

The following materials were used for experiments in the paper:

- ShapeNet https://shapenet.org/ 
- Pix3d http://pix3d.csail.mit.edu/
- Pix2Mesh. Used under the Apache License 2.0. Commit 7c5a7a1. https://github.com/nywang16/Pixel2Mesh
- Pix2Vox. Used under the MIT License, copyright 2018 Haozhe Xie. Commit f1b8282. https://github.com/hzxie/Pix2Vox
- AtlasNet. Used under the MIT License, copyright 2019 ThibaultGROUEIX. Commit 22a0504. https://github.com/ThibaultGROUEIX/AtlasNet
- Occupancy Networks. Used under the MIT License, copyright 2019 Lars Mescheder, Michael Oechsle, Michael Niemeyer, Andreas Geiger, Sebastian Nowozin. Commit 406f794. https://github.com/autonomousvision/occupancy_networks

F-score and chamfer-L2 is built in. For the EMD and 3D IoU metrics, please refer to the following repositories.
- https://github.com/daerduoCarey/PyTorchEMD
- https://gist.github.com/LMescheder/b5e03ffd1bf8a0dfbb984cacc8c99532
- https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/eval.py

## Citations

If you use this code for your research, please consider citing:

```
@InProceedings{Leung_2022_CVPR,
		author = {Leung, Brandon and Ho, Chih-Hui and Vasconcelos, Nuno},
		title = {Black-Box Test-Time Shape REFINEment for Single View 3D Reconstruction},
		booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
		month = {June},
		year = {2022}
		}
```
