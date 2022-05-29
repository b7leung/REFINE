The included Python 3.7 code is our implementation of REFINE, as described in CVPR 2022 Submission #2659, "Black-Box Test-Time Shape REFINEment for Single View 3D Reconstruction".
To get started, please refer to the Jupyter Notebook "Simulate_Full_Pipeline.ipynb", which REFINES several examples. 

Code was tested on a GTX 1080Ti GPU and Intel_R_Xeon_R_E5-1650_v3_3 CPU (or similar), running Ubuntu 18.04.3. 
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
- open3d
- sklearn

The following assets were used for experiments in the paper. Please refer to them to reproduce experiments.
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
