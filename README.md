<p align="center">

  <h1 align="center">Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes</h1>
  <p align="center">
    <a href="https://niujinshuchong.github.io/">Zehao Yu</a>
    ·
    <a href="https://tsattler.github.io/">Torsten Sattler</a>
    ·
    <a href="http://www.cvlibs.net/">Andreas Geiger</a>

  </p>

  <h2 align="center">SIGGRAPH ASIA 2024 (Journal Track)</h2>

  <h3 align="center"><a href="https://drive.google.com/file/d/1_IEpaSqDP4DzQ3TbhKyjhXo6SKscpaeq/view?usp=share_link">Paper</a> | <a href="https://arxiv.org/pdf/2404.10772.pdf">arXiv</a> | <a href="https://niujinshuchong.github.io/gaussian-opacity-fields/">Project Page</a>  </h3>
  <div align="center"></div>
</p>


<p align="center">
  <a href="">
    <img src="./media/teaser_gof.png" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
Gaussian Opacity Fields (GOF) enables geometry extraction with 3D Gaussians directly by indentifying its level set. Our regularization improves surface reconstruction and we utilize Marching Tetrahedra for adaptive and compact mesh extraction.</p>
<br>

# Updates

* **[2024.09.11]**: GOF is accepted to SIGGRAPH ASIA 2024 Journal Track. We updated paper with more details, explanations, and ablations.

* **[2024.06.10]**: 🔥 Improve the training speed by 2x with [merged operations](https://github.com/autonomousvision/gaussian-opacity-fields/pull/58). 6 scenes in TNT dataset can be trained in ~24 mins and the bicycle scene in the Mip-NeRF 360 dataset can be trained in ~45 mins. Please pull the latest code and reinstall with `pip install submodules/diff-gaussian-rasterization` to use it.

# Installation
Clone the repository and create an anaconda environment using
```
git clone git@github.com:autonomousvision/gaussian-opacity-fields.git
cd gaussian-opacity-fields

conda create -y -n gof python=3.8
conda activate gof

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/

# tetra-nerf for triangulation
cd submodules/tetra-triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal
cmake .
# you can specify your own cuda path
# export CPATH=/usr/local/cuda-11.3/targets/x86_64-linux/include:$CPATH
make 
pip install -e .
```

# Dataset

Please download the Mip-NeRF 360 dataset from the [official webiste](https://jonbarron.info/mipnerf360/), the NeRF-Synthetic dataset from the [NeRF's official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), the preprocessed DTU dataset from [2DGS](https://surfsplatting.github.io/), the proprocessed Tanks and Temples dataset from [here](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/tree/main). You need to download the ground truth point clouds from the [DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36) and save to `dtu_eval/Offical_DTU_Dataset` to evaluate the geometry reconstruction. For the [Tanks and Temples](https://www.tanksandtemples.org/download/) dataset, you need to download the ground truth point clouds, alignments and cropfiles and save to `eval_tnt/TrainingSet`, such as `eval_tnt/TrainingSet/Caterpillar/Caterpillar.ply`.


# Training and Evaluation
```
# you might need to update the data path in the script accordingly

# NeRF-synthetic dataset
python scripts/run_nerf_synthetic.py

# Mip-NeRF 360 dataset
python scripts/run_mipnerf360.py

# Tanks and Temples dataset
python scripts/run_tnt.py

# DTU dataset
python scripts/run_dtu.py
```


# Personal contribution (eyess-glitch)
This repo contains the modified extract_mesh.py file which contains a strategy implemented for selecting a sub-obtimal subset of images during the compation of opacities through a IoU based pruning-method.

One of the fundamental steps in GoF is computing the opacity values of points within the scene. Opacity directly influences the scene geometry as it is converted into a Signed Distance Function (SDF).  Given a set of views \( V_1, V_2, \dots, V_K \), temporally ordered so that \( V_i \) was captured after \( V_{i-1} \) and before \( V_{i+1} \), we assume that all images were acquired under the same conditions, ensuring constant environmental factors such as lighting, and that the scene remains static.  Under these assumptions, it is highly likely that there exist pairs of images \( (V_i, V_j) \) with \( j > i \) that exhibit significant overlap, covering nearly the same portion of the scene with only minor pixel variations. In this scenario, the computed opacity values for \( V_i \) and \( V_j \) will be nearly identical, leading to redundant calculations.  To minimize redundancy, a subset of views with minimal overlap is selected following these steps:  

1. **Initial Selection:** An initial reference image \( V_i \) is chosen, starting with \( V_1 \).  
2. **Feature Extraction:** Each subsequent image \( V_j \) (where \( j > i \)) is compared with \( V_i \). ORB features and keypoints are extracted, as ORB provides a good balance between computational efficiency and feature representativeness.  
3. **Homography Computation:** A homography is computed to align \( V_j \) with \( V_i \).  
4. **IoU Calculation:** The Intersection over Union (IoU) between the aligned images is calculated as:  

   ```math
   \text{IoU}(V_i, V_j) = \frac{|V_i \cap V_j|}{|V_i \cup V_j|}

5. **Filtering:**  
- If the **IoU exceeds** a predefined threshold, **\( V_j \) is discarded**.  
- Otherwise, **\( V_j \) is retained** and becomes the new reference image **\( V_i \)**, continuing the comparison process with the remaining images.

The generated mesh produced by the model with the proposed IoU procedure was tested on the TNT-dataset, evaluating metrics such as Precision and Recall, for several IoU thresholds alongside the mesh generated by the base methodogy (Base). 

<p align="center">
  <img src="results.PNG" alt="Results" width="70%" />
</p>

# Custom Dataset
We use the same data format from 3DGS, please follow [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) to prepare the your dataset. Then you can train your model and extract a mesh (we use the Tanks and Temples dataset for example)
```
# training
# -r 2 for using downsampled images with factor 2
# --use_decoupled_appearance to enable decoupled appearance modeling if your images has changing lighting conditions
python train.py -s TNT_GOF/TrainingSet/Caterpillar -m exp_TNT/Caterpillar -r 2 --use_decoupled_appearance

# extract the mesh after training
python extract_mesh.py -m exp_TNT/Caterpillar --iteration 30000

# you can open extracted mesh with meshlab or using the following script based on open3d
python mesh_viewer.py exp_TNT/Caterpillar/test/ours_30000/fusion/mesh_binary_search_7.ply
```

# Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [Mip-Splatting](https://github.com/autonomousvision/mip-splatting). Regularizations and some visualizations are taken from [2DGS](https://surfsplatting.github.io/). Tetrahedra triangulation is taken from [Tetra-NeRF](https://github.com/jkulhanek/tetra-nerf). Marching Tetrahdedra is adapted from [Kaolin](https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py) Library. Evaluation scripts for DTU and Tanks and Temples dataset are taken from [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation) respectively. We thank all the authors for their great work and repos. 

# Citation
If you find our code or paper useful, please cite
```bibtex
@article{Yu2024GOF,
  author    = {Yu, Zehao and Sattler, Torsten and Geiger, Andreas},
  title     = {Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes},
  journal   = {ACM Transactions on Graphics},
  year      = {2024},
}
```
If you find the regularizations useful, please kindly cite
```bibtex
@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}
```
