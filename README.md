## NeuroNURBS
Official Implementation of our Paper "[NeuroNURBS: Learning Efficient Surface Representations for 3D Solids](https://arxiv.org/abs/2411.10848)".

![diagram](neuronurbs_diagram.png)

Boundary Representation (B-Rep) is the de facto represen-tation of 3D solids in Computer-Aided Design (CAD). B-Rep solids are defined with a set of NURBS (Non-UniformRational B-Splines) surfaces forming a closed volume. Torepresent a surface, current works often employ the UV-grid approximation, i.e., sample points uniformly on thesurface. However, the UV-grid method is not efficient insurface representation and sometimes lacks precision andregularity. In this work, we propose NeuroNURBS, a repre-sentation learning method to directly encode the parametersof NURBS surfaces. Our evaluation in solid generation andsegmentation tasks indicates that the NeuroNURBS per-forms comparably and, in some cases, superior to UV-grids,but with a significantly improved efficiency: for trainingthe surface autoencoder, GPU consumption is reduced by86.7%; memory requirement drops by 79.9% for storing3D solids. Moreover, adapting BrepGen for solid genera-tion with our NeuroNURBS improves the FID from 30.04to 27.24, and resolves the undulating issue in generatedsurfaces. 

## Environment 

```
conda create -n p39 python==3.9.2
conda init
conda activate p39
conda install -c conda-forge lambouj::occwl
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
conda install -c conda-forge pytorch3d::pytorch3d
# conda install 'numpy<2.0' 
```
 
## Note
This code builds on the previous work "[BrepGen: A B-rep Generative Diffusion Model with Structured Latent Geometry](https://arxiv.org/abs/2401.15563)", but here we use NURBS parameters instead of UV-grids.

## Citation
If you find our work useful in your research, please cite the following paper
```
@misc{fan2024neuronurbslearningefficientsurface,
      title={NeuroNURBS: Learning Efficient Surface Representations for 3D Solids}, 
      author={Jiajie Fan and Babak Gholami and Thomas Bäck and Hao Wang},
      year={2024},
      eprint={2411.10848},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.10848}, 
}
```
