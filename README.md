# Context Matters: Graph-based Self-supervised Representation Learning for Medical Images

Official PyTorch implementation for paper *Context Matters: Graph-based Self-supervised Representation Learning for Medical Images, accepted by AAAI 2021*.

<p align="center">
  <img width="75%" height="%75" src="./utils/model_arch.png">
</p>

## Absrtact
Supervised learning method requires a large volume of annotated datasets. Collecting such datasets is time-consuming and expensive. Until now, very few annotated COVID-19 imaging datasets are available. Although self-supervised learning enables us to bootstrap the training by exploiting  unlabeled data, the generic self-supervised methods for natural images do not sufficiently incorporate the context. For medical images, a desirable method should be sensitive enough to detect deviation from normal-appearing tissue of each anatomical region; here, anatomy is the context. We introduce a novel approach with two levels of self-supervised representation learning objectives: one on the regional anatomical level and another on the patient-level. We use graph neural networks to incorporate the relationship between different anatomical regions. The structure of the graph is informed by anatomical correspondences between each patient and an anatomical atlas. In addition, the graph representation has the advantage of handling any arbitrarily sized image in full resolution. Experiments on large-scale Computer Tomography (CT) datasets of lung images show that our approach compares favorably to baseline methods that do not account for the context. We use the learnt embedding to quantify the clinical progression of COVID-19 and show that our method generalizes well to COVID-19 patients from different hospitals. Qualitative results suggest that our model can identify clinically relevant regions in the images.

#### [[Paper & Supplementary Material]](https://arxiv.org/abs/2012.06457) [[Video]](https://slideslive.com/38949020/context-matters-graphbased-selfsupervised-representation-learning-for-medical-images)
    @inproceedings{sun2021context,
      title={Context Matters: Graph-based Self-supervised Representation Learning for Medical Images},
      author={Sun, Li and Yu, Ke and Batmanghelich, Kayhan},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={35},
      number={6},
      pages={4874--4882},
      year={2021}
    }

### Requirements
- [PyTorch](https://pytorch.org/get-started/locally/)
- [SimpleITK](https://anaconda.org/SimpleITK/simpleitk)
- [easydict](https://anaconda.org/conda-forge/easydict)
- [torch_geometric](https://github.com/rusty1s/pytorch_geometric)
- [monai](https://github.com/Project-MONAI/MONAI)
- [tensorboard\_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
- [ANTs](http://stnava.github.io/ANTs/)

### Preprocess Data
Please follow the instructions [here](./preprocess/)

### Training
```bash
sh train.sh
```
The hyperparameter setting can be found in train.sh
### Evaluation
```
python extract_feature.py --exp-dir path_to_work_dir \
                          --root_dir path_to_data_dir \
                          --checkpoint_patch path_to_patch-level_encoder_checkpoint \
                          --checkpoint_graph path_to_subject-level_GCN_checkpoint
```

### Pretrained weights
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Dataset</th>
<th valign="bottom">Anatomy</th>
<th valign="bottom">Checkpoint</th>
<!-- TABLE BODY -->
<tr><td align="left"><a href="https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000179.v6.p2">COPDGene</a></td>
<td align="center">Lung</td>
<td align="center"><a href="https://drive.google.com/file/d/1SoW5zoTAagEwZk3AZ3wmTj05lq4bO-7v/view?usp=sharing">Download</a></td>
</tr>
  
</tbody></table>

### Reference
MoCo v2: https://github.com/facebookresearch/moco
