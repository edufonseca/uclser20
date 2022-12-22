# Unsupervised Contrastive Learning of <br> Sound Event Representations

This repository contains the code for the following paper. If you use this code or part of it, please cite:

>Eduardo Fonseca, Diego Ortego, Kevin McGuinness, Noel E. O'Connor, Xavier Serra, "Unsupervised Contrastive Learning of Sound Event Representations", ICASSP 2021.

<p align="center">

| <a href="https://arxiv.org/pdf/2011.07616.pdf" target="_blank">arXiv</a>     | <a href="http://www.eduardofonseca.net/assets/slides/ICASSP21_slides_uclser.pdf" target="_blank">slides</a>  | <a href="http://www.eduardofonseca.net/assets/posters/ICASSP21_poster_uclser.pdf" target="_blank">poster</a> | <a href="http://www.eduardofonseca.net/papers/2021/05/01/unsupervised-contrastive-learning-icassp.html" target="_blank">blog post</a> | <a href="https://youtu.be/Tm3vnetCvDk" target="_blank">video</a> |
| :----:  |   :----:  |  :----: | :----: | :----: |

	
</p>

### Update:
1) If you are interested in this paper, you can find an extended version of it with additional explanations and discussion in Sections 6.1, 6.2, and 6.3 of my <a href="http://www.eduardofonseca.net/assets/pdf/phd_thesis_eduardofonseca_final_corr.pdf" target="_blank">PhD thesis</a>.
2) If you are interested in contrastive audio representation learning, you can have a look at our other paper that received the “Best Audio Representation Learning Paper Award” at WASPAA 2021: <a href="https://arxiv.org/abs/2105.02132" target="_blank">Self-Supervised Learning from Automatically Separated Sound Scenes</a>.

---

In "Unsupervised Contrastive Learning of Sound Event Representations", we propose to learn sound event representations using the proxy task of contrasting differently augmented views of sound events, inspired by SimCLR [[1]](#references). The different views are computed by:  
- sampling TF patches at random within every input clip,  
- mixing resulting patches with unrelated background clips (*mix-back*), and  
- other data augmentations (DAs) (RRC, compression, noise addition, SpecAugment [[2]](#references)).

Our proposed system is illustrated in the figure.

<p align="center">

<img src="/figs/system_whiteback.png" alt="system" width="650"/>

</p>

Our results suggest that unsupervised contrastive pre-training can mitigate the impact of data scarcity and increase robustness against noisy labels. Please check our paper for more details, or have a quicker look at our slide deck, poster, blog post, or video presentation (see links above).

This repository contains the framework that we used for our paper. It comprises the basic stages to learn an audio representation via unsupervised contrastive learning, and then evaluate the representation via supervised sound event classifcation. The system is implemented in PyTorch.

## Dependencies
This framework is tested on Ubuntu 18.04 using a conda environment. To duplicate the conda environment:

`conda create --name <envname> --file spec-file.txt`

## Directories and files

`FSDnoisy18k/` includes folders to locate the FSDnoisy18k dataset and a `FSDnoisy18k.py` to load the dataset (train, val, test), including the data loader for contrastive and supervised training, applying transforms or *mix-back* when appropriate   
`config/` includes `*.yaml` files defining parameters for the different training modes  
`da/` contains data augmentation code, including augmentations mentioned in our paper and more  
`extract/` contains feature extraction code. Computes an .hdf5 file containing log-mel spectrograms and associated labels for a given subset of data    
`logs/` folder for output logs  
`models/` contains definitions for the architectures used (ResNet-18, VGG-like and CRNN)  
`pth/` contains provided pre-trained models for ResNet-18, VGG-like and CRNN  
`src/` contains functions for training and evaluation in both supervised and unsupervised fashion  
`main_train.py` is the main script  
`spec-file.txt` contains conda environment specs

## Usage

#### (0) Download the dataset

Download FSDnoisy18k [[3]](#references) from Zenodo through the <a href="http://www.eduardofonseca.net/FSDnoisy18k/" target="_blank">dataset companion site</a>, unzip it and locate it in a given directory. Fix paths to dataset in `ctrl` section of `*.yaml`. It can be useful to have a look at the different training sets of FSDnoisy18k: a larger set of noisy labels and a small set of clean data [[3]](#references). We use them for training/validation in different ways.

#### (1) Prepare the dataset

Create an .hdf5 file containing log-mel spectrograms and associated labels for each subset of data:  

```
python extract/wav2spec.py -m test -s config/params_unsupervised_cl.yaml
```  

Use `-m` with `train`, `val` or `test` to extract features from each subset. All the extraction parameters are listed in `params_unsupervised_cl.yaml`. Fix path to .hdf5 files in `ctrl` section of `*.yaml`.

#### (2) Run experiment

Our paper comprises three training modes. For convenience, we provide `yaml` files defining the setup for each of them.

1.  **Unsupervised contrastive representation learning** by comparing differently augmented views of sound events. The outcome of this stage is a trained encoder to produce low-dimensional representations. Trained encoders are saved under `results_models/` using a folder name based on the string `experiment_name` in the corresponding yaml (make sure to change it).
```
CUDA_VISIBLE_DEVICES=0 python main_train.py -p config/params_unsupervised_cl.yaml &> logs/output_unsup_cl.out
```

2. **Evaluation of the representation** using a previously trained encoder. Here, we do supervised learning by minimizing cross entropy loss without data agumentation. Currently, we load the provided pre-trained models sitting in `pth/` (you can change this in `main_train.py`, search for `select model`). We follow two evaluation methods:

	- **Linear Evaluation**: train an additional linear classifier on top of the pre-trained unsupervised embeddings.
        ```
        CUDA_VISIBLE_DEVICES=0 python main_train.py -p config/params_supervised_lineval.yaml &> logs/output_lineval.out
        ```

	- **End-to-end Fine Tuning**: fine-tune entire model on two relevant downstream tasks after initializing with pre-trained weights. The two downstream tasks are:
		-  training on the larger set of noisy labels and validate on train\_clean. This is chosen by selecting `train_on_clean: 0` in the yaml.
		-  training on the small set of clean data (allowing 15% for validation). This is chosen by selecting `train_on_clean: 1` in the yaml.  
		
		After choosing the training set for the downstream task, run:
        ```
        CUDA_VISIBLE_DEVICES=0 python main_train.py -p config/params_supervised_finetune.yaml &> logs/output_finetune.out
        ```

The setup in the yaml files should provide the best results reported in our paper. JFYI, the main flags that determine the training mode are `downstream`, `lin_eval` and `method` in the corresponding yaml (they are already adequately set in each yaml).

#### (3) See results:

Check the `logs/*.out` for printed results at the end. Main evaluation metric is balanced (macro) top-1 accuracy. Trained models are saved under `results_models/models*` and some metrics are saved under `results_models/metrics*`.


## Model Zoo

We provide pre-trained encoders as described in our paper, for ResNet-18, VGG-like and CRNN architectures. See `pth/` folder. Note that better encoders could likely be obtained through a more exhaustive exploration of the data augmentation compositions, thus defining a more challenging proxy task. Also, we trained on FSDnoisy18k due to our limited compute resources at the time, yet this framework can be directly applied to other larger datasets such as FSD50K or AudioSet.


## Citation
```
@inproceedings{fonseca2021unsupervised,
  title={Unsupervised Contrastive Learning of Sound Event Representations},
  author={Fonseca, Eduardo and Ortego, Diego and McGuinness, Kevin and O'Connor, Noel E. and Serra, Xavier},
  booktitle={2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2021},
  organization={IEEE}
}

```

## Contact

You are welcome to contact efonseca@google.com should you have any question/suggestion. You can also create an issue.

## Acknowledgment

This work is a collaboration between the <a href="https://www.upf.edu/web/mtg" target="_blank">MTG-UPF</a> and <a href="https://old.insight-centre.org/" target="_blank">Dublin City University's Insight Centre</a>. This work is partially supported by Science Foundation Ireland (SFI) under grant number SFI/15/SIRG/3283 and by the Young European Research University Network under a 2020 mobility award. Eduardo Fonseca is partially supported by a Google Faculty Research Award 2018. The authors are grateful for the GPUs donated by NVIDIA.


## References

[1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A Simple Framework for Contrastive Learning of Visual Representations,” in Int. Conf. on Mach. Learn. (ICML), 2020

[2] Park et al., SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. InterSpeech 2019

[3] E. Fonseca, M. Plakal, D. P. W. Ellis, F. Font, X. Favory, X. Serra, "Learning Sound Event Classifiers from Web Audio with Noisy Labels", In proceedings of ICASSP 2019, Brighton, UK


