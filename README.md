# BGGN

Official Implementation of our 2024 ICML paper:

**Intersectional Unfairness Discovery**. 

*Gezheng Xu, Qi Chen, Charles Ling, Boyu Wang, Changjian Shui*

([arxiv](https://arxiv.org/abs/2405.20790))

## Current (Important) File Organization:
  - ```/DATASOURCE/``` # dataset (celeba and toxic)
  - ```/EXPS/``` # training scripts
  - ```/output/``` # intermediate data, logs and model params will be stored here
  - ```/data/``` # data pre-processing and related Dataset structure
  - ```train.py``` # main training code
  - ```train_f_model.py``` # main training code for f(x) 
  - ```baseline.py``` # main training code for (relaxed) search tree methods
  - ```engine_*.py``` # concrete training process for different models
  - ```*utils.py``` # auxiliary utils 
  - ```loaders.py``` # dataloader
  - ```network.py``` # all networks utilzed
  - ```hypers.py``` # fixed hyperparams used in this paper
  - ```losses.py``` # external losses used in this paper


## Training Guide

### Training from scratch

1. Toxic dataset
   - Data Preparation
        1. Download Toxic Comment from its homepage on Kaggle Challenge, save ```all_data_with_identities.csv``` to ```./DATASOURCE/toxic/all_data_with_identities.csv```
        2. Download the provided DistilBERT Embeddings and Pre-trained model's prediction results on original TOXIC dataset from: [TOXIC Data](https://drive.google.com/file/d/1NSXRec0xl57_UAQwTl-6lwznpfcETXui/view?usp=drive_link). Save ```toxic_from_wilds_all.h5``` to ```./DATASOURCE/toxic/toxic_from_wilds_all.h5```.
        3. run ```generate_multi_attrs_toxic()``` function in ```./data/toxic_data_preprocessing.py``` to generate necessary input files for following training.

   - Intersectional Sensitive Attribute Dataset Preparation - f(x) training and bias value dataset prepartion
        1. run ```python train_f_model.py --config EXPS/toxic_f_model.yml```
        2. run ```f_model_df_merge_from_seed()``` in ```./data/toxic_data_preprocessing.py``` to merge the dataset. The default merged dataset is ```./DATASOURCE/toxic/Models/f_model_train/all_data_frame_with_yscores.pickle```
     Remark: Download the pre-process [data file (all_data_frame_with_yscores.pickle) for Toxic dataset](https://drive.google.com/file/d/1qSzbQznhRCsytJ8trCObAqCgzK2Ykf3p/view?usp=drive_link).

   - Generative Model Training and Evaluation
        1. run ```python train.py --config EXPS/toxic_train.yml``` to train and evaluate the generative model.

2. CelebA dataset
   - Data Preparation
        1. Download celebA dataset (```Img/```, ```Eval/```,```Anno/``` folders into ```./DATASOURCE/celebA```) from its [HOMEPAGE](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
        2. unzip ```./DATASOURCE/celebA/Img/img_align_celeba.zip``` into ```./DATASOURCE/celebA/Img``` 
        3. run ```celebA_raw_to_dfpickle()``` function in ```./data/celebA_data_preprocessing.py``` to reorganize the images and generate an index pickle file, which will be saved as ```./DATASOURCE/celebA/Anno/data_frame.pickle```.

   - Intersectional Sensitive Attribute Dataset Preparation - f(x) training and bias value dataset prepartion
        1. run ```python train_f_model.py --config EXPS/celebA_f_model.yml``` to generate sensitive attribute dataset under different seeds
        2. run ```f_model_df_merge_from_seed()``` in ```./data/celebA_data_preprocessing.py``` to merge the dataset. The default merged dataset is ```./DATASOURCE/celebA/Models/f_model_train/all_data_frame_with_yscores.pickle```.
     Remark: Download the pre-process [data file (all_data_frame_with_yscores.pickle) for CelebA dataset](https://drive.google.com/file/d/1GPwiKHFw8ZSA8MbzygATRkXm2zHi1QzR/view?usp=drive_link).

   - Generative Model Training and Evaluation
        1. run ```python train.py --config EXPS/celebA_attractive_train.yml``` to train and evaluate the generative model.
 

### Fast Train

1. Toxic dataset
     1. Download the ```toxic_fast_train``` folder from the provided link: [fast train output folder](https://drive.google.com/drive/folders/1aPNWStlKeoWaUhKb1xzvGkiw4CepmSuQ?usp=drive_link). Save it to the ```./output``` dir.
     2. run ```python train.py --config EXPS/toxic_train_forBGGN.yml``` to train and evaluate the BGGN fine-tuning result.

2. CelebA dataset
     1. Download the ```celebA_attractive_fast_train``` folder from the provided link: [fast train output folder](https://drive.google.com/drive/folders/1aPNWStlKeoWaUhKb1xzvGkiw4CepmSuQ?usp=drive_link). Save it to the ```./output``` dir.
     2. run ```python train.py --config EXPS/celebA_attractive_train_forBGGN.yml``` to train and evaluate the BGGN fine-tuning result.

## Citation

```
@misc{xu2024intersectional,
      title={Intersectional Unfairness Discovery}, 
      author={Gezheng Xu and Qi Chen and Charles Ling and Boyu Wang and Changjian Shui},
      year={2024},
      eprint={2405.20790},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
