# FedDFP
This repo contains the code of our paper "Flexible and Scalable Federated Learning with Deep Feature Prompts for Digital Pathology" which is now under review for npj Digital Medicine.

Datasets
====
You can access and download the data from the following links:

**CAMELYON16**: Access from [Offical Site](https://camelyon16.grand-challenge.org/Data/)

**CAMELYON17**: Access from [Offical Site](https://camelyon17.grand-challenge.org/Data/)

**TCGA-IDH**: Access from [Google Drive](https://drive.google.com/drive/folders/1jgTOKWLtPzsxLic51glabGZdc-4aTdmG?usp=sharing)

Pre-Process
====
For WSI pre-processing, we mainly focus on cropping the slides into patches and then compress patch images into features. Since **TCGA-IDH** already provides the cropped patches, we will skip this step for it. For **CAMELYON16** and **CAMELYON17**, we follow [CLAM](https://github.com/mahmoodlab/CLAM) to perform the pre-processing. Example codes are provided:

```
python3 create_patches_fp.py \
--source $DATA_DIRECTORY \
--save_dir $RESULTS_DIRECTORY \
--patch_size 256 \
--patch 

python3 extract_features_fp.py \
--model_name ViT_S_16 \
--data_h5_dir $DIR_TO_COORDS \
--data_slide_dir $DATA_DIRECTORY \
--csv_path $CSV_FILE_NAME \
--feat_dir $FEATURES_DIRECTORY \
--batch_size 512 \
--slide_ext .tif
```
Training and Testing
====
Training and testing can be done by running the following command:
```
./run.sh #CAMELYON16
./run_17.sh #CAMELYON17
./run_idh.sh #TCGA-IDH
```

Citation
====
If you find this code useful, please consider citing our paper:
```
@article{Cong2025FedDFP,
  title={Flexible and Scalable Federated Learning with Deep Feature Prompts for Digital Pathology},
  author={Cong, Cong and Liu, Sidong and Song, Yang and others},
  journal={npj Digital Medicine},
  year={2025}
}
```
