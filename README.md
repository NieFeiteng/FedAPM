# FedAPM
Code for paper "FedAPM"

Introduction
This repo holds the source code and scripts for reproducing the key experiments of our paper: FedAPM.

Datasets:
<!-- cremad -->
#### 0. Download data: 
cd data
bash download_cremad.sh
#Data will be under data/creama_d

#### 1. Partition the data
#This data has a natural partition (Speaker ID).
python3 features/data_partitioning/crema_d/data_partition.py

#### 2. Feature extraction
# extract mobilenet_v2 framw-wise feature
python3 features/feature_processing/crema_d/extract_frame_feature.py --feature_type mobilenet_v2
# extract mfcc (audio) feature
taskset -c 1-30 python3 features/feature_processing/crema_d/extract_audio_feature.py --feature_type mfcc

<!-- ku-har -->
#### 0. Download data: 
cd data
bash download_ku_har.sh
cd ..

Data will be under data/ku-har

#### 1. Partition the data

This data has the natural partition, so we do not use any simulating. We partition the data in 5-fold, so we can get averaged performance.

python3 features/data_partitioning/ku-har/data_partition.py

The return data is a list, each item containing [key, file_name, label]

#### 2. Feature extraction
For KU-HAR dataset, the feature extraction mainly handles normalization.
python3 features/feature_processing/ku-har/extract_feature.py

#### 0. Download data: The data will be under data/crisis-mmd by default. 

You can modify the data path in system.cfg to the desired path.

cd data
bash download_crisismmd.sh
cd ..

Data will be under data/crisis-mmd

#### 1. Partition the data
We partition the data using direchlet distribution.
# Low data heterogeneity
python3 features/data_partitioning/crisis-mmd/data_partition.py --alpha 1.0


#### 2. Feature extraction

For Crisis-MMD dataset, the feature extraction includes text/visual feature extraction.

# extract mobilenet_v2 feature
python3 features/feature_processing/crisis-mmd/extract_img_feature.py --feature_type mobilenet_v2 --alpha 1.0

# extract mobile-bert feature
python3 features/feature_processing/crisis-mmd/extract_text_feature.py --feature_type mobilebert --alpha 1.0

