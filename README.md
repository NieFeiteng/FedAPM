# FedAPM
Code for paper "FedAPM"

Introduction
This repo holds the source code and scripts for reproducing the key experiments of our paper: FedAPM.

#### 0. Download data: 
```
cd data
bash download_cremad.sh
bash download_ku_har.sh
bash download_crisismmd.sh
```



#### 1. Partition the data
#This data has a natural partition (Speaker ID).

```
python3 features/data_partitioning/crema_d/data_partition.py
python3 features/data_partitioning/ku-har/data_partition.py
python3 features/data_partitioning/crisis-mmd/data_partition.py --alpha 1.0
```



#### 2. Feature extraction
```
# extract mobilenet_v2 feature
taskset 100 python3 extract_frame_feature_crema_d.py --feature_type mobilenet_v2 --alpha 1.0

# extract mfcc feature
taskset 100 python3 extract_audio_feature_crema_d.py --feature_type mfcc --alpha 1.0


# extract mobilenet_v2 feature
taskset 100 python3 extract_frame_feature_crisis_mmd.py --feature_type mobilenet_v2 --alpha 1.0

# extract mfcc feature
taskset 100 python3 extract_audio_featurecrisis_mmd.py --feature_type mfcc --alpha 1.0

python3 extract_feature_ku_har.py

```

You can run run_download_data.sh, run_data_partitioning, and run_feature_processing.sh directly to get the processed data.

#### 3. Run

Run run_multiple_methods.sh and run_multiple_parameters to get the experimental data in sections 6.2 and 6.3 of the paper. The experimental data file is under /save.

After obtaining the experimental data, run generate_plots.sh to get Table 3, Figure 1, Figure 4, Figure 5 and Figure 6 in the paper.
