# extract mobilenet_v2 feature
taskset 100 python3 extract_frame_feature_crema_d.py --feature_type mobilenet_v2 --alpha 1.0

# extract mfcc feature
taskset 100 python3 extract_audio_feature_crema_d.py --feature_type mfcc --alpha 1.0


# extract mobilenet_v2 feature
taskset 100 python3 extract_frame_feature_crisis_mmd.py --feature_type mobilenet_v2 --alpha 1.0

# extract mfcc feature
taskset 100 python3 extract_audio_featurecrisis_mmd.py --feature_type mfcc --alpha 1.0

python3 extract_feature_ku_har.py
