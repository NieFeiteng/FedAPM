# extract mobilenet_v2 feature
python3 generate_data/feature_processing/extract_frame_feature_crema_d.py --feature_type mobilenet_v2
# extract mfcc feature
taskset -c 1-30 python3 generate_data/feature_processing/extract_audio_feature_crema_d.py --feature_type mfcc


# extract mobilenet_v2 feature
taskset 100 python3 generate_data/feature_processing/extract_img_feature_crisis_mmd.py --feature_type mobilenet_v2 --alpha 1.0
                 
# extract mfcc feature
taskset 100 python3 generate_data/feature_processing/extract_text_feature_crisis_mmd.py --feature_type mobilebert --alpha 1.0

python3 generate_data/feature_processing/extract_feature_ku_har.py
