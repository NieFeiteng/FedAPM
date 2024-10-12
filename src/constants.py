
# define feature len mapping
feature_len_dict = {
    'mobilenet_v2':     1280, 
    'whisper_tiny':     384, 
    'mfcc':             80,  
    'bert':             768, 
    'mobilebert':       512,
    'watch_acc':        3,
    'acc':              3,
    'gyro':             3,
    'i_to_avf':         6,
    'v1_to_v6':         6
}

# define num of class dict
num_class_dict = {
    'crema_d':              4,
    'extrasensory':         4,
    'ku_har':               8,
    'crisis_mmd':           8,
}


# define max feature len in temporal
max_class_dict = {
    'extrasensory':         6,
    'extrasensory_watch':   6
}