dataset_info = {
    'cifar10': {
        'model': 'CNN',
        'seeds': [4, 5, 6, 7, 8],
        'lrs': [1.0, 0.5, 0.2, 0.1],
        'rhos': [0.1, 0.05, 0.02, 0.01],
        'fracs': [0.5, 0.3, 0.2, 0.1]
    },
    'crisis_mmd': {
        'model': 'ImageTextClassifier',
        'seeds': [1, 3, 4, 6, 7],
        'lrs': [1.0, 0.5, 0.2, 0.1],
        'rhos': [0.1, 0.05, 0.02, 0.01],
        'fracs': [0.5, 0.3, 0.2, 0.1]
    },
    'ku_har': {
        'model': 'HARClassifier',
        'seeds': [9, 10, 11, 12, 13],
        'lrs': [1.0, 0.5, 0.2, 0.1],
        'rhos': [0.1, 0.05, 0.02, 0.01],
        'fracs': [0.5, 0.3, 0.2, 0.1]
    },
    'crema_d': {
        'model': 'MMActionClassifier',
        'seeds': [10, 11, 12, 13, 14],
        'lrs': [1.0, 0.5, 0.2, 0.1],
        'rhos': [0.1, 0.05, 0.02, 0.01],
        'fracs': [0.5, 0.3, 0.2, 0.1]
    }
}
