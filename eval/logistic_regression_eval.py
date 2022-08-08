from .logistic_regression import LREvaluator
import random
import torch

def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = [i for i in range(num_samples)]
    random.shuffle(indices)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }

def linear_eval(features, labels, split=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # features = features.numpy()
    features = torch.FloatTensor(features).to(device)
    # labels = labels.numpy()
    labels = torch.LongTensor(labels).to(device)
    num_samples = features.shape[0]
    if split is None:
        split = get_split(num_samples=num_samples, train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator(num_epochs=500)(features, labels, split)
    return result
