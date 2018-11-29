from final_pro.featuring.reduction import Extractor
import numpy as np
from torchvision import models, datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os


def apply(model, dataloader, feat_dim, num_total=None):
    features = []
    targets = []

    for i, (inputs, labels) in enumerate(dataloader):
        outputs = model(inputs)
        feat = outputs.squeeze().numpy()
        labels = labels.squeeze().numpy()

        assert len(feat.shape) == 2 and feat.shape[1] == feat_dim
        assert len(labels.shape) == 1 and labels.shape[0] == feat.shape[0]

        features.append(feat)
        targets.append(labels)

        if num_total is not None:
            print('\rExtraction progress: {:.3f}%'.format(i*100*feat.shape[0] / num_total), end='')
    print()

    features = np.concatenate(features)  # (N, D)
    targets = np.concatenate(targets)    # (N,)
    assert features.shape[0] == targets.shape[0]

    return features, targets


def load_model_and_data(mpath, dpath, res_dim=20):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, res_dim)

    extractor = Extractor(backbone=model, ft_dim=res_dim, out_dim=10)
    extractor.load_state_dict(torch.load(mpath, map_location='cpu'))

    test_set = datasets.MNIST(root=dpath, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ]), train=False)

    return extractor, test_set


def save_feat_and_target(features, targets, folder, mode='b'):
    feat_file = os.path.join(folder, 'MNIST_TestSet10K_DimReduced')
    target_file = os.path.join(folder, 'MNIST_TestSet10K_GroundTruths')

    if mode == 'b':
        np.save(feat_file, features, allow_pickle=False)
        np.save(target_file, targets, allow_pickle=False)
    elif mode == 't':
        np.savetxt(feat_file+'.txt', features)
        np.savetxt(target_file+'.txt', targets)
    else:
        raise ValueError('Not Supported!')

    return True


def main():
    reduced_dim = 20
    extractor, dataset = load_model_and_data(r'D:\cityU\Courses\MachineLearning\Project\fine_tuned_resnet18_v1.pt',
                                             r'D:\cityU\Courses\MachineLearning\Project\data\MNIST',
                                             res_dim=reduced_dim)
    extractor.eval()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, drop_last=False)
    with torch.set_grad_enabled(False):
        features, targets = apply(model=extractor.backbone, dataloader=dataloader,
                                  feat_dim=reduced_dim, num_total=len(dataset))
    if save_feat_and_target(features, targets, folder=r'D:\cityU\Courses\MachineLearning\Project', mode='b'):
        print('Saved!')


if __name__ == '__main__':
    main()
