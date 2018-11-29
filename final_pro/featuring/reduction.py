from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from torch.utils.data import DataLoader
import time
import copy
import torch.nn.functional as F


class Extractor(nn.Module):
    def __init__(self, backbone, ft_dim, out_dim):
        super(Extractor, self).__init__()
        self.backbone = backbone
        self.output = nn.Linear(ft_dim, out_dim)

    def forward(self, x):
        intermediate = self.backbone(x)
        activation = F.relu(intermediate)
        return self.output(activation)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def fine_tuning_model(model, criterion, optimizer, scheduler=None, num_epoch=25):
    since = time.time()
    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)

        for phase in ['train', 'eval']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.
            running_correct = 0
            num_total = dataset_sizes[phase]
            num_processed = 0

            for i, (inputs, labels) in enumerate(dataloader[phase]):
                num_processed += inputs.shape[0]
                print('\rSamples processed: {}/{}, {:.3f}%'.format(
                    num_processed, num_total, num_processed * 100 / num_total), end='')

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predicts = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(predicts == labels.data)

            epoch_loss = running_loss / num_total
            epoch_acc = running_correct.double() / num_total

            print('\r{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
        print()

    elapsed = time.time() - since
    print('Training costs {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))
    print('Best eval acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_weights)
    return model


def reduction_with_resnet(res_dim):
    model = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model, feature_extracting=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, res_dim)
    extractor = Extractor(model, res_dim, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(extractor.parameters(), lr=1e-3, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return fine_tuning_model(extractor, criterion, optimizer, num_epoch=5)


if __name__ == '__main__':
    folder = r'D:\cityU\Courses\MachineLearning\Project\data\MNIST'

    train_set = datasets.MNIST(root=folder, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ]), train=True)

    test_set = datasets.MNIST(root=folder, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ]), train=False)

    dataloader = {'train': DataLoader(train_set, batch_size=32, shuffle=True),
                  'eval': DataLoader(test_set, batch_size=32, shuffle=False)}

    dataset_sizes = {'train': len(train_set), 'eval': len(test_set)}

    my_model = reduction_with_resnet(res_dim=20)
    torch.save(my_model.state_dict(), r'D:\cityU\Courses\MachineLearning\Project\fine_tuned_resnet18.pt')
