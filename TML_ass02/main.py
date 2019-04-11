from TML_ass02.data import CheckInSet
from TML_ass02.model import MultiLayer, ArcMSE
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import time
import torch.nn as nn


def read_and_process():
    train_data = CheckInSet('train.csv')
    val_data = CheckInSet('validation.csv')
    test_data = CheckInSet('test.csv')

    train_data.whiten()
    val_data.whiten(stat=train_data.moments)
    test_data.whiten(stat=train_data.moments)

    return train_data, val_data, test_data


def train_model(model, data, criterion, device, num_epoch=100, use_scheduler=False,
                opt_type='sgd', val_data=None, batch_size=32, **kwargs):
    t_start = time.time()

    optimizer = None
    scheduler = None
    model = model.to(device)

    if opt_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    if use_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=num_epoch//3, gamma=0.1)

    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4) \
        if val_data is not None else None
    dataloader = {'train': train_loader, 'val': val_loader}

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()
            elif val_data is not None:
                model.eval()
            else:
                break

            loss_per_samp = 0.0
            num_samp = 0

            for batch in dataloader[phase]:
                feature = batch['feature'].to(device)
                label = batch['label'].to(device)
                num_samp += feature.size()[0]
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    prediction = model(feature)
                    loss = criterion(prediction, label)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                loss_per_samp += loss.item() * feature.size()[0]

            loss_per_samp /= num_samp
            print('{} Loss: {:.8f}'.format(phase, loss_per_samp))

    t_elapse = time.time() - t_start
    print('Training completes in {:.0f}m {:.0f}s'.format(t_elapse // 60, t_elapse % 60))
    return model


def main():
    mlp = MultiLayer(num_hidden=1, dim_hidden=50)
    train_data, val_data, test_data = read_and_process()

    # criterion = nn.MSELoss(reduction='elementwise_mean')
    criterion = ArcMSE.apply
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mlp = train_model(mlp, train_data, criterion, device, num_epoch=25, use_scheduler=True, val_data=val_data)

    torch.save(mlp.state_dict(), r'D:\cityU\Courses\Topics-in-ML\assignment2\model_saver\naive_02.pt')


if __name__ == '__main__':
    main()
