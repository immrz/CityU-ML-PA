import torch
from torch.utils.data import Dataset
import os


class CheckInSet(Dataset):
    def __init__(self, fname):
        time_norm = 24*3600 + 60*60 + 60

        samples = []
        with open(os.path.join(r'D:\cityU\Courses\Topics-in-ML\assignment2\social-checkin-prediction', fname), 'r') \
                as fi:
            for line in fi:
                sep = line.strip().split(',')
                samp = []
                for i in range(1, 13, 3):
                    samp.append(float(sep[i]))
                    samp.append(float(sep[i + 1]))
                    h, m, s = map(int, sep[i + 2].split(':'))
                    samp.append((h*3600 + m*60 + s) / time_norm)
                samples.append(samp)
        self.ft = torch.tensor([x[:9] for x in samples])  # (N, 9)
        self.label = torch.tensor([x[9:] for x in samples])  # (N, 3)
        self.n_samp = len(samples)
        self.moments = None

    def __len__(self):
        return self.n_samp

    def __getitem__(self, item):
        return {'feature': self.ft[item], 'label': self.label[item]}

    def whiten(self, stat=None):
        """Normalize the (x, y) coordinates in features and labels by subtracting mean and being divided by stddev.
        :param stat: The mean and stddev of training data, if provided.
        """
        if stat is None:
            combined = torch.cat([self.ft, self.label], dim=1)
            mean = torch.mean(combined, 0)  # (12,)
            std = torch.std(combined, 0)  # (12, )
            self.moments = (mean, std)
        else:
            mean, std = stat

        for i in [0, 1, 3, 4, 6, 7]:
            self.ft[:, i] = (self.ft[:, i] - mean[i]) / std[i]
        for i in range(2):
            self.label[:, i] = (self.label[:, i] - mean[i + 9]) / std[i + 9]


if __name__ == '__main__':
    train_data = CheckInSet('train.csv')
    a = train_data[3]
    print(a)
