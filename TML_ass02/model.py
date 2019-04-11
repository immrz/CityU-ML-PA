import torch.nn as nn
import torch.nn.functional as F
import torch


class MultiLayer(nn.Module):
    def __init__(self, num_hidden, dim_hidden, residual=False, bn=False, dropout=False, relu=False):
        super(MultiLayer, self).__init__()

        assert num_hidden >= 1
        self.num_hidden = num_hidden
        self.dim_hidden = dim_hidden
        self.residual = residual
        self.bn = bn
        self.dropout = dropout
        self.relu = relu

        self.fc = []
        for i in range(num_hidden):
            input_dim = 9 if i == 0 else dim_hidden
            layer = nn.Linear(input_dim, dim_hidden)
            self.fc.append(layer)
        self.out = nn.Linear(dim_hidden, 3)

    def forward(self, x):
        for i in range(self.num_hidden):
            layer = self.fc[i]
            inter = layer(x)

            # activation
            if self.relu:
                x = F.relu(inter)
            else:
                x = torch.tanh(inter)
        return self.out(x)


class ArcMSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        diff = target - input
        abs_diff = torch.abs(diff)
        over_half = abs_diff[..., -1] > 0.5

        abs_diff[over_half] += torch.tensor([0.0, 0.0, -1.0])
        avg_loss = torch.mean(abs_diff ** 2)

        sign = torch.sign(diff[..., -1])
        shift = 2 * sign * over_half.type_as(sign)
        ctx.save_for_backward(-diff, shift)
        return avg_loss

    @staticmethod
    def backward(ctx, grad_output):
        diff, shift = ctx.saved_tensors
        const = torch.numel(diff)
        grad_input = 2 * diff

        shift_ts = torch.zeros_like(diff)
        shift_ts[..., -1] = shift
        grad_input += shift_ts

        return grad_input / const, None
