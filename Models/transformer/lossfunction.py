import torch
from torch import nn
from torch.autograd import Variable


class LossCompute:
    """A simple loss compute and train function."""

    def __init__(self, seq2seq_criterion, opt=None):
        self.seq2seq_criterion = seq2seq_criterion
        self.opt = opt
        self.softmax_fun = nn.Softmax(dim=-1)

    def __call__(self, sen, sen_label, norm, train=True):

        s2s_loss = self.seq2seq_criterion(sen.contiguous().view(-1, sen.size(-1)),
                                          sen_label.contiguous().view(-1)) / norm
        if self.opt is not None and train:
            s2s_loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return s2s_loss.data * norm, s2s_loss


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
