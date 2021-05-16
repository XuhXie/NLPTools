import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute, print_step=1):
    total_tokens = 0
    total_loss = 0
    model.train()
    with tqdm(total=len(data_iter)) as pbar:
        for i, (src, tgt) in enumerate(data_iter):
            src = src.cuda()
            tgt = tgt.cuda()
            batch = Batch(src, tgt)
            out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            if i % print_step == 1:
                pbar.update(print_step)
                pbar.set_description("Loss %s" % loss / batch.ntokens)

    return total_loss / total_tokens
