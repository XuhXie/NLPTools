from .train import Batch
import torch
from torch.autograd import Variable
from .model import subsequent_mask


def greedy_decode(model, src, src_mask, max_len=25, start_symbol=2):
    memory = model.encode(src, src_mask)
    batch_size = src.shape[0]
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
    return ys


def run_val(data_iter, model, loss_compute, epoch):
    total_tokens = 0
    total_loss = 0
    model.eval()
    for i, (src, tgt) in enumerate(data_iter):
        src = src.cuda()
        tgt = tgt.cuda()
        batch = Batch(src, tgt)
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens, train=False)
        total_loss += loss
        total_tokens += batch.ntokens

    print(f"Epoch {epoch}:  Loss: {total_loss / total_tokens}")

    return total_loss / total_tokens

