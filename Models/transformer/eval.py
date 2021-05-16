from .train import Batch
import torch
from torch.autograd import Variable
from .model import subsequent_mask
from nltk.translate.bleu_score import corpus_bleu


def convert_id_to_token(tokens, idx2token, cut=False, end_symbl='[SEP]'):
    tokens = tokens.numpy().tolist()
    result = [idx2token[i] for i in tokens]
    result.append(end_symbl)  # 防止结果没有</s>
    if cut:
        result = result[1:]
        index = result.index(end_symbl)
        result = result[:index]
    return result


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
        # next_word = next_word.data
        # next_pos = next_pos.data
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
    return ys


def test_bleu(dataloader, model, idx2token, idx2pos, epoch, maxlen=17, start_symbol=2, start_symbol_pos=1, write=True):
    model.eval()
    test_src, test_tgt = [], []
    sens, poss = [], []
    for (src, tgt, src_pos, tgt_pos) in dataloader:
        test_src += src
        test_tgt += tgt
        src = src.cuda()
        tgt = tgt.cuda()
        src_pos = src_pos.cuda()
        tgt_pos = tgt_pos.cuda()
        batch = Batch(src, tgt, src_pos, tgt_pos)
        sens_, poss_ = greedy_decode(model, batch.src, batch.src_pos, batch.src_mask,
                                     max_len=maxlen, start_symbol=start_symbol, start_symbol_pos=start_symbol_pos)
        sens += sens_.cpu()
        poss += poss_.cpu()
    sen_ = []
    pos_ = []
    for (x, y) in zip(sens, poss):
        sen_.append(convert_id_to_token(x, idx2token=idx2token, cut=True))
        pos_.append(convert_id_to_token(y, idx2token=idx2pos, cut=True))
    # test_src_ = [[(convert_id_to_token(i, idx2token=idx2token, cut=True))] for i in test_src]
    test_tgt_ = [[(convert_id_to_token(i, idx2token=idx2token, cut=True))] for i in test_tgt]
    reference = test_tgt_
    if write:
        with open(f'test_tgt{epoch}.log', 'w') as f, open(f'test_src{epoch}.log', 'w') as f2:
            temp1 = [" ".join(i) for i in sen_]
            temp2 = [" ".join(i[0]) for i in test_tgt_]
            f.write("\n".join(temp1))
            f2.write("\n".join(temp2))

    candidate = sen_
    score = corpus_bleu(reference, candidate)
    print("Eval BLEU: {}".format(score))
    return score


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


def eval_save(best_loss, epoch, val_dataloader, model, loss_compute, path):
    eval_loss = run_val(val_dataloader, model, loss_compute)
    print("Eval Loss: {}".format(eval_loss))
    if eval_loss < best_loss or epoch % 10 == 0:
        print("save model")
        best_loss = eval_loss
        torch.save(model.state_dict(), f'{path}/epoch{epoch}_loss_{eval_loss}.pkl')

    return best_loss, eval_loss