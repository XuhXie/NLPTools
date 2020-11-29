import numpy
import torch
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


print(os.path.abspath(os.path.curdir))
basePath = os.path.abspath(os.path.curdir)

####################################################################
### Tools for Load Parallel Croup
####################################################################

def loadVocab(vocab_fpath, padTag=u'<pad>'):
    ## 第0个id 默认为 pad
    vocab = [line.split()[0] for line in open(vocab_fpath, 'r', encoding="utf8").read().splitlines()]
    token2idx = {token: idx + 1 for idx, token in enumerate(vocab)}
    idx2token = {idx + 1: token for idx, token in enumerate(vocab)}
    token2idx[padTag] = 0
    idx2token[0] = padTag
    return token2idx, idx2token


def loadText(path):
    return [line.strip().split(' ') for line in open(path, encoding="utf8").read().splitlines()]


def loadCroupPair(srcPath, tgtPath, vocabPath, maxlen=62):
    token2idx, _ = loadVocab(vocabPath)
    srcList = loadText(srcPath)
    tgtList = loadText(tgtPath)
    assert len(srcList) == len(tgtList)
    srcPadded = []
    tgtpadded = []
    for i, j in zip(srcList, tgtList):
        if (len(i) > maxlen or len(j) > maxlen):
            continue
        ## 每句话起始标志为 <s> 结束标志为</s>
        ## 不在词表内设置为0
        i = [token2idx.get(t, token2idx['<unk>']) for t in ['<s>'] + i + ['</s>']]
        j = [token2idx.get(t, token2idx['<unk>']) for t in ['<s>'] + j + ['</s>']]

        i = i + [0] * (maxlen + 2 - len(i))  # 默认0为 pad
        j = j + [0] * (maxlen + 2 - len(j))  # 默认0为 pad
        srcPadded.append(i)
        tgtpadded.append(j)
    assert len(srcPadded) == len(tgtpadded)

    return torch.tensor(srcPadded), torch.tensor(tgtpadded)


## 输入为tensor
def convert_id_to_token(tokens, idx2token):
    tokens = tokens.numpy().tolist()
    return [idx2token[i] for i in tokens]





####################################################################
#### Example

# devSrcPath = os.path.join(basePath, 'data_src')
# devTgtPath = os.path.join(basePath, 'data_tgt')
# vocabPath = os.path.join(basePath, 'jieba.vocab')
#
# token2idx, idx2token = loadVocab(vocabPath)

# BATCH_SIZE = 4
# train_inputs, train_labels = loadCroupPair(devSrcPath, devTgtPath, 'jieba.vocab')
# train_data = TensorDataset(train_inputs, train_labels)
# train_sampler = RandomSampler(train_data)
# train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
#
# for data, labels in train_dataloader:
#     print(data.shape)
#     print(labels.shape)
#     print(data)
#     print(labels)
#     print(convert_id_to_token(data[0], idx2token))
#     print(convert_id_to_token(labels[0], idx2token))
#     break