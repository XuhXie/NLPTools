#### Example: loaded pareallel croup

from Dataprocess.toolsParallelCroup import *

devSrcPath = os.path.join(basePath, 'data_src')
devTgtPath = os.path.join(basePath, 'data_tgt')
vocabPath = os.path.join(basePath, 'jieba.vocab')

token2idx, idx2token = loadVocab(vocabPath)

BATCH_SIZE = 4
train_inputs, train_labels = loadCroupPair(devSrcPath, devTgtPath, 'jieba.vocab')
train_data = TensorDataset(train_inputs, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

for data, labels in train_dataloader:
    print(data.shape)
    print(labels.shape)
    print(data)
    print(labels)
    print(convert_id_to_token(data[0], idx2token))
    print(convert_id_to_token(labels[0], idx2token))
    break