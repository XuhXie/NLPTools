"""
# one sentence token usage
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
text = "你好嘛 Abc123"
no_unk_BertTokenizer(text, tokenizer=tokenizer)

# croup token usage
list = token_Bert(path)
"""
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


def unk_to_token(origin, bertTokenlist, unk_tag='[UNK]'):
    """
    orign: origin senten string
    bertTokenlist: the result of tokenizer.tokenize(origin)
    返回: [unk] 代表的原单词
    """
    res = []
    bertTokenlist_mergred = []  # 如果出现连续连个[unk] 合并成一个
    for i, item in enumerate(bertTokenlist):
        if i > 0 and bertTokenlist[i - 1] == unk_tag and item == unk_tag:
            continue
        bertTokenlist_mergred.append(item)

    tokened_string = "".join(bertTokenlist_mergred)
    tokened_string = tokened_string.replace(" ", '')
    tokened_string = tokened_string.replace("#", '')

    split_unk = tokened_string.split(unk_tag)
    split_unk = [i for i in split_unk if len(i) > 0]
    if len(split_unk) == 0:
        return res, []
    split_unk = sorted(split_unk, key=len, reverse=True)

    origin = origin.replace("#", '')
    for i in split_unk:
        origin = origin.replace(i, ' ')

    res = [i for i in origin.split(" ") if len(i) > 0]

    return res, bertTokenlist_mergred


def reconstruct(origins, bertTokenlists, unk_tag='[UNK]'):
    """
    origins: [str1, str2, ...] 原文本
    bertTokenlists: [[token1, token2, ...], [token1, token2,...], ...] bertToeknizer 分词后的结果
    返回: 还原[unk] 之后的分词后原文本, 每个词以 空格隔开 [str1, str2, ...]
    """
    result = []
    count = 0
    for x, y in zip(origins, bertTokenlists):
        unks, bertTokenlist_mergred = unk_to_token(x, y, unk_tag=unk_tag)
        #         print(count)
        count += 1
        if unks:
            index = 0
            for i in range(len(bertTokenlist_mergred)):
                if bertTokenlist_mergred[i] == unk_tag:
                    try:
                        bertTokenlist_mergred[i] = unks[index]
                        index += 1
                    except:  # 特殊情况是，当原文中出现 # 符号 且夹在两个 [unk] 之间时会有bug，这种情况只出现了一次
                        bertTokenlist_mergred = [i for i in bertTokenlist_mergred if i != unk_tag]
                        break
        result.append(" ".join(bertTokenlist_mergred))
    return result


def token_Bert(path, unk_tag='[UNK]'):
    """
    path: 数据地址. 一行一个样本（一句话）
    """
    origins = [line.strip() for line in open(path, 'r', encoding="utf8").read().splitlines()]
    bertTokenlists = [tokenizer.tokenize(i) for i in origins]
    return reconstruct(origins=origins, bertTokenlists=bertTokenlists, unk_tag=unk_tag)


def write_files(name, list1, list2):
    with open(name + '.src', 'w') as f1, open(name + '.tgt', 'w') as f2:
        for i, j in zip(list1, list2):
            f1.write("".join(i) + '\n')
            f2.write("".join(j) + '\n')


def no_unk_BertTokenizer(text, tokenizer):
    tokens_with_unks = tokenizer.tokenize(text)
    result = reconstruct(origins=[text], bertTokenlists=[tokens_with_unks])[0]
    unks, _ = unk_to_token(text, tokens_with_unks)
    return result, unks



