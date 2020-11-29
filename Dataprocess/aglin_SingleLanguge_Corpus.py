import re
import jieba
from collections import defaultdict



def jaccard(x, y):
    x = set(x)
    y = set(y)
    return len(x & y) / len(x|y)
def countDic(x):
    d =  defaultdict(int)
    for i in x:
        d[i] = d[i] + 1
    return d
def jaccardRepeated(a, b):
    longDic = a if len(a) >= len(b) else b
    shortDic = a if len(a) < len(b) else b
    totalLen = len(a) + len(b)
    longLen, shortLen = len(longDic), len(shortDic)
    if totalLen == 0: return 1
    longDic = countDic(longDic)
    shortDic = countDic(shortDic)
    num = 0
    for key in shortDic.keys():
        num = num + min(shortDic[key], longDic[key])
    # 这里如果用总长度当分母会有问题：相似度永远不会到1
    return num/(longLen + shortLen - num)

def jacSeten(x, y, repeat = True, tokenMode = 'jieba'):
    if tokenMode == 'jieba':
        x = list(jieba.cut(x))
        y = list(jieba.cut(y))
    else:
        x = list(x)
        y = list(y)
    if (repeat): return jaccardRepeated(x, y)
    return jaccard(x, y)



def loadCiba(df):
    originList = []
    tranlatedList = []
    replaceList = []
    for i in range(df.shape[0]):
        origin = df.loc[i, '原语句'].strip()
        tranlate = df.loc[i, '翻译句'].strip()
        re100 = df.loc[i, '替换_100%'].strip()
        originList.append(origin)
        tranlatedList.append(tranlate)
        replaceList.append(re100)
    return originList, tranlatedList, replaceList


def addSeparator(tokens):
    result = []
    temp = ''
    for i in range(len(tokens)):
        if i % 2 == 0:
            temp = tokens[i].strip()
        else:
            temp += tokens[i]
            if len(temp) > 0:
                result.append(temp)
    if len(tokens) % 2 == 1 and temp != '':
        result.append(temp)
    return result


def cut_by_punctuation(originList, tranlatedList, mode=0):
    assert len(originList) == len(tranlatedList)
    originCut = []
    transCut = []
    for i in range(len(originList)):
        if mode == 0:
            temp = re.split(r"([。!！?？；;])", originList[i])
            temp2 = re.split(r"([。!！?？；;])", tranlatedList[i])
        else:
            temp = re.split(r"([。!！?？；;，,])", originList[i])
            temp2 = re.split(r"([。!！?？；;，,])", tranlatedList[i])

        temp = addSeparator(temp)
        temp2 = addSeparator(temp2)
        originCut.append(temp)
        transCut.append(temp2)
    return originCut, transCut


## 根据相似度 合并前后句，使平行语料对应上； 还需要优化
## Needed to be improved
def merge(sen1, sen2, tokenMode=None, maxlen=256):
    res1 = []
    res2 = []
    res1.append(sen1[0])
    res2.append(sen2[0])
    index1 = 1
    index2 = 1
    while ((index1 < len(sen1)) and (index2 < len(sen2))):
        sim = [0, 0, 0, 0, 0]
        #         sim1_2, sim11_2, sim1_22, sim_11_2, sim_1_22 = 0, 0, 0, 0, 0
        sim[0] = jacSeten(sen1[index1], sen2[index2], tokenMode=tokenMode)
        sim[1] = jacSeten(res1[-1] + sen1[index1], res2[-1], tokenMode=tokenMode)
        sim[2] = jacSeten(res1[-1], res2[-1] + sen2[index2], tokenMode=tokenMode)
        if (index1 + 1 < len(sen1)):
            sim[3] = jacSeten(sen1[index1] + sen1[index1 + 1], sen2[index2], tokenMode=tokenMode)
        if (index2 + 1 < len(sen2)):
            sim[4] = jacSeten(sen1[index1], sen2[index2] + sen2[index2 + 1], tokenMode=tokenMode)

        maxIndex = sim.index(max(sim))
        ## and len(res1[-1]) + len(sen1[index1]) <= maxlen
        if (maxIndex == 1 and len(res1[-1]) + len(sen1[index1]) <= maxlen):
            res1[-1] = res1[-1] + sen1[index1]
            index1 += 1
        ##  and len(res2[-1]) + len(sen2[index2]) <= maxlen
        elif (maxIndex == 2 and len(res2[-1]) + len(sen2[index2]) <= maxlen):
            res2[-1] = res2[-1] + sen2[index2]
            index2 += 1
        ## and (len(sen1[index1]) + len(sen1[index1+1])) <= maxlen
        elif (maxIndex == 3 and (len(sen1[index1]) + len(sen1[index1 + 1])) <= maxlen):
            res1.append(sen1[index1] + sen1[index1 + 1])
            res2.append(sen2[index2])
            index1 += 2
            index2 += 1
        ## and (len(sen2[index2]) + len(sen2[index2+1])) <= maxlen
        elif (maxIndex == 4 and (len(sen2[index2]) + len(sen2[index2 + 1])) <= maxlen):
            res1.append(sen1[index1])
            res2.append(sen2[index2] + sen2[index2 + 1])
            index1 += 1
            index2 += 2
        else:
            res1.append(sen1[index1])
            res2.append(sen2[index2])
            index1 += 1
            index2 += 1

    if (index1 < len(sen1)):
        res1[-1] = res1[-1] + ''.join(sen1[index1:])
    if (index2 < len(sen2)):
        res2[-1] = res2[-1] + ''.join(sen2[index2:])

    assert len(res1) == len(res2)
    return res1, res2


def croup_martch(originCut, transCut, isPrint=False, token=True):
    cout = 0
    originFinal = []
    transFinal = []
    for x, y in zip(originCut, transCut):
        x, y = merge(x, y, tokenMode=None)
        originFinal.append(x)
        transFinal.append(y)
        cout += 1
        if isPrint:
            print('\r', cout, end='')

    if (token):
        originCroupFinal = []
        transCroupFinal = []
        for x, y in zip(originFinal, transFinal):
            for s1, s2 in zip(x, y):
                originCroupFinal.append(jieba.lcut(s1, cut_all=False))
                transCroupFinal.append(jieba.lcut(s2, cut_all=False))
        if (isPrint):
            print("\nCroup Size: ", len(originCroupFinal))
        return originCroupFinal, transCroupFinal

    return originFinal, transFinal


## 过滤掉过长掉句子
def filterLength(list1, list2, maxlen=64, isPrint=False, returnFiltered=False):
    res1 = []
    res2 = []
    filtered1 = []
    filtered2 = []
    count = 0
    for i in range(len(list1)):
        if len(list1[i]) > maxlen or len(list2[i]) > maxlen:
            count += 1
            filtered1.append(''.join(list1[i]))
            filtered2.append(''.join(list2[i]))
            continue
        res1.append(list1[i])
        res2.append(list2[i])
    if isPrint:
        print("Filter Number: ", count, " Filtered Croup Size: ", len(res1))
    if returnFiltered:
        return res1, res2, filtered1, filtered2
    return res1, res2


## “long” 长句模式表示：先不按 逗号切分，用句子级别掉符号切分数据，再执行 之前掉Merge操作。 这时会有大量掉过长句子产生
## 对于过长句子，再用逗号等进行切分，再执行一遍平行句匹配Merge 操作，使能够充分利用数据
## 尽管进行了两次 Merge ， 还是有过长的句子。 这时候就需要 优化 上面的 Merge函数了 暂时不想去优化了。。。
## 如果不是 “long” 模式，则直接按逗号级别粒度进行切分
def make_Croup(src, tgt, maxlen=62, mode='long', isPrint=False):
    ## "long": cut by "。" then "，,"
    ## else cut by ",，" directly
    if (mode == 'long'):
        if isPrint:
            print("Long Sentence Mode")
        src, tgt = cut_by_punctuation(src, tgt, mode=0)
        src, tgt = croup_martch(src, tgt, isPrint=isPrint)
        src, tgt, f1, f2 = filterLength(src, tgt, maxlen=maxlen, isPrint=isPrint, returnFiltered=isPrint)
        if isPrint: print("Short Sentence Mode")
        f1, f2 = cut_by_punctuation(f1, f2, mode=1)
        f11, f22 = croup_martch(f1, f2, isPrint=isPrint)
        f11, f22 = filterLength(f11, f22, maxlen=maxlen, isPrint=isPrint, returnFiltered=False)
        src = src + f11
        tgt = tgt + f22
    else:
        if isPrint:
            print("Short Sentence Mode")
        src, tgt = cut_by_punctuation(src, tgt, mode=1)
        src, tgt = croup_martch(src, tgt, isPrint=isPrint)
        src, tgt, f1, f2 = filterLength(src, tgt, maxlen=maxlen, isPrint=isPrint, returnFiltered=True)
    if isPrint:
        print("\nFinal Croup Size: ", len(src))
    return src, tgt



#### Test
# src = ["some sentences", "some sentence"]
# tgt = ["some sentences", "some sentence"]
# src, tgt =  make_Croup(src=src, tgt=tgt, maxlen=64, mode='long', isPrint=True)
