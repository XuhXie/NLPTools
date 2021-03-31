# NLPTools



#### aglin_SingleLanguge_Corpus.py:

aglin parallel corpus with same language 对齐同一种语言的平行语料

merge same sentences with some metric like jaccard similarity 使用类似 jaccard 相似度的方法，合并相似的句子（拆分后） 。

```python
#### Test
src = ["some sentences", "some sentence"]
tgt = ["some sentences", "some sentence"]
src, tgt =  make_Croup(src=src, tgt=tgt, maxlen=64, mode='long', isPrint=True)
```
#### make_vocab.py is from [KEPN](https://github.com/LINMouMouZiBo/KEPN)
make vocab from dataset





