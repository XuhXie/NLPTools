from nltk.translate.bleu_score import corpus_bleu, sentence_bleu


class Bleu:
    def __init__(self, tgt_path, pre_path=None):
        self.tgt = Bleu.read_file(tgt_path)
        self.tgt = [[i] for i in self.tgt]
        self.pre = []
        if pre_path is not None:
            self.pre = Bleu.read_file(pre_path)

    def get_bleu(self, path=None):
        if path is None and self.pre == []:
            raise EOFError
        if path:
            self.pre = Bleu.read_file(path)
        return corpus_bleu(self.tgt, self.pre)

    @staticmethod
    def read_file(path):
        return [line.strip().split(" ") for line in open(path, 'r', encoding="utf8").read().splitlines()]
