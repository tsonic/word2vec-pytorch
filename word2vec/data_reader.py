import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from itertools import repeat

np.random.seed(12345)


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


# -----------------------------------------------------------------------------------------------------------------

class Word2vecDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        if idx == 0:
          print('new code!')
        # df = pd.DataFrame.from_records([{'word':k, 'id':v} for k,v in self.data.word2id.items()])
        # df_discards = pd.DataFrame.from_records([{'id':i, 'discard':discard} for i, discard in enumerate(self.data.discards)])
        # df = df.set_index('word').merge(df_discards, on='id')
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    # s = pd.Series(words, name = 'words')
                    # s = s.map(self.data.word2id)
                    # s = s.dropna().astype(int)
                    # filter_idx = np.random.rand(len(s)) < self.data.discards[s.values]
                    # word_ids = s.loc[filter_idx].tolist()

                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    # boundary = np.random.randint(1, self.window_size)
                    boundary = self.window_size

                    # df_list = []
                    # for i in range(-boundary, boundary + 1):
                    #     if i == 0:
                    #         continue
                    #     df = pd.DataFrame({'word':word_ids})
                    #     df['positive'] = df['word'].shift(i)
                    #     # efficient remove of na
                    #     if i > 0:
                    #         df = df.iloc[i:,]
                    #     elif i < 0:
                    #         df = df.iloc[:i,]
                    #     df_list.append(df)
                    # df = pd.concat(df_list * 5)
                    # df['positive'] = df['positive'].astype(int)
                    # df['negative'] = self.data.getNegatives(None, len(df))
                    # ret = list(df.itertuples(index=False, name=None))

                    ret = [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary + 1]) if u != v]
                    return ret

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
