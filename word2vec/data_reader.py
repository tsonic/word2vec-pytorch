import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from itertools import repeat
import itertools
import gc
import ipdb

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
        self.word_frequency = []

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
            self.word_frequency.append(c)
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(self.word_frequency) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        print('Initializing negative samples', flush=True)
        pow_frequency = np.array(self.word_frequency) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)#.astype(np.int32)
        # for wid, c in enumerate(count):
        #     self.negatives += [wid] * int(c)
        df = pd.DataFrame.from_records(enumerate(count))
        self.negatives = np.repeat(df[0].values, df[1].astype(int).values)
        #self.negatives = np.array(self.negatives, dtype=np.int32)
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
        #self.input_file = open(data.inputFileName, encoding="utf8")
        with open(data.inputFileName, encoding="utf8") as f:
            print('Creating words list...', flush=True)
            lines = f.readlines()
            self.words = list(itertools.chain(*[l.split() for l in lines]))
            self.words = [w for w in self.words if w in self.data.word2id]
        boundary = self.window_size
        df_list = []
        print('Creating training dataframe...', flush=True)
        df_short = pd.DataFrame({'word':self.words})
        df_short['id'] = df_short['word'].map(self.data.word2id)
        df_short.drop('word', axis=1, inplace=True)
        for i in range(-boundary, boundary + 1):
            if i == 0:
                continue
            
            df = df_short.copy()
            df['positive'] = df['id'].shift(i)
            df = df.dropna(subset=['positive'])
            df['positive'] = df['positive']
            df = df.query('id != positive')
            # # efficient remove of na
            # if i > 0:
            #     df = df.iloc[i:,]
            # elif i < 0:
            #     df = df.iloc[:i,]
            df_list.append(df)
        df = pd.concat(df_list)
        del df_list, lines, df_short
        gc.collect()
        print('Shuffling samples...', flush=True)
        df = df.sample(frac=1.0, replace=False)
        print('Creating negative samples...', flush=True)
        # neg= self.data.getNegatives(None, len(df) * 5)
        # self.data.negatives = None
        # gc.collect()
        # neg_reshape = neg.reshape((len(df),5))
        # neg_reshape_list = list(neg_reshape)
        # df['negative'] = neg_reshape_list
        # del neg, neg_reshape, neg_reshape_list
        # gc.collect()

        # print('Generating sample look up tables...', flush=True)
        # self.lookup = list(df.itertuples(index=False, name=None))
        self.id_list = df['id'].values
        self.positive_list = df['positive'].values
        print('Dataload initializing finished!', flush=True)


    def __len__(self):
        # return self.data.sentences_count
        return len(self.words)

    def __getitem__(self, idx):

        # if len(self.words) > 1:
        #     word_ids = [self.data.word2id[w] for w in words if
        #                 w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]
        
        return (self.id_list[idx], self.positive_list[idx], self.data.getNegatives(None, 5))

    @staticmethod
    def collate(batches):
        # all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        # all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        # all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        # batches is a list of list, first flatten to single list, then split 
        # each element tuple vertically into 3 long tuples
        all_u,all_v,all_neg_v = zip(*batches)

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
