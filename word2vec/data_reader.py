import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info
from itertools import repeat
import itertools
import gc
import ipdb

np.random.seed(12345)


class DataReader:

    NEGATIVE_TABLE_SIZE = 3e8

    def __init__(self, inputFileName, min_count,t=1e-4,ns_exponent=0.5):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = []
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = []

        self.inputFileName = inputFileName
        self.read_words(min_count)
        self.initTableNegatives(ns_exponent=ns_exponent)
        self.initTableDiscards(t=t)

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
            self.id2word.append(w)
            self.word_frequency.append(c)
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self,t):
        f = np.array(self.word_frequency, dtype = float) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self, ns_exponent):
        print('Initializing negative samples', flush=True)
        pow_frequency = np.array(self.word_frequency) ** ns_exponent
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)

        df = pd.DataFrame.from_records(enumerate(count))
        # the column 0 is the word index, column 1 is the count of the word
        self.negatives = np.repeat(df[0].astype(np.int64).values, df[1].astype(np.int64).values)
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
        print('monitor multi-worker')
        with open(data.inputFileName, encoding="utf8") as f:
            print('Creating words list...', flush=True)
            lines = f.readlines()
            self.words = list(itertools.chain(*[l.split() for l in lines]))
            self.words = [w for w in self.words if w in self.data.word2id]
            
        boundary = self.window_size
        df_list = []
        print('Creating training dataframe...', flush=True)
        df_short = pd.DataFrame({'word':self.words})
        df_short['id'] = df_short['word'].map(self.data.word2id).astype(np.int64)
        df_short['discard_limit'] = df_short['id'].map({i:limit for i, limit in enumerate(self.data.discards)})
        for i in range(-boundary, boundary + 1):
            if i == 0:
                continue
            
            df = df_short.copy(deep=True)
            df = df.assign(keep = lambda df: np.random.rand(len(df)) <  df['discard_limit'])
            print('keep ratio is %f' % df['keep'].mean())
            print('sample of discard words are :' + str(df.query('not keep')['word'].sample(n=5).tolist()))
            df = df.assign(positive = lambda df: df['id'].shift(i))
            df = df.dropna(subset=['positive'])
            df = df.query('id != positive and keep')
            df = df.drop(columns=['discard_limit','keep','word'])

            df_list.append(df)
        df = pd.concat(df_list)
        del df_list, lines, df_short
        gc.collect()
        print('Shuffling samples...', flush=True)
        df = df.sample(frac=1.0, replace=False)
        print('Creating negative samples...', flush=True)

        self.id_list = df['id'].astype(np.int64).values
        self.positive_list = df['positive'].astype(np.int64).values
        print('Dataload initializing finished!', flush=True)

        self.negative_collision = 0


    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        return idx

    def collate(self,batches):

        negs = self.data.getNegatives(None, len(batches) * 5).reshape((len(batches), 5))
        st = batches[0]
        ed = batches[-1] + 1

        return torch.LongTensor(self.id_list[st:ed]), torch.LongTensor(self.positive_list[st:ed]), torch.from_numpy(negs)
    
    @staticmethod
    def worker_init_fn(_):
        worker_info = get_worker_info()
        
        np.random.seed(np.random.get_state()[1][0] + worker_info.id) 
        np.random.shuffle(worker_info.dataset.data.negatives)
