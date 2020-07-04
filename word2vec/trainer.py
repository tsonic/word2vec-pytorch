import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from word2vec.data_reader import DataReader, Word2vecDataset
from word2vec.model import SkipGramModel


class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.001, min_count=12, num_workers=0, collate_fn='custom', iprint=500, t=1e-3, ns_exponent=0.75, 
                 optimizer='adam'):


        self.data = DataReader(input_file, min_count,t=t, ns_exponent=ns_exponent)
        dataset = Word2vecDataset(self.data, window_size)
        if collate_fn == 'custom':
            collate_fn = dataset.collate
        else:
            collate_fn = None
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.iprint = iprint
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.optimizer = optimizer

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):

        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            if self.optimizer == 'adam':
                optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            elif self.optimizer == 'sgd':
                optimizer = optim.SGD(self.skip_gram_model.parameters(), lr=self.initial_lr)
            else:
                raise Exception('Unknown optimizer!')

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))
            running_loss = 0.0
            iprint = len(self.dataloader) // 20
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    running_loss = running_loss * (1 - 5/iprint) + loss.item() * (5/iprint)
                    if i > 0 and i % iprint == 0:
                        print(" Loss: " + str(running_loss) + 'lr: ' 
                            + str([param_group['lr'] for param_group in optimizer.param_groups]))
            print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


if __name__ == '__main__':
    w2v = Word2VecTrainer(input_file="input.txt", output_file="out.vec")
    w2v.train()
