from torch.utils.data import Dataset
import torch
from collections import OrderedDict
import random 
## DATASETS ##

class SentimentDataloader(Dataset):
    def __init__(self,X,lengths,Y,batch_size) -> None:
        super(SentimentDataloader).__init__()
        self.X,self.Y = X,Y
        self.lengths = lengths
        self.batch_size = batch_size
        self.current = -1
        self._generate_batch_map()
    
    def _generate_batch_map(self):
        batch_map = OrderedDict()
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        for idx, length in enumerate(self.lengths):
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        self.batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i+self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                self.batch_list.append(group)
        # Shuffle the list
        random.shuffle(self.batch_list)
    
    def batch_count(self):
        return len(self.batch_list)

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current > len(self.batch_list)-1:
            self.current = -1
            # Shuffle the list
            random.shuffle(self.batch_list)
            raise StopIteration
        else:
            if torch.cuda.is_available():
                return torch.from_numpy(self.X[self.batch_list[self.current]]).type(torch.IntTensor).cuda(), \
                    torch.from_numpy(self.Y[self.batch_list[self.current]]).type(torch.IntTensor).cuda(), \
                    torch.from_numpy(self.lengths[self.batch_list[self.current]]).type(torch.IntTensor).cuda()
            else:
                return torch.from_numpy(self.X[self.batch_list[self.current]]).type(torch.IntTensor), \
                        torch.from_numpy(self.Y[self.batch_list[self.current]]).type(torch.IntTensor), \
                        torch.from_numpy(self.lengths[self.batch_list[self.current]]).type(torch.IntTensor)


class SentimentDataset(Dataset):
    def __init__(self,X,Y) -> None:
        super(SentimentDataset).__init__()
        self.X,self.Y = X,Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        if torch.cuda.is_available():
            return torch.from_numpy(self.X[idx,:]).type(torch.IntTensor).cuda(), torch.from_numpy(self.Y[idx]).type(torch.IntTensor).cuda()
        else:
            return torch.from_numpy(self.X[idx,:]).type(torch.IntTensor), torch.from_numpy(self.Y[idx]).type(torch.IntTensor)


# TRYING TORCHTEXT NEW FUNCTIONALITIES
if __name__=="__main__":
    
    # STEP 1 --> load the dataset
    from torchtext.datasets import IMDB

    train_iter, test_iter = IMDB(split=('train', 'test'))
    
    # STEP 2 --> tokenize the sequences and build the vocabulary
    from torchtext.data.utils import get_tokenizer
    tokenizer = get_tokenizer('basic_english')
    
    from collections import Counter
    from torchtext.vocab import vocab

    counter = Counter()
    for (label, line) in train_iter:
        counter.update(tokenizer(line))
    # print(list(counter.values())[0:10])
    # print(list(counter.keys())[0:10])
    vocab = vocab(counter, min_freq=100, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
    
    print("The length of the new vocab is", len(vocab))
    print(list(vocab.get_stoi().values())[0:20])
    print(list(vocab.get_stoi().keys())[0:20])
    print(vocab.get_itos()[0:20])
    
    # Got to define the transform for a text string and the corresponding label 
    text_transform = lambda x: [vocab['<BOS>']] + [vocab[token] if token in vocab.get_stoi().keys() else vocab['<unk>'] for token in tokenizer(x)] + [vocab['<EOS>']]
    label_transform = lambda x: 1 if x == 'pos' else 0
    # print("input to the text_transform:", "here is an example")
    # print("output of the text_transform:", text_transform("here is an example"))
    
    # STEP 3 --> GENERATE BATCH ITERATOR 
    
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence

    def collate_batch(batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(label_transform(_label))
            processed_text = torch.tensor(text_transform(_text))
            text_list.append(processed_text)
        return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0, batch_first=True)
    

    train_list = list(train_iter)
    batch_size = 128  # A batch size of 8
    
    import random
    def batch_sampler(train_list,batch_size,tokenizer):
        indices = [(i, len(tokenizer(s[1]))) for i, s in enumerate(train_list)]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths 
        for i in range(0, len(indices), batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))
        
        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), batch_size):
            yield pooled_indices[i:i + batch_size]
    
    bucket_dataloader = DataLoader(train_list, batch_sampler=batch_sampler(train_list,64,tokenizer),
                                collate_fn=collate_batch)

    data = (next(iter(bucket_dataloader)))
    # print(data[0])
    # print(data[1])
    print(data[0].shape)
    print(data[1].shape)
        