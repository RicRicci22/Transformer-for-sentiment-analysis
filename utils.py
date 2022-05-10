import csv 
import pickle
import argparse

from collections import Counter
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split
from torchtext.vocab import vocab
import random
import torch

def batch_sampler(train_list,batch_size):
        indices = [(i, len(s[1])) for i, s in enumerate(train_list) if len(s[1])>0]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths 
        for i in range(0, len(indices), batch_size * 100):
            pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))
        
        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), batch_size):
            yield pooled_indices[i:i + batch_size]
            
def load_vocab():
    try:
        with open('vocabulary.pkl','rb') as file:
            voc = pickle.load(file)
        return voc
    except:
        print('Vocabulary not found, run "python test.py --load_model True" to create it!')
            
def create_vocab(train_split,min_freq=100):
    counter = Counter()
    for (_, line) in train_split:
        counter.update(line)
        
    voc = vocab(counter, min_freq=min_freq, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
    # Set default inex 
    voc.set_default_index(voc['<unk>'])
    
    with open('vocabulary.pkl','wb') as file:
        pickle.dump(voc,file)
    
    return voc

    
def create_splits(file, test=0.3, val=0.1):
    # Function that tokenize the sentences, create a vocabulary and return datasets for train, val and test 
    # Define the tokenizer 
    tokenizer = get_tokenizer('basic_english')
    parsed_data = list()
    with open(file,'r') as openfile:
        data = csv.reader(openfile)
        for line in data:
            try:
                tweet = line[-1].split('-')[1]
                parsed_data.append((int(line[0]),tokenizer(tweet)))
            except:
                parsed_data.append((int(line[0]),tokenizer(line[-1])))
    
    # Divide in train-test
    train_split, test_split = train_test_split(parsed_data,test_size=test)
    
    # Divide in train-val 
    train_split, val_split = train_test_split(train_split,test_size=val)
    
    return train_split, test_split, val_split

def parameter_parser():

    parser = argparse.ArgumentParser(description = "Text Generation")

    parser.add_argument("--epochs", dest="num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=1024)
    parser.add_argument("--load_model", dest="load_model", type=bool, default=False)
    parser.add_argument("--sentence", dest="sentence", type=str, default="Bored at work")      
    parser.add_argument("--test_model", dest="test", type=bool, default=False)
    parser.add_argument("--modality", dest="mode", type=str, default='max')
    return parser.parse_args()

            
def load_model(model,path):
    with open(path,'rb') as file:
        weights = pickle.load(file)
    
    model.load_state_dict(weights)
    
    return model 


from torch.nn.utils.rnn import pad_sequence
def from_sentence_to_tensor(sentence,voc):
    tokenizer = get_tokenizer('basic_english')
    text_transform = lambda sentence: [voc['<BOS>']] + [voc[token] for token in tokenizer(sentence)] + [voc['<EOS>']]
    processed_text = torch.tensor(text_transform(sentence)).reshape((1,-1)) # To simulate batch
    if(torch.cuda.is_available()):
        return pad_sequence(processed_text, padding_value=3.0, batch_first=True).cuda()
        
    return pad_sequence(processed_text, padding_value=3.0, batch_first=True)