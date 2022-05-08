from regex import I
from utils import *
from model import Transformer_sentiment
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import pickle
import numpy as np 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

if __name__=="__main__":
    args = parameter_parser()
    # FIXING RANDOM SEEDS for more reproducibility
    torch.manual_seed(45)
    np.random.seed(45)
    device = "cuda"
    
    train_split,test_split,val_split = create_splits(r'training.1600000.processed.noemoticon.csv')
    voc = create_vocab(train_split,100)
    
    # Define transforms to convert input to numbers 
    text_transform = lambda x: [voc['<BOS>']] + [voc[token] for token in x] + [voc['<EOS>']]
    label_transform = lambda x: int(x/4) # CUSTOM 

    def collate_batch(batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(label_transform(_label))
            processed_text = torch.tensor(text_transform(_text))
            text_list.append(processed_text)
        if(torch.cuda.is_available()):
            return pad_sequence(text_list, padding_value=3.0, batch_first=True).cuda(), torch.tensor(label_list).cuda()
        
        return pad_sequence(text_list, padding_value=3.0, batch_first=True), torch.tensor(label_list).type(torch.LongTensor)
    
    model = Transformer_sentiment(len(voc),3.0,max_len=1000,embed_size=args.hidden_dim,modality=args.mode)
    
    if(torch.cuda.is_available()):
        # print('Model to CUDA!')
        model = model.cuda()
        
    # Optimizer initialization
    if(args.load_model):
        # Load the network
        if(args.mode == 'max'):
            model = load_model(model,'weights_max.pt')
        elif(args.mode == 'mean'):
            model = load_model(model,'weights_mean.pt')
        model = model.eval()
        
        if(args.test):
            sentence = from_sentence_to_tensor(args.sentence,voc)
            out=model(sentence)
            out = torch.exp(out).squeeze()
            argmax = torch.argmax(out)
            if(argmax):
                print('Im '+str(round(out[argmax].item()*100,0))+'% sure that you are happy! :)')
            else:
                print('Im '+str(round(out[argmax].item()*100,0))+'% sure that you are sad.. :(')
        else:
            testloader = DataLoader(test_split,batch_sampler=batch_sampler(test_split,1),collate_fn=collate_batch)
            print('\n###########  Calculating test accuracy  ###########\n')
            true_positive=torch.tensor(0).to(device)
            with torch.no_grad():
                for j, datatest in enumerate(tqdm(testloader)):
                    X,y = datatest
                    out = model(X)
                    print(torch.exp(out))
                    argmax = torch.argmax(out,dim=1)
                    true_positive+=torch.sum((argmax==y))
                testaccuracy = true_positive/len(test_split)
                print('Test accuracy: ',str(round(testaccuracy.item(),3)))
        
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        # Set model in training mode
        loss_fn = nn.NLLLoss()
        model.train()
        for e in range(args.num_epochs):
            trainloader = DataLoader(train_split,batch_sampler=batch_sampler(train_split,128),collate_fn=collate_batch)
            epoch_loss = 0
            for i, data in enumerate(tqdm(iter(trainloader))):
                X,y = data
                out = model(X)
                loss = loss_fn(out,y)
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss+=loss.item()
            
            print('Epoch '+str(e)+' loss: '+str(round(epoch_loss/((i+1)),5)))
            
            # CALCULATE ACCURACY ON VALIDATION SET
            valloader = DataLoader(val_split,batch_sampler=batch_sampler(val_split,64),collate_fn=collate_batch)
            true_positive=torch.tensor(0).to(device)
            model.eval()
            with torch.no_grad():
                print('\nCalculating validation accuracy\n')
                for j, dataval in enumerate(tqdm(iter(valloader)),0):
                    X,y = dataval
                    out = model(X)
                    argmax = torch.argmax(out,dim=1)
                    true_positive+=torch.sum((argmax==y))
                valaccuracy = true_positive/len(val_split)
                print('Validation accuracy: ',str(round(valaccuracy.item(),3)))
            # Put the model in training again
            model.train()
            
                    
        # Save the network 
        with open('weights.pt','wb') as file:
            pickle.dump(model.state_dict(),file)
    