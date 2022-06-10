import streamlit as st
from utils import *
from model import Transformer_sentiment
import torch
import numpy as np 
from torch.nn.utils.rnn import pad_sequence
from utils import *

torch.manual_seed(45)
np.random.seed(45)
device = "cuda"
voc = load_vocab()

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

mode = "max"
model = Transformer_sentiment(src_vocab_size=len(voc),src_pad_idx=3.0,modality=mode,max_len=1000)

if(torch.cuda.is_available()):
    model = model.cuda()
# Load the network
if(mode == 'max'):
    model = load_model(model,'weights_max.pt')
elif(mode == 'mean'):
    model = load_model(model,'weights_mean.pt')
model = model.eval()


st.set_page_config(page_title="Sentiment Analysis", page_icon="❓", layout="centered")


st.title("Sentiment analysis")
st.markdown("""I am a machine learning based sentiment analysis tool.""")
st.markdown("""I can analyze the sentiment of a sentence and give you a sentiment score.""")

text = st.text_input(label = "Enter a sentence to analyze")

if st.button("Analyze"):
    if text != '':
        sentence = from_sentence_to_tensor(str(text),voc)
        out=model(sentence)
        out = torch.exp(out).squeeze()
        argmax = torch.argmax(out)
        if(argmax):
            st.write('Im '+str(round(out[argmax].item()*100,0))+'% sure that you are happy! :)')
        else:
            st.write('Im '+str(round(out[argmax].item()*100,0))+'% sure that you are sad.. :(')
    else:
        st.error("Please enter a sentence to analyze")
