# TRANSFORMERS FOR SENTIMENT ANALYSIS
Pytorch implementation of transformer architecture as described in the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf).

A repo where I try to build transformer architecture, havily based upon online tutorials.
Upon others, I mention the awesome [video](https://www.youtube.com/watch?v=U0s0f995w14&t=913s) of Aladdin Pearsson, which brought me to this equally awesome [guide](http://peterbloem.nl/blog/transformers) on transformers and the attention mechanism. 
I also thank Peter bloem for his [repository](https://github.com/pbloem/former), where there is a similar implementation, and from which I took inspiration for some little details. 

## What are transformers?

A transformer layer can be thought as a box which takes in input a series of elements (in this case words), and computes intermediate results as a weighted sum of the inputs. A transformer is usually a concatenation of several transformer layers, which produces words embedding considering the semantic and positions of the words in the original sentence. 
It is then up to the downstream goal to handle these representations to perform the desired task.

In this example, the network will use transformer layers to convert the input sentence embeddings to "hidden" embeddings, then these are averaged/maxpooled and a classification layer is used to convert the averaged embedding to binary probabilities. 

The classification is made between happyness or sadness, even tough the network is trained to recognize positive or negative sentiment in tweets. 

## Dataset  
The dataset that will be used is called Sentiment140, and it comprises 1.6 million tweets, labeled as containing negative or positive sentiment. 
The dataset can be found [HERE](https://www.kaggle.com/datasets/kazanova/sentiment140). It is a csv file, and should be put in the same folder that contains the code.

## How to use the code 
To run training, simply use 
```bash
python test.py
```
To evaluate the model on the test set use 
```bash
python test.py --load_model True
```
To run the network on your test sentence, just type
```bash
python test.py --load_model True --test_model True --sentence "your sentence here!"
```
## Pretrained weights

In the folder you can find pretrained weights for the two fusion modalities, namely max-pool and average. The defaul modality is max-pooling, to experiment with averaging mode you can add the --modality mean option when invoking python. 
The weights have been trained using these hyperparameters
```bash
torch.manual_seed(45)
np.random.seed(45)
Optimizer: adam 
Learning rate: 0.0001
Loss function: NNL loss 
N. epochs: 10
```

