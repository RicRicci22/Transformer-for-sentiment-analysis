# TRANSFORMERS FROM SCRATCH 

A repo where I try to build transformer architecture, havily based upon online tutorials.
Upon others, I mention the awesome [video](https://www.youtube.com/watch?v=U0s0f995w14&t=913s) of Aladdin Pearsson, which brought me to this also awesome guide on transformers and the attention mechanism http://peterbloem.nl/blog/transformers. 
I also thank https://github.com/pbloem/former, where there are similar implementation, and from which I took inspiration for some little details. 

## What are transformers?

A transformer layer can be thought as a box which takes in input a series of elements (in this case words), and computes intermediate results as a weighted sum of the inputs. A transformer is usually a concatenation of several transformer layers, which produces words embedding considering the semantic and positions of the words in the original sentence. 
It is then up to the downstream goal to handle these representations to perform the desired task.

In this example, when you run "python test.py", the network will use transformer layers to combine the input sentence, then the representations are averaged/maxpooled and a classification is made upon this averaged/maxpooled result. 

The classification is made between happyness or sadness, even tough the network is trained to recognize positive or negative sentiment in tweets. 

To run the network on your test sentence, just type "python test.py --load_model True --test_model True --sentence "your sentence here!"

DOWNLOAD THE DATASET HERE https://www.kaggle.com/datasets/kazanova/sentiment140
And put it in the same folder as the code 


