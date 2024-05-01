# NLPAdvMLHW3
![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/f8186818-767e-42ed-ba4b-53ab8d0d3822)

# Content
1. [Introduction](#introduction)
2. [Overview](#Overview)
3. [Methods](#Methods)
    * [Custom Model 1](#custom-model-1)
    * [Custom Model 2](#custom-model-2)
    * [Custom Model 3](#custom-model-3)
4. [Conclusion](#Conclusion)

# Introduction

As a team of three, at Columbia University, our project will be a step-by-step guide on how to train and test different types of NLP models. Such models include (Convo1D, Glove, Bert). 

For the final submission of the project, we have included the three requirements: 
- GitHub README (the current file you are reading)
- [Part 1](https://github.com/fsarshad/NLPAdvMLHW3/blob/main/assignment3_Part1_1_sc4040_ajh2259_fsa2139.ipynb)
- [Part 2](https://github.com/fsarshad/NLPAdvMLHW3/blob/main/assignment3_Part2_1_sc4040_ajh2259_fsa2139.ipynb)
- [Frontend Report](https://github.com/fsarshad/NLPAdvMLHW3/blob/main/frontend_notebook.ipynb) + [modularized .py files]()

# Overview 

Our effort focused on the Stanford Sentiment Treebank - Movie Review Classification Competition. We first finished the to_series() method and then created a document term matrix from the words in the training set. We then deleted stop words that appeared too often to be relevant and used the TD-IDF algorithm to determine how common terms are generic. Use the Term Frequency - Inverse Document Frequency (TF-IDF) calculation to determine how common terms are in general. After that, we created a function to change data using a processor. New samples were entered into a DTM using terminology from the training set. We then performed typical EDA on our code, visualizing it using class balance, review lengths, word frequency per class, Wordcloud, and so on. We next set up lemmatization, stemming, and other text preparation. This was included in the preprocessing function. Next, we fit the model to preprocessed data and saved both the preprocessing code and the model. Then we re-fitted a superior RF model. Following the refit, we intended to do a grid search across at least two hypergrams of RF to select the best model. Once this was accomplished, we extracted and printed the best score and parameters. Later, we tried at least three alternative models from standard ML architectural imports, tabulating the results for comparison. Finally, we discussed which models did better and why. In the second part, we worked on recurrent models, and 3 other models like CNN and transformers. Then, we discussed the results. In part 2 b we incorporated modularization, developed a simple front-end notebook, and other GitHub section development per requirements. 

# Methods  

Before we explain the models we wanted to get a general understanding of the dataset. This led us to perform general eda. 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/5a40970f-530c-4d1f-87bc-ac624409a719)

The image above displays a word cloud of the top 10 negative word reviews (after cleaning). We noticed that words such as movie, film, end, bad, feel, little, nothing, lack, and plot were some of the negative-ish words mentioned in the dataset. 

After examining the negative-ish words we wanted to explore the top 10 positive word reviews. 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/5dee9121-9f6e-4c63-961f-ef84cadcb564)

The image above displays a word cloud of the top 10 positive word reviews (after cleaning). We noticed that words such as good, fun, comedy, best, performance, love, great, heart, well, fill, make, one, and movie were some of the positive words mentioned in the dataset. 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/4592961e-c521-4eeb-8434-46f4c3fbb416)

Next, in the visual above, we explored the length of movie reviews by sentiment. The green trend represents the positive while the blue represents the negative.  The x-axis represents the character length of the review while the y-axis is the count of reviews that are each length. 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/6087b76e-f233-4a42-a452-df06b788d90e)

The visual above showcases the dataset distribution by sentiment with the blue bar representing negative having 3310 and the orange bar representing positive having 3610. There is a slightly more positive dataset distribution by sentiment. 

Below we have the basic model performance. 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/c5ed0a55-4902-4d8d-a6d5-6e14d6135e2b)

We then repeated the submission process to improve our place on the leaderboard. 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/7f6ad48d-03a4-49f5-af40-4cbed8dba2bb)

We then set up callbacks, and a learning rate scheduler, and performed early stopping. Next, we trained the LSTM Model, with its performance shown below: 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/867989e7-ada0-4263-be6f-b33eac1986f1)

Compared to the basic deep learning model, the LSTM model performed better in both training accuracy (0.99 vs 0.94) AND val_accuracy (0.74 vs 0.68). This is most likely because the LSTM is capturing long-term dependencies and sequential context in the review data that the basic deep learning model cannot. Being able to remember information from the beginning of the sequence and use it to make predictions at the end of the sequence, makes the LSTM model far more effective at predicting the most likely sentiment of the review. The basic deep learning model, on the other hand, treats each word in the review as independent and does not consider the order of the words—making it far less effective.

However, we do note that the LSTM model is overfitting to the training data, as the training accuracy is significantly higher than the validation accuracy—and the validation loss is increasing while the training loss continues to decrease. To address this, we could try adding dropout layers or using regularization techniques.

We then compared two or more models: 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/b1fb1b6a-8037-4768-ad2f-7b575d132da8)

Once that was finished, we tuned a model within the range of hyperparameters with a Keras Tuner. 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/500faa62-1a2f-4878-a588-22773b9a0a06)

The best hyperparameters found were:

Number of embedding dimensions: 128 (from 32)
Number of LSTM units: 128 (from 32)
Number of dense units: 192 (from 0, technically)
Learning rate: 0.0001 (from 0.001, a tenth of the original value)

The higher number of embedding dimensions, LSTM units, and newly introduced dense units as hidden layers all allow the model to capture more information about the reviews and the relationships between words. In addition, the lower learning rate allows the model to learn more slowly and avoid overshooting the optimal solution. This combination of hyperparameters allows the model to learn more effectively and generalize better to unseen data.

Next, we Trained three more prediction models to try to predict the SST sentiment dataset well. 

* # Custom Model 1
![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/286d7292-e624-42d2-8919-f5a19e7efabf)


![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/7944859e-830d-4844-824f-08d7338169c4)

Convo1d Models are one-dimensional CNNs used to process dimensional sequential data. Examining the model performance visual allows us to understand the trajectory of the training loss and training validation trend. The trend indicates the effect of the loss during the training and validation. As the viewer, you want to identify whether the trend falls over epochs. This means the model is gaining knowledge from the dataset itself. If you see that the training loss is minimized while the validation loss is going up, it implies overfitting. Overfitting means, that the model is close to the training dataset. 

* # Custom Model 2

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/55f0bd6e-c116-4d67-a13e-e67ee1e74123)


![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/ff47d289-b14f-4c4b-bc16-dbccfbab7181)

* # Custom Model 3

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/29135917/9467708c-3d8c-443d-ac87-3b675f6fcaf4)
<img width="351" alt="image" src="https://github.com/fsarshad/NLPAdvMLHW3/assets/29135917/57d0b948-2fd2-436b-95c2-b9d863235ec3">


# Conclusion 

After experimenting with various models, the fine-tuned BERT model performed the best, achieving a validation accuracy of 0.85, F1 score of 0.85, precision of 0.85, and recall of 0.85 (interesting as to why they're identical). The reasoning behind this is that it is a transformer-based model that is pre-trained on a large corpus of text data. The model has a deep architecture that allows it to capture complex relationships between words and sentences. By leveraging its pre-trained knowledge of language and context, the fine-tuned model is able to learn the sentiment of the movie reviews quite well. Furthermore, since we do include techniques such as dropout, regularization, and learning rate scheduling, the model is able to generalize well to unseen data and avoid overfitting.

# References 
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA. Association for Computational Linguistics.

Cer, D., Yang, Y., Kong, S., Hua, N., Limtiaco, N., John, R. S., Constant, N., Yuan, S., Tar, C., Sung, Y., Strope, B., & Kurzweil, R. (2018). Universal Sentence Encoder. ArXiv. /abs/1803.11175

Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. ArXiv. /abs/1810.04805

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/projects/glove/

Maas, A., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. Learning Word Vectors for Sentiment Analysis. http://www.aclweb.org/anthology/P11-1015 
