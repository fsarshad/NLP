# NLPAdvMLHW3
![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/f8186818-767e-42ed-ba4b-53ab8d0d3822)

# Content
1. [Introduction](#introduction)
2. [Overview](#Overview)
3. [Methods](#Methods)
    * [Custom Model 1](#custom-model-1)
    * [Custom Model 2](#custom-model-2)
    * [Custom Model 3](#custom-model-3)
4. [Results](#Results)
    * [Fine Tuning](#fine-tuning)
5. [Conclusion](#Conclusion)
6. [References](#References)

# Introduction

As a team of three, at Columbia University, our project will be a step-by-step guide on how to train and test different types of NLP models. Such models include (Convo1D, Glove, Bert, and GP2). 

For the final submission of the project, we have included the three requirements: 
- GitHub README (the current file you are reading)
- [Final Report]()
- [Frontend Report]() + [modularized .py files]()

# Overview 

Our effort focused on the Stanford Sentiment Treebank - Movie Review Classification Competition. We first finished the to_series() method and then created a document term matrix from the words in the training set. We then deleted stop words that appeared too often to be relevant and used the TD-IDF algorithm to determine how common terms are generic. Use the Term Frequency - Inverse Document Frequency (TF-IDF) calculation to determine how common terms are in general. After that, we created a function to change data using a processor. New samples were entered into a DTM using terminology from the training set. We then performed typical EDA on our code, visualizing it using class balance, review lengths, word frequency per class, Wordcloud, and so on. We next set up lemmatization, stemming, and other text preparation. This was included in the preprocessing function. Next, we fit the model to preprocessed data and saved both the preprocessing code and the model. Then we re-fitted a superior RF model. Following the refit, we intended to do a grid search across at least two hypergrams of RF to select the best model. Once this was accomplished, we extracted and printed the best score and parameters. Later, we tried at least three alternative models from standard ML architectural imports, tabulating the results for comparison. Finally, we discussed which models did better and why.

# Methods  

Before we explain the models we wanted to get a general understanding of the dataset. This led us to perform general eda. 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/5a40970f-530c-4d1f-87bc-ac624409a719)

The image above displays a word cloud of the top 10 negative word reviews (after cleaning). We noticed that words such as movie, film, end, bad, feel, little, nothing, lack, and plot were some of the negative-ish words mentioned in the dataset. 

After examining the negative-ish words we wanted to explore the top 10 positive word reviews. 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/5dee9121-9f6e-4c63-961f-ef84cadcb564)

The image above displays a word cloud of the top 10 positive word reviews (after cleaning). We noticed that words such as good, fun, comedy, best, performance, love, great, heart, well, fill, make, one, and movie were some of the positive words mentioned in the dataset. 

Next, we explored the length of movie reviews by sentiment. The green trend represents the positive while the blue represents the negative.  The x-axis represents the character length of the review while the y-axis is the count of reviews that are each length. 

![image](https://github.com/fsarshad/NLPAdvMLHW3/assets/51839755/11bc6d31-fe1d-4746-b724-e5a6558022df)


The visual above showcases the dataset distribution by sentiment with the blue bar representing negative having 3310 and the orange bar representing positive having 3610. There is a slightly more positive dataset distribution by sentiment. 


* # Custom Model 1
  
* # Custom Model 2 

* # Custom Model 3 

# Results 

* # Fine Tuning 

# Conclusion 

# References 

Things to cite: Stanford review dataset (https://ai.stanford.edu/~amaas/data/sentiment/), 
GloVe (https://nlp.stanford.edu/projects/glove/), 
BERT (https://arxiv.org/abs/1810.04805)
Universal Sentence Encoder: https://arxiv.org/abs/1803.11175
