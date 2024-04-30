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



* # Custom Model 1
  
* # Custom Model 2 

* # Custom Model 3 

# Results 

* # Fine Tuning 

# Conclusion 

# References 

