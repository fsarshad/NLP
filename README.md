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


# Methods  

Our project revolved around the Stanford Sentiment Treebank - Movie Review Classification Competition. We first completed the to_series() function and then built a document term matrix our of words in the training set. We then removed stop words that occur too frequently to be useful and applied the TD-IDF formula to weigh how common words are general. 
Use the Term Frequency - Inverse Document Frequency (TF-IDF) formula to weigh by how common words are generally. Afterward, we wrote a function to transform data with a processor. New Samples were put into a DTM based on vocabulary from the training set. We then did standard EDA and visualized our code through class balance, review lengths, word frequency per class, Wordcloud, etc. We then programmed lemmatization, stemming, and other text preprocessing. This was included in the preprocessing function. Next, we fit the model on preprocessed data and saved the preprocessor function and model. Then, we Re-fitted a better RF-Model. Following the refitted we wanted to explore grid search over at least 2 hypergrams of RF in order to determine the best model. Once that was achieved we extracted and printed the best score and parameters. Later, we experimented with at least 3 different models from classic ML architecture imports and tabularized the results in order to compare them. At the end, we discussed which models performed better and why. 

* # Custom Model 1
  
* # Custom Model 2 

* # Custom Model 3 

# Results 

* # Fine Tuning 

# Conclusion 

# References 

