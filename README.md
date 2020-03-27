# Advanced-learning-models-course
MSIAM program

### Introduction
The goal of this data challenge was to predict whether a DNA sequence region is binding site to a specific transcription factor. It is a classification task, solving by implementing machine learning algorithms.  
### Data
This data challenge contained three datasets. 
It was necessary to predict separately the labels for each dataset. 
Each train dataset contained 2000 sequences, where one raw is one specific sequence motif in the genome. Each test dataset contained 1000 test sequences, for which I must predict labels: bound or unbound. 

### Representation of data
All sequences in data consist of 4 unique letters: G, T, A, C.
For representation of data it was considered all possible sub sequences with length equal to k=4 (experimentally selected), i.e. 4^4 = 256 sub sequences (list subsequence). 
Further we consider one input sequence (for example x_train[0]). We create a dictionary, where clues are all possible sub sequences (consisted of k elements) of input sample and values are the number of times the current sub sequence occurs in this input sample. We build a feature map (array features_array), elements of which is the number of times each element of list subsequence contains in the current input sample (x_train[0]).

### Models
In the beginning I tried using models from scikit-learn library to see what model gives the best score. I considered logistic regression and SVM with different kernels. SVM with rbf kernel showed the best result, therefore it was decided to realize it. 

Class SVM was realized. The main items of the realization are the next:

    \item Creating a Gram matrix
    \item Finding the components for quadratic program problem  
    \item Solving the quadratic program problem obtaining Lagrange multipliers (using library ’cvxopt')
    \item Find the support vectors threshold = 1e-6 using the solution of the problem 
    \item Fitting support vectors with the intercept
    \item Prediction function

There were considered different values of k (length of sub sequences) and penalty coefficient C. k=4 and k=5 showed approximately the same results.

### Results
My first submissions were examples of working of libraries from scikit-learn. The score 0.60533 logistic regression algorithm showed. SVM with linear kernel gave score 0.62933. After that on kaggle my implementation of SVM with rbf and polynomial kernel were tested. On the private leaderboard the best result SVM with polynomial kernel showed: score 0.68, but on the public leaderboard I have the best result 0.66266, obtained by SVM with rbf kernel (k=4). 


