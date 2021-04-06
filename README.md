This repository contains the code for implementing SOM-Hebb model on a Diabetes dataset

The above classifier classifies the dataset based on the serverity of the condition, precisely into high, moderate and low categories.

In this project we aim to perform data segregation by means of two very well known
unsupervised Neural network algorithms, namely Hebbian and Kohonen. These
algorithms endeavor to recognize underlying relationships in a set of data through a
process that mimics the way the human brain operates. Data clustering has proven to
be a well established way to identify similar groups of objects with common features. By
the use of a competitive learning technique such as kohonen we have mapped our high
dimensional dataset to nearby nodes in the 2D space.Thus, each winner node
represents a small segment of the dataset. Our dataset consisting of 768 elements with
8 feature values is now being mapped on a 10*10 2D-grid. These winner neurons form
the first hidden layer of the neural network. The net input for the next hidden layer will
be the updated weights from the kohonen network. The second hidden layer uses the
Hebbian rule according to which simultaneous activation of neurons leads to
pronounced increase in synaptic strength between them. So it is used as an associator
to establish associations between patterns. The output layer consists of clusters,which
are formed by the K-Means algorithm with corresponding similar weights updated by the
hidden Hebbian layer which will be the final classification of the input data. The
incorporation of two hidden layers ensures greater accuracy and pattern association.
This report proposes that the Hebbian network optimizes the clustering ability of
Kohonen in a way to find hidden correlations between the clusters.
Diabetes being a prevalent metabolic disease in women as well as men affect women in
a slightly different manner due to many weighted factors such as hormones,
inflammations and pregnancies. We have segregated this dataset into high, moderate
and mild categories which indicates the level of diabetes each patient is showing.



