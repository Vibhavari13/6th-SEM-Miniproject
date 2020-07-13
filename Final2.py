import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from sklearn.cluster import KMeans
from SimpSOM import somNet 
from numpy import random
import pandas as pd
import os
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from mpl_toolkits import mplot3d
from SimpSOM import somNet 
from collections import Counter, defaultdict
import seaborn as sns
from tensorflow._api.v1.compat.v1.keras import estimator
result = []
arr = []
clusNo=[]

class HebbRule(object):      
    def train_weights(self, train_data):
        print("Start to train weights.")
        num_data =  len(train_data)
        self.num_neuron = train_data[0]
        
        # initialize weights
        W = np.zeros((self.num_neuron, self.num_neuron))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data*self.num_neuron)
        
        # Hebb rule
        for i in tqdm(range(num_data)):
            t = train_data[i] - rho
            W += np.outer(t, t)
        
        # Make diagonal element of W into 0
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        
        self.W = W 
        print(W)
        n = W.flatten()
        print(n)
        m = np.unique(n)
        a = np.trim_zeros(m)
        result.append(a)

        return(a)
        
        
    def main(self):
        arr = []
        print("The resultant hebbian weights array is as follows:")
        arr = np.array(result)
        print(arr)   
        print("first element in array")
        print(arr[0])   
        print("Number of elements in weights array")
        print(len(arr))
        
        df=pd.DataFrame(result)
        df.dropna(axis=0,inplace=True)
        df.info()
        row_count = 0
        for col in df.index: 
            row_count += 1
        print(row_count)

        
        kmeans5 = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
        x1=kmeans5.fit_predict(df)
        from sklearn import metrics
        labels5 = kmeans5.labels_
        metrics.silhouette_score(df, labels5, metric = 'euclidean')
        metrics.calinski_harabasz_score(df, labels5)
        print(" The silhouette score for K-means 5 is :", metrics.silhouette_score(df, labels5, metric = 'euclidean'))
        print(" The Calinsky-Harabasz score for K-means 5 is :",metrics.calinski_harabasz_score(df, labels5))
        
        kmeans6 = KMeans(n_clusters = 6, init = 'k-means++', random_state = 0)
        x1=kmeans6.fit_predict(df)
        labels6 = kmeans6.labels_
        metrics.silhouette_score(df, labels6, metric = 'euclidean')
        metrics.calinski_harabasz_score(df, labels6)
        print(" The silhouette score for K-means 6 is :", metrics.silhouette_score(df, labels6, metric = 'euclidean'))
        print(" The Calinsky-Harabasz score for K-means 6 is :",metrics.calinski_harabasz_score(df, labels6))

        kmeans3 = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
        x1=kmeans3.fit_predict(df)
        labels = kmeans3.labels_
        print(" The silhouette score for K-means 3 is :", metrics.silhouette_score(df, labels, metric = 'euclidean'))
        print(" The Calinsky score for K-means 3 is :",metrics.calinski_harabasz_score(df, labels))
        
        
        kmeans3 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
        x1=kmeans3.fit_predict(df)
        labels = kmeans3.labels_
        print(" The silhouette score for K-means 4 is :", metrics.silhouette_score(df, labels, metric = 'euclidean'))
        print(" The Calinsky score for K-means 4 is :",metrics.calinski_harabasz_score(df, labels))

        
        
        sum_of_squared_distances = []
        K = range(1,15)
        for k in K:
            k_means = KMeans(n_clusters=k)
            model = k_means.fit(df)
            
            sum_of_squared_distances.append(k_means.inertia_)
            
        plt.plot(K, sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('sum_of_squared_distances')
        plt.title('Elbow method for optimal k')
        plt.show()

        x2 = x1
        print(x2)
        print("This is to test cluster number; The cluster number at index 8 is:")
        print(x2[8])
        
        
        df1= pd.DataFrame(result)
        df1.dropna(axis=0,inplace=True)
        row = 0
        for col in df1.index: 
            row += 1

        i = 0
        for i in range(row):
            clusNo.append(x2[i])
            
        df1['ClusterNumber']=clusNo
        df1.rename( columns={0:'Weights'}, inplace=True )
        
        
        plt.scatter(df1['Weights'],df1['ClusterNumber'])
        plt.xlabel('Hebbian weights')
        plt.ylabel('Cluster numbers')
        plt.show()
        
        x = df1.copy()
        kmeans4 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
        x1=kmeans3.fit_predict(x)
        estimator = KMeans(n_clusters=3)
        estimator.fit(x)
        
        
        clusters = x.copy()
        clusters['clusterpred']=kmeans4.fit_predict(x)
        print(clusters)
        centroids = kmeans4.cluster_centers_
        print('The centroid values are as follows:')
        print(centroids)
        plt.scatter(clusters['Weights'],clusters['ClusterNumber'],c=clusters['clusterpred'],cmap='rainbow')
        plt.scatter(centroids[:,0] ,centroids[:,1], color='black', marker='x')
        plt.xlabel('Hebbian Weights')
        plt.ylabel('Cluster Numbers')
        plt.show()
        
        
        plt.figure('3d viewing', figsize=(7,5))
        ax = plt.axes(projection = '3d')
        ax.scatter(clusters['ClusterNumber'],clusters['Weights'],c=kmeans3.labels_);
        
        k_means = KMeans(n_clusters=4)
        k_means.fit(clusters)
        k_means_predicted = k_means.predict(clusters)
        
        accuracy = round((np.mean(k_means_predicted==kmeans4.labels_))*100)
        print('Accuracy for 3-d viewing:'+str(accuracy))
        
        ax = plt.axes(projection = '3d')
        ax.scatter(clusters['clusterpred'],clusters['Weights'], c=kmeans4.labels_ , cmap='Set2', s=128,alpha=0.8)
        plt.show()

        print('A list of all cluster numbers for all winner nodes is displayed as follows:')
        print(estimator.labels_)
        print("The nodes that belong to each cluster are as follows:")
        print({i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)})

        return(df1['ClusterNumber'])
        
   
    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()
