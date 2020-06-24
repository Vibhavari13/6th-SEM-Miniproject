# Importing the libraries
import numpy as np
import itertools
from matplotlib.gridspec import GridSpec
np.random.seed(1)
import matplotlib.pyplot as plt
import pandas as pd
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
import Final22
from neupy import algorithms
from pandas import read_csv
from collections import defaultdict
import os
import sklearn.metrics
from SimpSOM import somNet

# Importing the dataset 
script_dir = os.path.dirname(__file__)
training_set_path = os.path.join(script_dir, 'diabetes.csv')
dataset = pd.read_csv(training_set_path)
X = dataset.iloc[:, :-1].values
y=dataset.iloc[:,-1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 9))
X = sc.fit_transform(X)
y = dataset.iloc[:, -1].values
print(X)

# Training the SOM
from minisom import MiniSom

som = MiniSom(x=10, y=10, input_len=len(X.T),learning_rate=1.6,random_seed=10)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=500)
qt=som.quantization_error(X)
print("error:",qt)


# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
print("this is enumerateX",X)
win=[]
for i, x in enumerate(X):
    w = som.winner(x)
    win.append(w)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

mappings = som.win_map(X)
ids = np.concatenate((mappings[(8,7)], mappings[(7,7)]), axis=0)
print("These are the ids of winner nodes:",ids)

dataset['winner']=win
print(dataset.head())

plt.figure(figsize=(8, 7))
frequencies = som.activation_response(X)
plt.pcolor(frequencies.T, cmap='Blues') 
plt.colorbar()
plt.title("Frequency of choosing each winner neurons",fontsize=20);
plt.show()

print("length of X:",len(X))



labels_map = som.labels_map(X, y)
label_names = np.unique(y)

plt.figure(figsize=(10, 10))
the_grid = GridSpec(10, 10)
for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names]
    plt.subplot(the_grid[6-position[1], position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)
plt.legend(patches, label_names, bbox_to_anchor=(5, 9), ncol=3)
plt.show()

names = ['preg', 'gluc', 'BP', 'skin', 'Insulin', 'BMI', 'pedi', 'age']
correlations = dataset.corr()
print(correlations)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.title('Correlations Matrix',fontsize=20);
plt.show()
cor = defaultdict(list)


def main():
    # Load data 
    # Create Network Model
   
    model = Final22.HebbRule()
    E=[]
    for i, x in enumerate(X):
        w = som.winner(x)
        e=model.train_weights(w)
        E.append(list(e))
    
    
    dataset['Weights']= E
    print(dataset.head())

    print("show network weights matrix")
    model.plot_weights()

    print("Show Hebbian weights and further perform clustering.")
    r=model.main()

    dataset['Severity of diabetes']=r

    dataset['Severity of diabetes'].replace(to_replace =0, 
                 value ="LOW",inplace=True)
    dataset['Severity of diabetes'].replace(to_replace =1, 
                 value ="HIGH",inplace=True)
    dataset['Severity of diabetes'].replace(to_replace =2, 
                 value ="MODERATE",inplace=True)
    dataset.dropna(axis=0,inplace=True)
    dataset.reset_index(drop=True,inplace=True)

    print(dataset.head(10))
        

if __name__ == '__main__':
    main()
