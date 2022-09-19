import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from QNN import *
from qiskit.visualization import plot_histogram

principal_higgs = np.load('../Data/principal_higgs.npy')
higgs_labels = np.load('../Data/higgs_labels.npy')


X_train, X_test, Y_train, Y_test = train_test_split(principal_higgs, higgs_labels, test_size=0.2, train_size=0.8, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

#minmax = MinMaxScaler(feature_range=(-np.pi/2, np.pi/2))
minmax = MinMaxScaler(feature_range=(0, 1))
X_train = minmax.fit_transform(X_train)
X_test = minmax.fit_transform(X_test)
lamda= 1e-5 ; eta = 0.1 ; n_neuron = 100


theta = np.zeros_like(X_train)                            

newtheta_N, loss_N = train(X_train, Y_train, theta, lr=eta, epochs=10, shots=1000, backend='Nairobi')   

print(newtheta_N, loss_N)
np.save('../Data/theta_N',newtheta_N)
np.save('../Data/loss_N',loss_N)

newtheta_M, loss_M = train(X_train, Y_train, theta, lr=eta, epochs=10, shots=1000, backend='Manila')   

print(newtheta_M, loss_M)
np.save('../Data/theta_M',newtheta_M)
np.save('../Data/loss_M',loss_M)

"""Output 16.09.2022

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [1:37:49<00:00, 586.93s/it]
[[ 0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.  ]
 ...
 [ 0.    0.    0.    0.    0.  ]
 [ 0.    0.    0.    0.    0.  ]
 [ 0.15 -0.05  0.1   0.    0.  ]] [213.5, 194.5, 183.5, 173.5, 161.0, 148.5, 137.0, 135.0, 134.5, 131.5]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [58:13<00:00, 349.38s/it]
[[ 0.    0.    0.    0.    0.  ]
 [ 0.    0.1  -0.1   0.    0.05]
 [ 0.   -0.1   0.1   0.    0.  ]
 ...
 [ 0.    0.    0.    0.    0.  ]
 [-0.2   0.2   0.1   0.05  0.  ]
 [ 0.15 -0.05 -0.1   0.    0.  ]] [185.0, 151.5, 129.0, 117.0, 105.0, 99.0, 92.5, 90.0, 88.5, 87.0]

"""
