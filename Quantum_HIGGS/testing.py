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
minmax = MinMaxScaler(feature_range=(0, 1))
X_train = minmax.fit_transform(X_train)
X_test = minmax.fit_transform(X_test)
lamda= 0.001 ; eta = 1.0 ; n_neuron = 100

theta = np.zeros_like(X_train)

ypred, counts= qnn(X_train, theta, backend = 'nairobi', shots=1000)

def truncate_keys(a, length): 
    return dict((k[:length], v) for k, v in a.items())

qcounts = truncate_keys(counts, len(X_train[0]))  # Only get quantum state as label
plot_histogram(qcounts, figsize=[18,10], filename= '../Plots/QuantumCounts.png')


newtheta, loss = train(X_train, Y_train, theta, eta, 1, shots=1000)  # Takes way too long

print(newtheta, loss)
np.save('theta',newtheta)
