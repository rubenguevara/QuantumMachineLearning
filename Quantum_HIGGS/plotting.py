import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from QNN import *
from qiskit.visualization import plot_histogram
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

principal_higgs = np.load('../Data/principal_higgs.npy')
higgs_labels = np.load('../Data/higgs_labels.npy')
theta_N = np.load('../Data/theta_N.npy')
theta_M = np.load('../Data/theta_M.npy')
loss_N = np.load('../Data/loss_N.npy')
loss_M = np.load('../Data/loss_M.npy')


X_train, X_test, Y_train, Y_test = train_test_split(principal_higgs, higgs_labels, test_size=0.2, train_size=0.8, random_state=42)
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

minmax = MinMaxScaler(feature_range=(0, 1))
X_train = minmax.fit_transform(X_train)
X_test = minmax.fit_transform(X_test)

backend1 = 'Nairobi'
backend2 = 'Manila'

ypred, counts1= qnn(X_train[0], theta_N[0], backend1, shots=1000, visual=True)
ypred, counts2= qnn(X_train[0], theta_M[0], backend2, shots=1000, visual=True)

filename = '../Plots/QuantumCounts.png'
lgnd = [backend1, backend2]
plot_histogram([counts1, counts2], legend=lgnd, figsize=[15,8], filename= filename, bar_labels=False)
plt.show()

y_pred_N_train = np.zeros(len(Y_train))
y_pred_M_train = np.zeros(len(Y_train))
for i in tqdm(range(len(Y_train))):
    y_pred_N_train[i], c = qnn(X_train[i], theta_N[i], backend1, shots=1000)
    y_pred_M_train[i], c = qnn(X_train[i], theta_M[i], backend2, shots=1000)

y_pred_N_test = np.zeros(len(Y_test))
y_pred_M_test = np.zeros(len(Y_test))
for i in tqdm(range(len(Y_test))):
    y_pred_N_test[i], c = qnn(X_test[i], theta_N[i], backend1, shots=1000)
    y_pred_M_test[i], c = qnn(X_test[i], theta_M[i], backend2, shots=1000)


weight_train = np.ones(len(Y_train))
weight_train[Y_train==0] = np.sum(weight_train[Y_train==1])/np.sum(weight_train[Y_train==0])

weight_test = np.ones(len(Y_test))
weight_test[Y_test==0] = np.sum(weight_test[Y_test==1])/np.sum(weight_test[Y_test==0])


fpr_N_train, tpr_N_train, thresholds = roc_curve(Y_train, y_pred_N_train, pos_label=1, sample_weight=weight_train)
roc_auc_N_train = auc(fpr_N_train, tpr_N_train)

fpr_N_test, tpr_N_test, thresholds = roc_curve(Y_test, y_pred_N_test, sample_weight=weight_test)
roc_auc_N_test = auc(fpr_N_test, tpr_N_test)

fpr_M_test, tpr_M_test, thresholds = roc_curve(Y_test, y_pred_M_test, pos_label=1, sample_weight=weight_test)
roc_auc_M_test = auc(fpr_M_test, tpr_M_test)

fpr_M_train, tpr_M_train, thresholds = roc_curve(Y_train, y_pred_M_train, pos_label=1, sample_weight=weight_train)
roc_auc_M_train = auc(fpr_M_train, tpr_M_train)

plt.figure(2)
lw = 2
plt.plot(fpr_N_train, tpr_N_train, lw=lw, label='Training set (area = %0.2f)' % roc_auc_N_train)
plt.plot(fpr_N_test, tpr_N_test, lw=lw, label='Testing set (area = %0.2f)' % roc_auc_N_test)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for model on HiggsML dataset using Nairobi')
plt.legend(loc="lower right")
plt.savefig('../Plots/QuantumROC_Nairobi')
plt.show()

plt.figure(3)
lw = 2
plt.plot(fpr_M_train, tpr_M_train, lw=lw, label='Training set (area = %0.2f)' % roc_auc_M_train)
plt.plot(fpr_M_test, tpr_M_test, lw=lw, label='Testing set (area = %0.2f)' % roc_auc_M_test)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for model on HiggsML dataset using Manila')
plt.legend(loc="lower right")
plt.savefig('../Plots/QuantumROC_Manila')
plt.show()



epochs = np.linspace(0,9,10)
plt.figure(4)
plt.plot(epochs, loss_M, label='Loss function for Manila')
plt.plot(epochs, loss_N, label='Loss function for Nairobi')
plt.xlabel('Epochs')
plt.legend()
plt.ylabel('Loss')
plt.savefig('../Plots/QuantumLoss')
plt.show()

"""Result 19.09.2022

        ┌──────────────┐     ┌───────┐                            ░ ┌─┐            
  q0_0: ┤ Rx(0.021829) ├──■──┤ Ry(0) ├────────────────────────────░─┤M├────────────
        ├─────────────┬┘┌─┴─┐└───────┘┌───────┐                   ░ └╥┘┌─┐         
  q0_1: ┤ Rx(0.24747) ├─┤ X ├────■────┤ Ry(0) ├───────────────────░──╫─┤M├─────────
        ├─────────────┤ └───┘  ┌─┴─┐  └───────┘┌───────┐          ░  ║ └╥┘┌─┐      
  q0_2: ┤ Rx(0.27525) ├────────┤ X ├──────■────┤ Ry(0) ├──────────░──╫──╫─┤M├──────
        ├─────────────┤        └───┘    ┌─┴─┐  └───────┘┌───────┐ ░  ║  ║ └╥┘┌─┐   
  q0_3: ┤ Rx(0.36785) ├─────────────────┤ X ├──────■────┤ Ry(0) ├─░──╫──╫──╫─┤M├───
        └┬───────────┬┘                 └───┘    ┌─┴─┐  ├───────┤ ░  ║  ║  ║ └╥┘┌─┐
  q0_4: ─┤ Rx(0.282) ├───────────────────────────┤ X ├──┤ Ry(0) ├─░──╫──╫──╫──╫─┤M├
         └───────────┘                           └───┘  └───────┘ ░  ║  ║  ║  ║ └╥┘
meas: 5/═════════════════════════════════════════════════════════════╩══╩══╩══╩══╩═
                                                                     0  1  2  3  4 
        ┌──────────────┐     ┌───────┐                            ░ ┌─┐            
 q18_0: ┤ Rx(0.021829) ├──■──┤ Ry(0) ├────────────────────────────░─┤M├────────────
        ├─────────────┬┘┌─┴─┐└───────┘┌───────┐                   ░ └╥┘┌─┐         
 q18_1: ┤ Rx(0.24747) ├─┤ X ├────■────┤ Ry(0) ├───────────────────░──╫─┤M├─────────
        ├─────────────┤ └───┘  ┌─┴─┐  └───────┘┌───────┐          ░  ║ └╥┘┌─┐      
 q18_2: ┤ Rx(0.27525) ├────────┤ X ├──────■────┤ Ry(0) ├──────────░──╫──╫─┤M├──────
        ├─────────────┤        └───┘    ┌─┴─┐  └───────┘┌───────┐ ░  ║  ║ └╥┘┌─┐   
 q18_3: ┤ Rx(0.36785) ├─────────────────┤ X ├──────■────┤ Ry(0) ├─░──╫──╫──╫─┤M├───
        └┬───────────┬┘                 └───┘    ┌─┴─┐  ├───────┤ ░  ║  ║  ║ └╥┘┌─┐
 q18_4: ─┤ Rx(0.282) ├───────────────────────────┤ X ├──┤ Ry(0) ├─░──╫──╫──╫──╫─┤M├
         └───────────┘                           └───┘  └───────┘ ░  ║  ║  ║  ║ └╥┘
meas: 5/═════════════════════════════════════════════════════════════╩══╩══╩══╩══╩═
                                                                     0  1  2  3  4 
"""