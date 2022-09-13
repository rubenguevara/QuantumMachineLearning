from re import I
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

# Replacing -999 (nan) values to mean of the row
higgs_data = pd.read_csv("../Data/training.csv")
higgs_data = pd.DataFrame(higgs_data).replace(-999, np.NaN)
higgs_data = higgs_data.fillna(value=higgs_data.mean())

#To make binary versions of signal or background
higgs_data['Label'] = pd.DataFrame(higgs_data['Label']).replace('s', 1)
higgs_data['Label'] = pd.DataFrame(higgs_data['Label']).replace('b', 0)

higgs_features = higgs_data.copy()
higgs_EventId = higgs_features.pop('EventId')
higgs_Weight = higgs_features.pop('Weight')
higgs_labels = higgs_features.pop('Label')

higgs_train, higgs_test, higgs_train_label, higgs_test_label = train_test_split(higgs_features, higgs_labels, test_size=0.2, train_size=0.8, random_state=42)
higgs_train = np.array(higgs_train)
higgs_test = np.array(higgs_test)
higgs_train_label = np.array(higgs_train_label)
higgs_test_label = np.array(higgs_test_label)
higgs_model = tf.keras.models.load_model('../Data/Supervised_Higgs')

higgs_pred_label = higgs_model.predict(higgs_test).ravel()
test = higgs_test_label
pred = higgs_pred_label


fpr, tpr, thresholds = roc_curve(test, pred, pos_label=1)
roc_auc = auc(fpr,tpr)
plt.figure(1)
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for model on HiggsML dataset')
plt.legend(loc="lower right")
plt.savefig('../Plots/ROC')
plt.show()

plt.figure(2, figsize=[10,8])
lw = 2
n_components = [20,15,10,9,8,7,6,5,4,3,2]
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='All features (area = %0.2f)' % roc_auc)
for i in n_components:
    pca = PCA(n_components=int(i), random_state=42)
    principal_higgs = pca.fit_transform(higgs_features)
    higgspca_train, higgspca_test, higgspca_train_label, higgspca_test_label = train_test_split(principal_higgs, higgs_labels, test_size=0.2, train_size=0.8, random_state=42)
    higgspca_train = np.array(higgspca_train)
    higgspca_test = np.array(higgspca_test)
    higgspca_train_label = np.array(higgspca_train_label)
    higgspca_test_label = np.array(higgspca_test_label)
    higgspca_model = tf.keras.models.load_model('../Data/%g_PCA_Supervised_Higgs'%int(i))
    higgspca_pred_label = higgspca_model.predict(higgspca_test).ravel()
    testpca = higgspca_test_label
    predpca = higgspca_pred_label
    fprpca, tprpca, thresholdspca = roc_curve(testpca, predpca, pos_label=1)
    rocpca_auc = auc(fprpca,tprpca)
    plt.plot(fprpca, tprpca, lw=lw, label='%g features (area = %0.2f)' %(int(i),rocpca_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC PCA comparison')
plt.legend(loc="lower right")
plt.savefig('../Plots/PCA_ROC')
plt.show()


plt.figure(3, figsize=[10,6])
n, bins, patches = plt.hist(pred[test==0], 200, facecolor='blue', alpha=0.2,label="Background")
n, bins, patches = plt.hist(pred[test==1], 200, facecolor='red' , alpha=0.2, label="Signal")
plt.xlabel('TF output')
plt.xlim([0,1])
plt.ylabel('Events')
plt.title('Model output, HiggsML dataset, validation data')
plt.grid(True)
plt.legend()
plt.savefig('../Plots/VAL')
plt.show()

i = 6
pca = PCA(n_components=int(i), random_state=42)
principal_higgs = pca.fit_transform(higgs_features)
higgspca_train, higgspca_test, higgspca_train_label, higgspca_test_label = train_test_split(principal_higgs, higgs_labels, test_size=0.2, train_size=0.8, random_state=42)
higgspca_train = np.array(higgspca_train)
higgspca_test = np.array(higgspca_test)
higgspca_train_label = np.array(higgspca_train_label)
higgspca_test_label = np.array(higgspca_test_label)
higgspca_model = tf.keras.models.load_model('../Data/%g_PCA_Supervised_Higgs'%int(i))
higgspca_pred_label = higgspca_model.predict(higgspca_test).ravel()
testpca = higgspca_test_label
predpca = higgspca_pred_label

plt.figure(4, figsize=[10,6])
n, bins, patches = plt.hist(predpca[testpca==0], 200, facecolor='blue', alpha=0.2,label="Background")
n, bins, patches = plt.hist(predpca[testpca==1], 200, facecolor='red' , alpha=0.2, label="Signal")
plt.xlabel('TF output')
plt.xlim([0,1])
plt.ylabel('Events')
plt.title('PCA Model output with %g features, HiggsML dataset, validation data' %int(i))
plt.grid(True)
plt.legend()
plt.savefig('../Plots/PCA_VAL')
plt.show()

rs=42

s = higgs_data.loc[higgs_data['Label'] == 1]
b = higgs_data.loc[higgs_data['Label'] == 0]
s_pr = s.shape[0]/(s.shape[0]+b.shape[0])
s_n = round(s_pr*1000)

s = s.sample(n =s_n, random_state = rs)
b = b.sample(n =1000-s_n, random_state = rs)

new_higgs = pd.concat([s,b])
new_higgs = new_higgs.sample(frac = 1, random_state = rs).reset_index(drop=True)
higgs_features = new_higgs.copy()
higgs_EventId = higgs_features.pop('EventId')
higgs_Weight = higgs_features.pop('Weight')
higgs_labels = higgs_features.pop('Label')

plt.figure(5, figsize=[10,8])
lw = 2
n_components = [20,15,10,9,8,7,6,5,4,3,2]
fpr6, tpr6, thresholds = roc_curve(testpca, predpca, pos_label=1)
roc_auc6 = auc(fpr6,tpr6)
plt.plot(fpr6, tpr6, color='darkorange', lw=lw, label='PCA 6 Non red. (area = %0.2f)' % roc_auc6)
for i in n_components:
    pca = PCA(n_components=int(i), random_state=42)
    principal_higgs = pca.fit_transform(higgs_features)
    higgspca_train, higgspca_test, higgspca_train_label, higgspca_test_label = train_test_split(principal_higgs, higgs_labels, test_size=0.2, train_size=0.8, random_state=42)
    higgspca_train = np.array(higgspca_train)
    higgspca_test = np.array(higgspca_test)
    higgspca_train_label = np.array(higgspca_train_label)
    higgspca_test_label = np.array(higgspca_test_label)
    higgspca_model = tf.keras.models.load_model('../Data/%g_RED_Supervised_Higgs'%int(i))
    higgspca_pred_label = higgspca_model.predict(higgspca_test).ravel()
    testpca = higgspca_test_label
    predpca = higgspca_pred_label
    fprpca, tprpca, thresholdspca = roc_curve(testpca, predpca, pos_label=1)
    rocpca_auc = auc(fprpca,tprpca)
    plt.plot(fprpca, tprpca, lw=lw, label='%g red. features (area = %0.2f)' %(int(i),rocpca_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.02])
plt.ylim([-0.01, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC PCA comparison')
plt.legend(loc="lower right")
plt.savefig('../Plots/RED_ROC')
plt.show()


i = 5
pca = PCA(n_components=int(i), random_state=42)
principal_higgs = pca.fit_transform(higgs_features)
higgspca_train, higgspca_test, higgspca_train_label, higgspca_test_label = train_test_split(principal_higgs, higgs_labels, test_size=0.2, train_size=0.8, random_state=42)
higgspca_train = np.array(higgspca_train)
higgspca_test = np.array(higgspca_test)
higgspca_train_label = np.array(higgspca_train_label)
higgspca_test_label = np.array(higgspca_test_label)
higgspca_model = tf.keras.models.load_model('../Data/%g_PCA_Supervised_Higgs'%int(i))
higgspca_pred_label = higgspca_model.predict(higgspca_test).ravel()
testpca = higgspca_test_label
predpca = higgspca_pred_label

plt.figure(6, figsize=[10,6])
n, bins, patches = plt.hist(predpca[testpca==0], 20, facecolor='blue', alpha=0.2,label="Background")
n, bins, patches = plt.hist(predpca[testpca==1], 20, facecolor='red' , alpha=0.2, label="Signal")
plt.xlabel('TF output')
plt.xlim([0,1])
plt.ylabel('Events')
plt.title('Reduced Model output with %g features, HiggsML dataset, validation data' %int(i))
plt.grid(True)
plt.legend()
plt.savefig('../Plots/RED_VAL')
plt.show()