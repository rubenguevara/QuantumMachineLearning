from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

rs = 42

# Replacing -999 (nan) values to mean of the row
higgs_data = pd.read_csv("../Data/training.csv")
higgs_data = pd.DataFrame(higgs_data).replace(-999, np.NaN)
higgs_data = higgs_data.fillna(value=higgs_data.mean())

#To make binary versions of signal or background
higgs_data['Label'] = pd.DataFrame(higgs_data['Label']).replace('s', 1)
higgs_data['Label'] = pd.DataFrame(higgs_data['Label']).replace('b', 0)

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
higgs_pca = higgs_features.copy()

pca = PCA(n_components=5, random_state=rs)
principal_higgs = pca.fit_transform(higgs_pca)

np.save('../Data/principal_higgs', principal_higgs)
np.save('../Data/higgs_labels', higgs_labels)
