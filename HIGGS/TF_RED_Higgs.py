from sklearn.decomposition import PCA
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

pca = PCA(n_components=6, random_state=rs)
principal_higgs = pca.fit_transform(higgs_pca)
print("\n")
print("--"*20)
print("How much information is conserved after PCA with 6 components: ", sum(pca.explained_variance_ratio_))
print("--"*20)
print("\n")
higgs_train, higgs_test, higgs_train_label, higgs_test_label = train_test_split(principal_higgs, higgs_labels, test_size=0.2, train_size=0.8, random_state=42)
higgs_train = np.array(higgs_train)
higgs_test = np.array(higgs_test)
higgs_train_label = np.array(higgs_train_label)
higgs_test_label = np.array(higgs_test_label)

normalize = layers.Normalization()
normalize.adapt(higgs_train)

def NN_model(inputsize, norm, n_layers,n_neuron,eta,lamda):
    model=tf.keras.Sequential([norm])      
    for i in range(n_layers):       #Run loop to add hidden layers to the model
        if (i==0):                  #First layer requires input dimensions
            model.add(layers.Dense(n_neuron,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(lamda),input_dim=inputsize))
        else:                       #Subsequent layers are capable of automatic shape inferencing
            model.add(layers.Dense(n_neuron,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(lamda)))
    model.add(layers.Dense(1,activation='sigmoid'))  #1 outputs - signal or no signal
    sgd=tf.optimizers.SGD(learning_rate=eta)
    model.compile(loss=tf.losses.BinaryCrossentropy(),
                optimizer=sgd,
                metrics = [tf.keras.metrics.BinaryAccuracy()])
    return model
        
def grid_search(n_lays, eta, lamda, n_neuron, eps):
    n_layers = n_lays                                                        #Define number of hidden layers in the model
    epochs = eps                                                             #Number of reiterations over the input data
    Train_accuracy = np.zeros((len(lamda),len(eta),len(n_neuron)))           #Define matrices to store accuracy scores as a function
    Test_accuracy = np.zeros((len(lamda),len(eta),len(n_neuron)))            #of learning rate and number of hidden neurons for 

    print('Go for a stroll, this will take a while')
    for i in range(len(lamda)):                                              #run loops over hidden neurons and learning rates to calculate 
        for j in range(len(eta)):                                            #accuracy scores 
            for k in range(len(n_neuron)):
                print("lambda:",i+1,"/",len(lamda),", eta:",j+1,"/",len(eta),", n_neuron:",k+1,"/",len(n_neuron))
                higgsmodel=NN_model(higgs_train.shape[1], normalize, n_layers,n_neuron[k],eta[j],lamda[i])
                higgsmodel.fit(higgs_train,higgs_train_label,epochs=epochs, verbose=0)
                print("Accuracy test")
                Train_accuracy[i,j,k]=higgsmodel.evaluate(higgs_train,higgs_train_label)[1]
                Test_accuracy[i,j,k]=higgsmodel.evaluate(higgs_test,higgs_test_label)[1]
    
    max=np.max(Train_accuracy)
    train_indices=np.where(Train_accuracy==max)
    print("Best training accuracy:",max)
    print("The parameters are: lambda:",lamda[int(train_indices[0])],", eta:", eta[int(train_indices[1])],"and n_neuron:",n_neuron[int(train_indices[2])])

    max=np.max(Test_accuracy)
    indices=np.where(Test_accuracy==max)
    print("Best testing accuracy:",max)
    print("The parameters are: lambda:",lamda[int(indices[0])],", eta:", eta[int(indices[1])],"and n_neuron:",n_neuron[int(indices[2])])
    return Train_accuracy, Test_accuracy, indices, train_indices


def plot_data(x,y,s,ind,data,title=None):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)
    
    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])  
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]
    
    
    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)
    if s == "l":
        ax.set_ylabel('$\eta$',fontsize=fontsize)
        ax.set_xlabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for $\eta$ and hidden neuron with $\lambda$ = %g' %lamda[int(ind[0])]
        ax.set_title(titlefr)
    elif s =="n":
        ax.set_xlabel('$\eta$',fontsize=fontsize)
        ax.set_ylabel('$\lambda$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for $\eta$ and $\lambda$ with %g hidden neurons' %n_neuron[int(ind[2])]
        ax.set_title(titlefr)
    elif s =="e":
        ax.set_ylabel('$\lambda$',fontsize=fontsize)
        ax.set_xlabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
        titlefr= title[:-2] + 'scores for $\lambda$ and hidden neuron with $\eta$ = %g' %eta[int(ind[1])]
        ax.set_title(titlefr)
        
    plt.tight_layout()
    titlefig = title.replace(' ', '_')
    plt.savefig('../Plots/'+titlefig)
    
    plt.show()
    
    
n_neuron = np.logspace(0,3,4,dtype=int)                               #Define number of neurons per layer
eta = np.logspace(-3, 0, 4)                                           #Define vector of learning rates (parameter to SGD optimiser)
lamda = np.logspace(-5, -2, 4)                                        #Define hyperparameter

Train_accuracy, Test_accuracy, indices, train_ind = grid_search(3, eta, lamda, n_neuron, 10)

plot_data(n_neuron, eta, "l", train_ind, Train_accuracy[int(train_ind[0]),:,:], 'RED Training ne')
plot_data(n_neuron, eta, "l", indices, Test_accuracy[int(indices[0]),:,:], 'RED Testing ne')
plot_data(eta, lamda, "n", train_ind, Train_accuracy[:,:,int(train_ind[2])], 'RED Training le')
plot_data(eta, lamda, "n", indices, Test_accuracy[:,:,int(indices[2])], 'RED Testing le')
plot_data(n_neuron, lamda, "e", train_ind, Train_accuracy[:,int(train_ind[1]),:], 'RED Training nl')
plot_data(n_neuron, lamda, "e", indices, Test_accuracy[:,int(indices[1]),:], 'RED Testing nl')

def PCA_tester(n_components, data_features, data_labels, eta, lamda, n_neuron):
    pca = PCA(n_components, random_state=42)
    higgs_pca = data_features.copy() 
    principal_higgs = pca.fit_transform(higgs_pca)
    info = sum(pca.explained_variance_ratio_)*100
    print("\n")
    print("--"*20)
    print("How much information is conserved after PCA with %g components: %.2f%%" %(n_components, info))
    print("--"*20)
    print("\n")
    higgs_train, higgs_test, higgs_train_label, higgs_test_label = train_test_split(principal_higgs, data_labels, test_size=0.2, train_size=0.8, random_state=42)
    higgs_train = np.array(higgs_train)
    higgs_test = np.array(higgs_test)
    higgs_train_label = np.array(higgs_train_label)
    higgs_test_label = np.array(higgs_test_label)
    norm = layers.Normalization()
    norm.adapt(higgs_train)
    model=NN_model(higgs_train.shape[1], norm, 3, n_neuron, eta, lamda)
    model.fit(higgs_train, higgs_train_label, epochs=10, verbose=0)
    model.save('../Data/%g_RED_Supervised_Higgs'%n_components)
    print('Training accuracy')
    train = model.evaluate(higgs_train,higgs_train_label)[1]
    print('Testing acccuracy')
    test = model.evaluate(higgs_test,higgs_test_label)[1]
    return info, train*100, test*100


n_components = [2,3,4,5,6,7,8,9,10,15,20]
info = np.zeros(len(n_components)+1)
train = np.zeros(len(n_components)+1)
test = np.zeros(len(n_components)+1)
for i,j in zip(n_components, range(len(info))):
    info[j], train[j], test[j] = PCA_tester(int(i), higgs_features, higgs_labels,eta[int(indices[1])], lamda[int(indices[0])], 100)
n_components.append(30)
info[-1] = 100; test[-1] = 84.1; train[-1] = 85.1

plt.figure(figsize=[10,6])
plt.subplot(3,1,1)
plt.plot(n_components, info)
plt.xticks(np.linspace(2,30,29))
plt.ylabel('Information [%]')
plt.subplot(3,1,2)
plt.plot(n_components, train)
plt.xticks(np.linspace(2,30,29))
plt.ylabel('Training accuracy [%]')
plt.subplot(3,1,3)
plt.plot(n_components, test)
plt.xticks(np.linspace(2,30,29))
plt.ylabel('Testing accuracy [%]')
plt.xlabel('Number of features')
plt.savefig('../Plots/RED_info')
plt.show()

""" Result 19.07.2022

...
----------------------------------------
How much information is conserved after PCA with 6 components:  0.9675937999245314
----------------------------------------

...
Best training accuracy: 0.8149999976158142
The parameters are: lambda: 1e-05 , eta: 1.0 and n_neuron: 100
Best testing accuracy: 0.8299999833106995
The parameters are: lambda: 0.001 , eta: 1.0 and n_neuron: 100

...

----------------------------------------
How much information is conserved after PCA with 2 components: 87.02%
----------------------------------------


2022-07-19 12:26:38.281361: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
Training accuracy
25/25 [==============================] - 0s 1ms/step - loss: 0.6834 - binary_accuracy: 0.6900
Testing acccuracy
7/7 [==============================] - 0s 1ms/step - loss: 0.6877 - binary_accuracy: 0.6650


----------------------------------------
How much information is conserved after PCA with 3 components: 91.46%
----------------------------------------


Training accuracy
25/25 [==============================] - 0s 1ms/step - loss: 0.7204 - binary_accuracy: 0.6150
Testing acccuracy
7/7 [==============================] - 0s 2ms/step - loss: 0.6811 - binary_accuracy: 0.6300


----------------------------------------
How much information is conserved after PCA with 4 components: 93.81%
----------------------------------------


Training accuracy
25/25 [==============================] - 0s 1ms/step - loss: 0.6275 - binary_accuracy: 0.7300
Testing acccuracy
7/7 [==============================] - 0s 2ms/step - loss: 0.6208 - binary_accuracy: 0.7450


----------------------------------------
How much information is conserved after PCA with 5 components: 95.34%
----------------------------------------


Training accuracy
25/25 [==============================] - 0s 1ms/step - loss: 0.5776 - binary_accuracy: 0.7638
Testing acccuracy
7/7 [==============================] - 0s 3ms/step - loss: 0.5667 - binary_accuracy: 0.7700


----------------------------------------
How much information is conserved after PCA with 6 components: 96.76%
----------------------------------------


Training accuracy
25/25 [==============================] - 0s 2ms/step - loss: 0.5460 - binary_accuracy: 0.7925
Testing acccuracy
7/7 [==============================] - 0s 1ms/step - loss: 0.5460 - binary_accuracy: 0.8100


----------------------------------------
How much information is conserved after PCA with 7 components: 97.77%
----------------------------------------


Training accuracy
25/25 [==============================] - 0s 1ms/step - loss: 0.5308 - binary_accuracy: 0.8037
Testing acccuracy
7/7 [==============================] - 0s 2ms/step - loss: 0.5276 - binary_accuracy: 0.8200


----------------------------------------
How much information is conserved after PCA with 8 components: 98.58%
----------------------------------------


Training accuracy
25/25 [==============================] - 0s 1ms/step - loss: 0.5665 - binary_accuracy: 0.7887
Testing acccuracy
7/7 [==============================] - 0s 2ms/step - loss: 0.5787 - binary_accuracy: 0.7650


----------------------------------------
How much information is conserved after PCA with 9 components: 99.01%
----------------------------------------


Training accuracy
25/25 [==============================] - 0s 1ms/step - loss: 0.5143 - binary_accuracy: 0.8175
Testing acccuracy
7/7 [==============================] - 0s 2ms/step - loss: 0.5187 - binary_accuracy: 0.8450


----------------------------------------
How much information is conserved after PCA with 10 components: 99.39%
----------------------------------------


Training accuracy
25/25 [==============================] - 0s 2ms/step - loss: 0.5284 - binary_accuracy: 0.8150
Testing acccuracy
7/7 [==============================] - 0s 2ms/step - loss: 0.5417 - binary_accuracy: 0.8200


----------------------------------------
How much information is conserved after PCA with 15 components: 99.98%
----------------------------------------


Training accuracy
25/25 [==============================] - 0s 1ms/step - loss: 0.4742 - binary_accuracy: 0.8625
Testing acccuracy
7/7 [==============================] - 0s 2ms/step - loss: 0.5633 - binary_accuracy: 0.8250


----------------------------------------
How much information is conserved after PCA with 20 components: 100.00%
----------------------------------------


Training accuracy
25/25 [==============================] - 0s 1ms/step - loss: 0.4444 - binary_accuracy: 0.8775
Testing acccuracy
7/7 [==============================] - 0s 2ms/step - loss: 0.6147 - binary_accuracy: 0.8200
"""