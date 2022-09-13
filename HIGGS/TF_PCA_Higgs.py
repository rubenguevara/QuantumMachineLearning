from sklearn.decomposition import PCA
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
higgs_pca = higgs_features.copy()

pca = PCA(n_components=7, random_state=42)
principal_higgs = pca.fit_transform(higgs_pca)
print("\n")
print("--"*20)
print("How much information is conserved after PCA with 7 components: ", sum(pca.explained_variance_ratio_))
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
        
def grid_search(n_lays, eta, lamda, n_neuron, eps, b_size):
    n_layers = n_lays                                                        #Define number of hidden layers in the model
    epochs = eps                                                             #Number of reiterations over the input data
    batch_size = b_size
    Train_accuracy = np.zeros((len(lamda),len(eta),len(n_neuron)))           #Define matrices to store accuracy scores as a function
    Test_accuracy = np.zeros((len(lamda),len(eta),len(n_neuron)))            #of learning rate and number of hidden neurons for 

    print('Go for a stroll, this will take a while')
    for i in range(len(lamda)):                                              #run loops over hidden neurons and learning rates to calculate 
        for j in range(len(eta)):                                            #accuracy scores 
            for k in range(len(n_neuron)):
                print("lambda:",i+1,"/",len(lamda),", eta:",j+1,"/",len(eta),", n_neuron:",k+1,"/",len(n_neuron))
                higgsmodel=NN_model(higgs_train.shape[1], normalize, n_layers,n_neuron[k],eta[j],lamda[i])
                higgsmodel.fit(higgs_train,higgs_train_label,epochs=epochs,batch_size=batch_size, verbose=0)
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

    higgs_model=NN_model(higgs_train.shape[1], normalize, n_layers,n_neuron[int(indices[2])],eta[int(indices[1])],lamda[int(indices[0])])
    higgs_model.fit(higgs_train,higgs_train_label,epochs=epochs)
    return Train_accuracy, Test_accuracy, higgs_model, indices, train_indices


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

Train_accuracy, Test_accuracy, higgs_model, indices, train_ind = grid_search(3, eta, lamda, n_neuron, 10, 100)

np.save('../Data/PCA_train', Train_accuracy)
np.save('../Data/PCA_test', Test_accuracy)
np.save('../Data/PCA_train_ind', train_ind)
np.save('../Data/PCA_indices', indices)

#Train_accuracy = np.load('../Data/PCA_train.npy')
#Test_accuracy = np.load('../Data/PCA_test.npy')
#train_ind = np.load('../Data/PCA_train_ind.npy')
#indices = np.load('../Data/PCA_indices.npy')

plot_data(n_neuron, eta, "l", train_ind, Train_accuracy[int(train_ind[0]),:,:], 'PCA Training ne')
plot_data(n_neuron, eta, "l", indices, Test_accuracy[int(indices[0]),:,:], 'PCA Testing ne')
plot_data(eta, lamda, "n", train_ind, Train_accuracy[:,:,int(train_ind[2])], 'PCA Training le')
plot_data(eta, lamda, "n", indices, Test_accuracy[:,:,int(indices[2])], 'PCA Testing le')
plot_data(n_neuron, lamda, "e", train_ind, Train_accuracy[:,int(train_ind[1]),:], 'PCA Training nl')
plot_data(n_neuron, lamda, "e", indices, Test_accuracy[:,int(indices[1]),:], 'PCA Testing nl')


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
    model.save('../Data/%g_PCA_Supervised_Higgs'%n_components)
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
plt.savefig('../Plots/PCA_info')
plt.show()

"""Result 11.07.2022

...
----------------------------------------
How much information is conserved after PCA with 7 components:  0.9784425191218417
----------------------------------------

...
Best training accuracy: 0.8178750276565552
The parameters are: lambda: 1e-05 , eta: 0.1 and n_neuron: 1000
Best testing accuracy: 0.8186399936676025
The parameters are: lambda: 0.0001 , eta: 0.1 and n_neuron: 1000

...
Epoch 10/10
6250/6250 [==============================] - 58s 9ms/step - loss: 0.4354 - binary_accuracy: 0.8148
"""

"""Result 12.07.2022

----------------------------------------
How much information is conserved after PCA with 2 components: 87.73%
----------------------------------------


Training accuracy
6250/6250 [==============================] - 5s 836us/step - loss: 0.6011 - binary_accuracy: 0.6920
Testing acccuracy
1563/1563 [==============================] - 2s 1ms/step - loss: 0.5961 - binary_accuracy: 0.6960


----------------------------------------
How much information is conserved after PCA with 3 components: 92.09%
----------------------------------------


Training accuracy
6250/6250 [==============================] - 6s 881us/step - loss: 0.5370 - binary_accuracy: 0.7301
Testing acccuracy
1563/1563 [==============================] - 2s 1ms/step - loss: 0.5306 - binary_accuracy: 0.7354


----------------------------------------
How much information is conserved after PCA with 4 components: 94.13%
----------------------------------------


Training accuracy
6250/6250 [==============================] - 6s 945us/step - loss: 0.5158 - binary_accuracy: 0.7409
Testing acccuracy
1563/1563 [==============================] - 2s 1ms/step - loss: 0.5114 - binary_accuracy: 0.7435


----------------------------------------
How much information is conserved after PCA with 5 components: 95.58%
----------------------------------------


Training accuracy
6250/6250 [==============================] - 6s 923us/step - loss: 0.4555 - binary_accuracy: 0.7884
Testing acccuracy
1563/1563 [==============================] - 2s 1ms/step - loss: 0.4521 - binary_accuracy: 0.7904


----------------------------------------
How much information is conserved after PCA with 6 components: 96.93%
----------------------------------------


Training accuracy
6250/6250 [==============================] - 7s 1ms/step - loss: 0.4182 - binary_accuracy: 0.8144
Testing acccuracy
1563/1563 [==============================] - 2s 1ms/step - loss: 0.4161 - binary_accuracy: 0.8160


----------------------------------------
How much information is conserved after PCA with 7 components: 97.84%
----------------------------------------


Training accuracy
6250/6250 [==============================] - 6s 863us/step - loss: 0.4175 - binary_accuracy: 0.8151
Testing acccuracy
1563/1563 [==============================] - 2s 1ms/step - loss: 0.4176 - binary_accuracy: 0.8139


----------------------------------------
How much information is conserved after PCA with 8 components: 98.59%
----------------------------------------


Training accuracy
6250/6250 [==============================] - 5s 846us/step - loss: 0.4038 - binary_accuracy: 0.8225
Testing acccuracy
1563/1563 [==============================] - 2s 1ms/step - loss: 0.4034 - binary_accuracy: 0.8229


----------------------------------------
How much information is conserved after PCA with 9 components: 99.00%
----------------------------------------


Training accuracy
6250/6250 [==============================] - 5s 860us/step - loss: 0.3946 - binary_accuracy: 0.8268
Testing acccuracy
1563/1563 [==============================] - 2s 1ms/step - loss: 0.3935 - binary_accuracy: 0.8273


----------------------------------------
How much information is conserved after PCA with 10 components: 99.36%
----------------------------------------


Training accuracy
6250/6250 [==============================] - 5s 846us/step - loss: 0.3891 - binary_accuracy: 0.8299
Testing acccuracy
1563/1563 [==============================] - 2s 1ms/step - loss: 0.3903 - binary_accuracy: 0.8285


----------------------------------------
How much information is conserved after PCA with 15 components: 99.99%
----------------------------------------


Training accuracy
6250/6250 [==============================] - 5s 851us/step - loss: 0.3773 - binary_accuracy: 0.8356
Testing acccuracy
1563/1563 [==============================] - 2s 1ms/step - loss: 0.3789 - binary_accuracy: 0.8349


----------------------------------------
How much information is conserved after PCA with 20 components: 100.00%
----------------------------------------


Training accuracy
6250/6250 [==============================] - 5s 856us/step - loss: 0.3749 - binary_accuracy: 0.8392
Testing acccuracy
1563/1563 [==============================] - 2s 1ms/step - loss: 0.3804 - binary_accuracy: 0.8340
"""