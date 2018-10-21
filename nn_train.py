import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import numpy as np
import csv
import torch.utils.data as data_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import itertools
import matplotlib.pyplot as plt
import sys
import argparse
    
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
    
        self.classify = nn.Sequential(
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 2),
            #nn.Sigmoid()
        )                 
    
    def forward(self, x):
        classification = self.classify(x)
        return classification

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def train_network(X_train, y_train, BATCH_SIZE, EPOCH, LR):
    classification = NeuralNet()
    
    optimizer = torch.optim.Adam(classification.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    train = data_utils.TensorDataset(X_train, y_train)
    data_loader = Data.DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCH):
        runningLoss = 0.0
        for step, (x, b_label) in enumerate(data_loader):
            b_x = x.view(-1, 256)   # batch x, shape (batch, 256)    
            output = classification(b_x)
    
            loss = loss_func(output, b_label)      # mean square error
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()  
            runningLoss += loss.data[0] 
            
        print("At epoch {}, the loss is {}".format(epoch+1, runningLoss))
        
    return classification

def prediction(classification, X_test):
    train = data_utils.TensorDataset(X_test, y_test)
    data_loader = Data.DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=False)
    class_predictions = []
    for step, (x, b_label) in enumerate(data_loader):
        b_x = x.view(-1, 256)   # batch x, shape (batch, 256)    
        output = classification(b_x)
        class_predictions.append(output.detach().numpy())

def main():

    descriptionText = '''Provide NGRam file and packet-wise labelled csv file for training an Autoencoder.
    '''
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description=descriptionText, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # mandatory arguments
    parser.add_argument("-i", "--infile", action='store', dest='ngramfile', type=str, required=True, help='Path to the input ngram csv file.')
    parser.add_argument("-l", "--labelfile", action='store', dest='outfile', type=str, required=True, help='Path to the input packet label csv file.')

    # optional arguments
    parser.add_argument("-e", "--trainepochs", action='store', dest='epochs', type=int, required=False, default=20, help='Number of epochs for the training phase. Default is 20.')
    parser.add_argument("-b", "--batchsize", action='store', dest='batchsize', type=int, required=False, default=32, help='Batch size for training. Default is 32.')
    parser.add_argument("-lr", "--learnrate", action='store', dest='lr', type=float, required=False, default=1e-3, help='Learning Rate. Default is 1e-3.')
    parser.add_argument("-c", "--classificationType", action='store', dest='classType', type=str, required=False, default="Softmax", help='Choose classification type: Softmax, RF. Default is Softmax.')
    args = parser.parse_args()
    
    # Hyper Parameters
    EPOCH = args.epochs
    BATCH_SIZE = args.batchsize
    LR = args.lr
    #BETA = 0.5
    #RHO = 0.01
    if(args.classType != "Softmax" and args.classType != "RF"):
        print("Invalid classification type. Quit.")
        sys.exit(1)
    ######################################
    ### LOAD NGRAMS
    ######################################
    n_gram=[]
    with open(args.ngramfile) as f:
        websites = csv.reader(f)
        for r in websites:
            n_gram.append([float(i) for i in r])
    print len(n_gram)        
    ##########################
    ### LOAD LABELS
    ##########################
    labels=[]
    with open(args.outfile) as f:
        lab = csv.reader(f)
        for r in lab:
            labels.append(int(float(r[0])))
            
    X_train, X_test, y_train, y_test = train_test_split(n_gram, labels, test_size=0.3)      
    #X_crossval, X_test, y_crossval, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)  
            
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)        
    train = data_utils.TensorDataset(X_train, y_train)
    data_loader = Data.DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)      
    
    ##########################################################
    ### PERFORM TRAINING
    ##########################################################
    classification = train_network(X_train, y_train, BATCH_SIZE, EPOCH, LR)
    torch.save(classification, './train_data/yahoo_32k/trained_nn.pt')

if __name__ == "__main__":
    main()

