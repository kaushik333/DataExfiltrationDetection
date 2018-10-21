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
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
    
        self.encoder = nn.Sequential(
            nn.Linear(256, 128),
            #nn.ReLU(True),
            nn.Sigmoid(),
    #            nn.Tanh(),
            nn.Linear(128, 64),
        )
        self.decoder = nn.Sequential(
            nn.Sigmoid(),
            #nn.ReLU(True),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            #nn.ReLU(True),
            nn.Linear(128, 256),
    #            nn.Sigmoid(),       # compress to a range (0, 1)
        )
            
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class AE_classifier(nn.Module):
    def __init__(self,*k):
        super(AE_classifier, self).__init__()
    
        self.classifier = nn.Sequential(*k)
            
    def forward(self, x):
        output = self.classifier(x)
        return output

def weights_init(m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
    
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

def softmax_classification(train, classification):
    data_loader = Data.DataLoader(dataset=train, batch_size=1, shuffle=False)
    class_predictions = []
    for step, (x, b_label) in enumerate(data_loader):
        b_x = x.view(-1, 256)   # batch x, shape (batch, 256)    
        output = classification(b_x)
        class_predictions.append(output.detach().numpy())
        
    pred = np.concatenate(class_predictions, axis=0)
    m = nn.Softmax()
    pred_soft = m(torch.Tensor(pred))
    pred_soft = pred_soft.detach().numpy()
    pred_train = np.argmax(pred_soft, axis=1)

    return pred_train


def feature_extraction(autoencoder, data_loader):
    encoded_data = []
    for step, (x, b_label) in enumerate(data_loader):
        b_x = x.view(-1, 256)   # batch x, shape (batch, 256)
    
        encoded, decoded = autoencoder(b_x)
        encoded_data.append(encoded.detach().numpy())
    #    encoded_train_data.append(encoded)
    
    feat_train = np.concatenate(encoded_data, axis=0)
    
    return feat_train
    
def main():

    descriptionText = '''Provide NGRam file and packet-wise labelled csv file for training an Autoencoder.
    '''
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description=descriptionText, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # mandatory arguments
    parser.add_argument("-i", "--infile", action='store', dest='ngramfile', type=str, required=True, help='Path to the test input ngram csv file.')
    parser.add_argument("-l", "--labelfile", action='store', dest='outfile', type=str, required=True, help='Path to the test input packet label csv file.')
    parser.add_argument("-aep", "--aepath", action='store', dest='aepath', type=str, required=True, help='Path to load autoencoder model')
    parser.add_argument("-cp", "--classpath", action='store', dest='classpath', type=str, required=True, help='Path to load classification model')
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
    
    #autoencoder = torch.load(args.aepath)
    
    X_new = torch.FloatTensor(n_gram)
    y_new = torch.tensor(labels)   
    test = data_utils.TensorDataset(X_new, y_new)
    
    if args.classType=="Softmax":
        classification = torch.load(args.classpath)
        pred_new = softmax_classification(test, classification)

        cnf_matrix = confusion_matrix(y_new.detach().numpy(), pred_new)
        print cnf_matrix
        precision = 1.0*cnf_matrix[0,0]/(cnf_matrix[0,0] + cnf_matrix[1,0])
        recall = 1.0*cnf_matrix[0,0]/(cnf_matrix[0,0] + cnf_matrix[0,1])
        Fs = 1.0*2*precision*recall/(precision + recall)
        save_F = ['new data ', str(Fs)]
        print "F score is {}".format(Fs)
        np.set_printoptions(precision=2)
        class_names = np.array(['Malicious','Non-Malicious'])
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                              title='New data confusion matrix')
        plt.savefig('newdata.png')
        f = open('./f-scores.csv', 'a')
        writer = csv.writer(f)
        writer.writerow(save_F)
        f.close()


if __name__ == "__main__":
    main()
