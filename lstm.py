from torchtext import data
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import argparse
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import csv

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(RNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.batchnorm1 = nn.BatchNorm1d(embedding_dim,affine=True)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim, affine=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [sent len, batch size]
        
        embedded = self.dropout(self.batchnorm1(self.embedding(x)))
        
        #embedded = [sent len, batch size, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.batchnorm2(hidden)
        #output = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid. dim]
        #cell = [num layers * num directions, batch size, hid. dim]
        
        #hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        hidden = self.dropout(hidden[-1,:,:])
         
        #hidden [batch size, hid. dim * num directions]
            
        return self.fc(hidden.squeeze(0))

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc

def trainModel(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.feat).squeeze(1)
        #print predictions.shape
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        class_predictions = []
        class_labels = []
        for batch in iterator:

            predictions = model(batch.feat).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)
        
            class_predictions.append(predictions.detach().numpy())
            class_labels.append(batch.label.detach().numpy())

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        class_prediction = np.concatenate(class_predictions, axis=0)
        class_label = np.concatenate(class_labels, axis=0)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), class_prediction, class_label    

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

def main():
    descriptionText = '''Provide train and test data for RNN
                      '''
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description=descriptionText, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # mandatory arguments
    parser.add_argument("-tr", "--trainfile", action='store', dest='trainfile', type=str, required=True, help='Path to the training csv file.')
    parser.add_argument("-te", "--testfile", action='store', dest='testfile', type=str, required=True, help='Path to the testing csv file.')
    parser.add_argument("-ed", "--embdim", action='store', dest='embdim', type=int, required=True, help='Size of embedding dimension')
    parser.add_argument("-hd", "--hiddim", action='store', dest='hiddim', type=int, required=True, help='Size of hidden dimension')
    # optional arguments
    parser.add_argument("-e", "--trainepochs", action='store', dest='epochs', type=int, required=False, default=5, help='Number of epochs for the training phase. Default is 5.')
    parser.add_argument("-b", "--batchsize", action='store', dest='batchsize', type=int, required=False, default=64, help='Batch size for training. Default is 64.')
    args = parser.parse_args()
    
    FEAT = data.Field(tokenize='spacy')
    LABEL = data.LabelField(tensor_type=torch.FloatTensor)
    
    #fields = {'feature': ('f', FEAT), 'label': ('l', LABEL)}
    fields = [('feat', FEAT), ('label', LABEL)]
    
    train, test = data.TabularDataset.splits(
                    path = './',
                    train = args.trainfile,
                    test = args.testfile,
                    format = 'csv',
                    fields = fields
    )
    #train1 = train[0]
    print('vars(train[0]):', vars(train[0]))
    
    FEAT.build_vocab(train, max_size=270)
    LABEL.build_vocab(train, max_size=2)
    
    print('len(FEAT.vocab):', len(FEAT.vocab))
    print('len(LABEL.vocab):', len(LABEL.vocab))
        
    INPUT_DIM = len(FEAT.vocab)
    EMBEDDING_DIM = args.embdim
    HIDDEN_DIM = args.hiddim
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = False
    DROPOUT = 0.2
    
    BATCH_SIZE = args.batchsize
    
    (train_iterator, test_iterator) = data.Iterator.splits(
        (train,test), 
        batch_size=BATCH_SIZE, 
        sort_key=lambda x: len(x.feat), 
        repeat=False)
    #train_iterator = train_iterator[0]
    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    
    optimizer = torch.optim.Adam(model.parameters())
    
    criterion = nn.BCEWithLogitsLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    N_EPOCHS = args.epochs
    
    for epoch in range(N_EPOCHS):
    
        train_loss, train_acc = trainModel(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc, prediction, label = evaluate(model, test_iterator, criterion)
              
        print 'Epoch: {}, Train Loss: {}, Train Acc: {}%, Val. Loss: {}, Val. Acc: {}%'.format(epoch+1, train_loss, train_acc*100, valid_loss, valid_acc*100)
    rounded_preds = torch.round(F.sigmoid(torch.FloatTensor(prediction)))
    rounded_preds = rounded_preds.detach().numpy()
    cnf_matrix = confusion_matrix(label, rounded_preds)
    precision = 1.0*cnf_matrix[0,0]/(cnf_matrix[0,0] + cnf_matrix[1,0])
    recall = 1.0*cnf_matrix[0,0]/(cnf_matrix[0,0] + cnf_matrix[0,1])
    Fs = 1.0*2*precision*recall/(precision + recall)
    save_F = ['new data ', str(Fs)]
    print "F score is {}".format(Fs)
    np.set_printoptions(precision=2)
    class_names = np.array(['Non-Malicious', 'Malicious'])
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
    
