import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as d
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sn
from preprocessing import main as preprocess
from sklearn.metrics import confusion_matrix
import spacy
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
import torchdata.datapipes as dp
from gensim.models import Word2Vec

vectorizer = Word2Vec.load('./dataset/word2vec_million_cybersecurity_docs_models/model/1million.word2vec.model')

class CTIDataset(Dataset):
    def __init__(self):
        self.df = preprocess()
        self.padding = self.df.combined_processed.str.len().max()
        print(self.padding)
        self.labels = self.df['label']
        self.data = self.df['combined_processed'] 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        vectorized = [torch.tensor(vectorizer.wv[word]) for word in data if word in vectorizer.wv.key_to_index.keys()]
        vectorized = torch.stack(vectorized)
        vectorized = torch.nn.functional.pad(vectorized, (0,self.padding))
        return vectorized, label
        
#model parameters
batch_size = 128
learning_rate = 0.01
input_size = 100
hidden_size = 50
layers = 3
num_classes = 2
num_epochs = 20

# define the models
class LSTM_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, 1)
        self.dropout = nn.Dropout(0.125)
    def forward(self,x):
        x = self.lstm(x)
        x = F.relu(self.fc(x))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x
    
class GRU_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, 1)
        self.dropout = nn.Dropout(0.125)
    def forward(self,x):
        x = self.gru(x)
        x = F.relu(self.fc(x))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x
    
class combined_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, layers, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(1, hidden_size, layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(4*hidden_size, num_classes)
        self.dropout = nn.Dropout(0.125)
    def forward(self,x):
        lstm_out = self.lstm(x)
        gru_out = self.gru(x)
        concat = torch.cat((lstm_out[:, -1, :], gru_out[:, -1, :]), dim=1)
        x = self.dropout(F.relu(self.fc(concat)))
        return x

#check device for torch    
device = 'cpu'   
if torch.cuda.is_available():
    device = 'cuda'
print('training on: ' + str(device))


dst = CTIDataset()

#split dataset
train_dataset, test_dataset = d.random_split(dst, [0.8, 0.2])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,
                                        shuffle=True,)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,
                                        shuffle=False)


#set up the model, loss and optimizer
model = LSTM_model().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay = 0.1)
model.train()
#initialise lists for data to graph
trainLoss,testLosses, accuracies, all_preds, all_labels = [],[],[],[],[]

#Train for number of epochs
for epoch in range(num_epochs): 
    runningLoss = 0
    testLoss = 0
    correct = 0
    model.train()
    #run a training cycle for each image in the batch
    for text,labels in train_loader:
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        yPred = model(text)
        loss = loss_fn(yPred,labels)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()*text.size(0)
    #validate for each image in the test
    model.eval()
    for text,labels in test_loader:
        text, labels = text.to(device), labels.to(device)
        yPred = model(text)
        loss = loss_fn(yPred,labels)
        testLoss += loss.item()*text.size(0)
        correct += (yPred.argmax(1) == labels).type(torch.float).sum().item()
        #add predictions and labels to lists for confusion matrix
        all_preds.extend(yPred.argmax(1).cpu().numpy())  # Convert predictions to numpy array
        all_labels.extend(labels.cpu().numpy()) 
    #calculate losses and accuracy
    runningLoss = runningLoss/len(train_loader.sampler)
    testLoss = testLoss/len(test_loader.sampler)
    accuracy = 100* correct/len(test_dataset)
    trainLoss.append(runningLoss)
    #append data for graphs
    testLosses.append(testLoss)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:,.6f} \tValidation Accuracy: {:}'.format(epoch+1,runningLoss,testLoss, accuracy))


#plot the training and test loss
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), trainLoss, label='Train Loss')
plt.plot(range(num_epochs), testLosses, label='Test Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
 
plt.clf()
# Plot the confusion matrix as a heatmap
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sn.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(2), yticklabels=range(2))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
