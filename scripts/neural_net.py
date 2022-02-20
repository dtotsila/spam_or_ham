import torch
from torch.utils.data import DataLoader
from utils import CustomDataset, read_data, plot_cf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import torch.nn as nn
import torch.nn.functional as F

train = False
# Read Test Data
X_train, y_train = read_data('train')
X_test, y_test = read_data('test')
train_dataset = CustomDataset(X_train, y_train)
print(X_train.shape)
print(X_test.shape)

dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# Hyper Parameters
epochs = 500

# input_dim = tfidf vector length
input_dim = X_train.shape[1]

# 1 binary output (binary classification)
output_dim = 1 
learning_rate = 1e-3
class BinaryClassification(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassification, self).__init__()        
        self.layer_1 = nn.Linear(input_dim, 512) 
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 32)
        self.layer_out = nn.Linear(32, 1) 
        self.relu = nn.ReLU()
        
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = torch.sigmoid(self.layer_out(x))
        
        
        return x

model = BinaryClassification(input_dim)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
batches_per_epoch = X_train.shape[0]//128

if train:
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, output]
            inputs, outputs = data
            # print(inputs)
            batch_loss = 0.0
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = model(inputs.view(-1, input_dim))
            # print(output)
            loss = criterion( output, outputs.view(-1, output.shape[1]))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            # scheduler.step(loss.item())
        # scheduler.step()
        train_mean_loss = running_loss/batches_per_epoch
        
        print("Epoch: ", epoch+1, "Loss: ", train_mean_loss)
    print('Finished Training')
    torch.save(model.state_dict(),'models/nn.pt')
else:
    model.load_state_dict(torch.load('models/nn.pt'))
with torch.no_grad():
    model.eval()
    y_pred = []
    for i in range(X_test.shape[0]):
        # round to 0 or 1 so that class is determined
        output =torch.round(model(X_test[i])).detach().numpy()
        y_pred.append(output)
    plot_cf(confusion_matrix(y_test,y_pred),"Neural Network")
    print(confusion_matrix(y_test, y_pred)) 
    print(classification_report(y_test,y_pred)) 
    print(accuracy_score(y_test,y_pred))
