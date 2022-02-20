import torch
from torch.utils.data import DataLoader
from utils import  CustomDataset, read_data, plot_cf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
train = True

# Simple Logistic Regresion Module (sigmoid activated)
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim)     
    
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

# Read Test Data
X_train, y_train = read_data('train')
X_test, y_test = read_data('test')
train_dataset = CustomDataset(X_train, y_train)

# Train in batches 
dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
batches_per_epoch = X_train.shape[0]//128

# Hyper Parameters
epochs = 1000
learning_rate = 1e-2

# 1 binary output (binary classification)
output_dim = 1 
input_dim = X_test.shape[1]
model = LogisticRegression(input_dim,output_dim)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if train:
    # train loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, outputs = data
            batch_loss = 0.0
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(inputs.view(-1, input_dim))
            loss = criterion( output, outputs.view(-1, output.shape[1]))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_mean_loss = running_loss/batches_per_epoch
        print("Epoch: ", epoch+1, "Loss: ", train_mean_loss)
    print('Finished Training')

    # store trained model
    torch.save(model.state_dict(),'models/logistic_regression.pt')
else: 
    model.load_state_dict(torch.load('models/logistic_regression.pt'))
model.eval()
# test
# no derivative calculation so that the calculations are faster
with torch.no_grad():
    y_pred = []
    for i in range(X_test.shape[0]):
        # round to 0 or 1 so that class is determined
        output =torch.round(model(X_test[i])).detach().numpy()
        y_pred.append(output)
    plot_cf(confusion_matrix(y_test,y_pred),"Logistic Regression")
    print(confusion_matrix(y_test, y_pred)) 
    print(classification_report(y_test,y_pred)) 
    print(accuracy_score(y_test,y_pred))