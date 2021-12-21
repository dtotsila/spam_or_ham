from pandas.core.arrays.boolean import coerce_to_array
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
from utils import LogisticRegression, CustomDataset

# Read Data
df = pd.read_pickle('/home/dionis/Workspaces/spam_or_ham/data/tfidf.pkl')
df.category = pd.factorize(df.category)[0]
print(df)



X_train, X_test, y_train, y_test = train_test_split( df.iloc[:, 1:].to_numpy(), df["category"].to_numpy(), test_size=0.33)
X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
y_train, y_test = torch.Tensor(y_train),torch.Tensor(y_test)

train_dataset = CustomDataset(X_train, y_train)
dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)



# Hyper Parameters
epochs = 3000

# input_dim = tfidf vector length
input_dim = X_test.shape[1]

# 1 binary output (binary classification)
output_dim = 1 
learning_rate = 1e-4

model = LogisticRegression(input_dim,output_dim)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

losses = []
losses_test = []
Iterations = []
batches_per_epoch = X_train.shape[0]//512
iter = 0

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
        # print(output.shape, outputs.view(dim, -1).shape)
        # + 0.005 * (1-w_all).square().sum().sum()
        loss = criterion( output, outputs.view(-1, output.shape[1]))
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss.item())
    # scheduler.step()
    train_mean_loss = running_loss/batches_per_epoch
    # print(i)
    print("Epoch: ", epoch+1, "Loss: ", train_mean_loss)
print('Finished Training')


model.eval()
correct = 0
total = 0
for i in range(X_test.shape[0]):
    output =torch.round(model(X_test[i]))
    if(output == y_test[i]):
        correct +=1
    total += 1

print(correct, total)
print(correct/total * 100)