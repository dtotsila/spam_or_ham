import numpy as np
import socket
import threading
from collections import Counter
import pandas as pd
from utils import *
import pickle
import torch.nn as nn
import torch


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


# read columns template pandas series representing a tf idf vector for every message to be classified
template = pd.read_csv('data/test.csv', nrows=1)


# read idf needed for the tfidf vector representation of every new message for every term of the message
# that is in the corpus vocabulary tf * idf we used this method since we wanted to calculate the tfidf
# vector even for messages that have not been already represented
idf = pd.read_csv("data/idf.csv")

# load extra trees trained model
extra_trees = pickle.load(open('models/etc.pkl', 'rb'))
# load extra trees trained model
random_forest = pickle.load(open('models/rf.pkl', 'rb'))

# load trained neural net
model = BinaryClassification(len(template.iloc[0])-2)
model.load_state_dict(torch.load('models/nn.pt'))
model.eval()


PORT = 5000

# An IPv4 address is obtained
# for the server.
SERVER = socket.gethostbyname(socket.gethostname())

# Address is stored as a tuple
ADDRESS = (SERVER, PORT)

FORMAT = "utf-8"

# Lists that will contains
# all the clients connected to
# the server and their names.
clients, names = [], []

# Create a new socket for
# the server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# bind the address of the
# server to the socket
server.bind(ADDRESS)

# function to start the connection


def startChat():

    print("server is working on " + SERVER)

    # listening for connections
    server.listen()

    while True:

        # accept connections and returns
        # a new connection to the client
        #  and  the address bound to it
        conn, addr = server.accept()
        conn.send("NAME".encode(FORMAT))

        # 1024 represents the max amount
        # of data that can be received (bytes)
        name = conn.recv(1024).decode(FORMAT)

        # append the name and client
        # to the respective list
        names.append(name)
        clients.append(conn)
        print(f"Name is :{name}")
        # broadcast message
        broadcastMessage(f"{name} has joined the chat!".encode(FORMAT))
        conn.send('Connection successful!'.encode(FORMAT))

        # Start the handling thread
        thread = threading.Thread(target=handle, args=(conn, addr))
        thread.start()

        # no. of clients connected
        # to the server
        print(f"active connections {threading.activeCount()-1}")

# method to handle the
# incoming messages


def handle(conn, addr):

    print(f"new connection {addr}")
    connected = True

    while connected:
        # receive message
        message = conn.recv(1024)
        print(message)
        # broadcast message
        broadcastMessage(str.encode(is_spam(message)))

    # close the connection
    conn.close()

# method for broadcasting
# messages to the each clients


def broadcastMessage(message):
    for client in clients:
        client.send(message)


def is_spam(message):
    # read 3 of the best models and pass the text trough them
    # if at least one method cosiders that the message is spam instead of the message
    # an error is broadcasted

    # byte to str
    message = message.decode()

    # exctract user from message
    user = ""
    for i in range(0, len(message)):
        user += str(message[i])
        if message[i] == ":":
            break

    # isolate message withot the user:
    new_mes = message[i+1:]

    # preprocess message
    processed = message_processing(new_mes)

    # calculate parameters
    length = len(new_mes)
    has_num = has_numbers_text(new_mes)
    has_cur = has_currency_text(new_mes)

    # count term frequency for every word of the message
    tf = Counter(processed)

    # calculate tf-idf weight using corpus idf
    for col in template.columns:
        template[col].values[:] = 0.0

    # update template vector (zeros initialized)
    template.has_num.replace(0, has_num)
    template.has_money.replace(0, has_cur)
    template.length.replace(0, length)

    # convert message to tf-df vector
    # load random forest
    for key in tf:
        try:
            # operation to raise exception if word is not in corpus vocabulary
            a = (template[key])
            # update coresponding tf-idf value
            template[key] = tf.get(key) * idf.key.iloc[[0]].values
        except:
            print("word not in corpus vocabulary")

    # sum of the models output (range 0 to 3 )
    sum = 0
    # extract vector to be inserted in models
    input = template.iloc[0][2:].copy().to_numpy(
    ).reshape(-1, len(template.iloc[0][2:]))
    # extra trees output
    sum += extra_trees.predict(input)
    # random forest output
    sum += random_forest.predict(input)
    # neural net output
    with torch.no_grad():
        sum += torch.round(model(torch.Tensor(input.astype(float)))).item()
    # if no more than 1 model classified the message as spam return it as is
    if (sum < 2):
        return (message)
    # if at least two model marked the message as spam
    else:
        # Print in console that a spam message was found by the user
        print("SPAM Message found from : ", user)
        # return warning to be displayed so that all users know that a user tried to scam them
        return (str(user)+" tried to send a spam message")


# call the method to
# begin the communication
startChat()
