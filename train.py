"""
This script trains and evaluates a classification head for hateful meme detection.

The script includes the following functionalities:
1. Loads the training and test features and labels as well as the hard samples and prepares the data loaders. Note: training and test features and labels can be downloaded using the link: https://drive.google.com/drive/folders/10unJDls369wys_UYRZtq1C7-hOAhn1bq?usp=sharing.
2. Defines the classification head.
3. Trains the classification head using the proposed hard-mining approach and evaluates the performance in terms of test accuracy after each epoch.


Usage:
    python3 train.py --weight 0.05 --k 1
"""
#--------------------------------------------------
# Imports
#-------------------------------------------------- 
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

# set the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"


#--------------------------------------------------
# Hyperparameters
#-------------------------------------------------- 
# argument parser for the auxiliary weight and the number of nearest embeddings
parser = argparse.ArgumentParser(description='Training a classification head for hateful meme detection.')
parser.add_argument('--weight', type=float, default=0.05, help='auxiliary weight')
parser.add_argument('--k', type=int, default=1, help='n. of nearest embeddings')

args = parser.parse_args()

# Hyperparameters
aux_weight = args.weight
knn = args.k
learning_rate = 0.001
N_of_epochs = 500
batch_size = 64


#--------------------------------------------------
# Dataset
#-------------------------------------------------- 
# loads
train_features = torch.load('train_features.pt')
train_labels = torch.load('train_labels.pt')
test_features = torch.load('test_features.pt')
test_labels = torch.load('test_labels.pt')
hard_samples = torch.load('hard_samples.pt')

# data_loaders
train_dataset = torch.utils.data.TensorDataset(train_features,train_labels,hard_samples)
test_dataset = torch.utils.data.TensorDataset(test_features,test_labels)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# euclidean distance
def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))

# finding nearest embeddings
def find_nearest_embeddings(embeddings, labels, hard_indicators, k=knn):
    nearest_same_class = [None] * embeddings.size(0)
    nearest_opposite_class = [None] * embeddings.size(0)
    
    for i in range(embeddings.size(0)):
        if hard_indicators[i] == 1:
            same_class = (labels == labels[i]) & (hard_indicators == 0)
            opposite_class = (labels != labels[i])
            
            if same_class.sum() > 0:
                same_class_distances = torch.cdist(embeddings[i].unsqueeze(0), embeddings[same_class], p=2).squeeze(0)
                nearest_indices = torch.argsort(same_class_distances)[:k]
                nearest_same_class[i] = torch.mean(embeddings[same_class][nearest_indices], dim=0)
            
            if opposite_class.sum() > 0:
                opposite_class_distances = torch.cdist(embeddings[i].unsqueeze(0), embeddings[opposite_class], p=2).squeeze(0)
                nearest_indices = torch.argsort(opposite_class_distances)[:k]
                nearest_opposite_class[i] = torch.mean(embeddings[opposite_class][nearest_indices], dim=0)
                
    return nearest_same_class, nearest_opposite_class


#--------------------------------------------------
# Total Loss
#-------------------------------------------------- 
def total_loss(outputs, embeddings, labels, hard_indicators):
    criterion = nn.CrossEntropyLoss(reduction='none')
    regular_loss = criterion(outputs, labels)
    
    nearest_same_class, nearest_opposite_class = find_nearest_embeddings(embeddings, labels, hard_indicators)
    
    aux_loss = torch.tensor(0.0).to(device)
    hard_sample_count = 0
    
    for i in range(embeddings.size(0)):
        if hard_indicators[i] == 1:
            hard_sample_count += 1
            if nearest_same_class[i] is not None:
                aux_loss += euclidean_distance(embeddings[i], nearest_same_class[i])
            if nearest_opposite_class[i] is not None:
                aux_loss += 1 - euclidean_distance(embeddings[i], nearest_opposite_class[i])
    
    if hard_sample_count > 0:
        aux_loss /= hard_sample_count
    

    total_loss = regular_loss.mean() + aux_weight * aux_loss
    
    return total_loss


#--------------------------------------------------
# Classification Head
#-------------------------------------------------- 
input_dim = 3072
output_dim = 2  

class FCNET(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCNET, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  
        self.fc2 = nn.Linear(512, 256)        
        self.fc3 = nn.Linear(256, output_dim)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))  
        emb_ = torch.relu(self.fc2(x)) 
        out_ = self.fc3(emb_)              
        return out_, emb_
        
net = FCNET(input_dim, output_dim) 
net = net.to(device)


# optimizer
optimizer = optim.SGD(net.parameters(), lr= learning_rate, momentum=0.9)


#---------------------------------------------
#  Train the network
#---------------------------------------------
def train(epoch):
    net.train()

    for batch_idx, (inputs, labels, hards) in enumerate(trainloader, 0):
        inputs, labels, hards = inputs.to(device), labels.to(device), hards.to(device)
        optimizer.zero_grad()
        outputs,embeddings = net(inputs)
        loss = total_loss(outputs, embeddings, labels, hards)
        loss.backward()
        optimizer.step()


#---------------------------------------------
# Test the network on the test data
#---------------------------------------------
def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs,embeddings = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Epoch {}, Test accuracy: {:.3f}'.format(epoch, 100. * correct / len(testloader.dataset)))
        

# training and testing the network
for epoch in range(1, N_of_epochs + 1):
    train(epoch)
    test()  

