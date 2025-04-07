"""
This script evaluates the trained model for hateful meme detection.

The script includes the following functionalities:
1. Loads the test features and labels and prepares the data loader. Note: test features and labels can be downloaded using the link: https://drive.google.com/drive/folders/10unJDls369wys_UYRZtq1C7-hOAhn1bq?usp=sharing.
2. Defines the classification head and loads the trained model.
3. Evaluates the performance in terms of test accuracy.


Usage:
    python3 eval.py --weights_path /path/to/harmc_clip.pth
"""
#--------------------------------------------------
# Imports
#-------------------------------------------------- 
import torch
import torch.nn as nn
import argparse

# set the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# argument parser for path to trained model
parser = argparse.ArgumentParser(description='Evaluation of trained model for hateful meme detection.')
parser.add_argument('--weights_path', type=str, default='', help='path to trained model')

args = parser.parse_args()


#--------------------------------------------------
# Test Dataset
#-------------------------------------------------- 
# load test features and labels
test_features = torch.load('test_features.pt')
test_labels = torch.load('test_labels.pt')

# data_loader
test_dataset = torch.utils.data.TensorDataset(test_features,test_labels)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)


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
        x = torch.relu(self.fc2(x)) 
        x = self.fc3(x)              
        return x
        
net = FCNET(input_dim, output_dim)   
net = net.to(device)

# load the trained model
net.load_state_dict(torch.load(args.weights_path))


#---------------------------------------------
# Test the Network on the Test Data
#---------------------------------------------
def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test accuracy: {:.3f}'.format(100. * correct / len(testloader.dataset)))
        
 
test()

