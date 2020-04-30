#!/usr/bin/env python
# coding: utf-8

# In[160]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
#import matplotlib.pyplot as plt

def extract_data(x_data_filepath, y_data_filepath):
    X = np.load(x_data_filepath)
    y = np.load(y_data_filepath)
    return X, y


def data_visualization(images,labels):
    """
    Visualize 6 pictures per class using your prefered visualization library (matplotlib, etc)

    Args:
        images: training images in shape (num_images,3,image_H,image_W)
        labels: training labels in shape (num_images,)
    """
    
    class0 = []
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    for i in range(len(labels)):
        if labels[i]==0 and len(class0)<=5:
            class0.append(int(i))
        if labels[i]==1 and len(class1)<=5:
            class1.append(int(i))
        if labels[i]==2 and len(class2)<=5:
            class2.append(int(i))  
        if labels[i]==3 and len(class3)<=5:
            class3.append(int(i))  
        if labels[i]==4 and len(class4)<=5:
            class4.append(int(i))  
            
    img_id = [class0,class1, class2, class3, class4]
    img_id
    
    for i in range(5):
        for j in img_id[i]:
             
#            plt.figure()
#            plt.imshow(images[j][:,:,:].T.swapaxes(0,1))
             pass

# In[167]:


############################################################
# Extracting and loading data
############################################################
class Dataset(Dataset):
    def __init__(self, X, y):
        self.len = len(X)           
        if torch.cuda.is_available():
            self.x_data = torch.from_numpy(X).float().cuda()
            self.y_data = torch.from_numpy(y).long().cuda()
        else:
            self.x_data = torch.from_numpy(X).float()
            self.y_data = torch.from_numpy(y).long()
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# In[312]:


def create_validation(x_train,y_train):
    """
    Randomly choose 20 percent of the training data as validation data.

    Args:
        x_train: training images in shape (num_images,3,image_H,image_W)
        y_train: training labels in shape (num_images,)
    Returns:
        new_x_train: training images in shape (0.8*num_images,3,image_H,image_W)
        new_y_train: training labels in shape (0.8*num_images,)
        x_val: validation images in shape (0.2*num_images,3,image_H,image_W)
        y_val: validation labels in shape (0.2*num_images,)
    """

    indices = np.arange(x_train.shape[0])
    new_train_indices = indices[0:int(np.floor(0.8*len(indices)))]
    val_indices = indices[int(np.floor(0.8*len(indices))):]
    new_x_train = x_train[new_train_indices,:,:,:]
    new_y_train = y_train[new_train_indices]
    x_val = x_train[val_indices,:,:,:]
    y_val = y_train[val_indices]
    
    return new_x_train,new_y_train,x_val,y_val


# In[169]:


############################################################
# Feed Forward Neural Network
############################################################
class FeedForwardNN(nn.Module):
    """ 
        (1) Use self.fc1 as the variable name for your first fully connected layer
        (2) Use self.fc2 as the variable name for your second fully connected layer
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64*85*3, 2000)
        self.fc2 = nn.Linear(2000,5)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out

    """ 
        Please do not change the functions below. 
        They will be used to test the correctness of your implementation 
    """
    def get_fc1_params(self):
        return self.fc1.__repr__()
    
    def get_fc2_params(self):
        return self.fc2.__repr__()


# In[216]:


############################################################
# Convolutional Neural Network
############################################################
class ConvolutionalNN(nn.Module):
    """ 
        (1) Use self.conv1 as the variable name for your first convolutional layer
        (2) Use self.pool1 as the variable name for your first pooling layer
        (3) Use self.conv2 as the variable name for your second convolutional layer
        (4) Use self.pool2 as the variable name for you second pooling layer  
        (5) Use self.fc1 as the variable name for your first fully connected laye
        (6) Use self.fc2 as the variable name for your second fully connected layer
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 0)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(16,32, kernel_size = 3, stride = 1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(32*14*19,200)
        self.fc2 = nn.Linear(200,5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        
        return out
    
    def get_conv1_params(self):
        return self.conv1.__repr__()
    
    def get_pool1_params(self):
        return self.pool1.__repr__()

    def get_conv2_params(self):
        return self.conv2.__repr__()
      
    def get_pool2_params(self):
        return self.pool2.__repr__()
      
    def get_fc1_params(self):
        return self.fc1.__repr__()
    
    def get_fc2_params(self):
        return self.fc2.__repr__()

def normalize_image(image):
    """
    Normalize each input image

    Args:
        image: the input image in shape (3,image_H,image_W)
    Returns:
        norimg: the normalized image in the same shape as the input
    """


    r_mean = np.mean(image[0])
    g_mean = np.mean(image[1])
    b_mean = np.mean(image[2])
    
    r_std = np.std(image[0])
    g_std = np.std(image[1])
    b_std = np.std(image[2])
    
    image[0] = (image[0] - r_mean)/r_std
    image[1] = (image[1] - g_mean)/g_std
    image[2] = (image[2] - b_mean)/b_std
    
    norimg = image
    
    return norimg


# In[295]:


############################################################
# Optimized Neural Network
############################################################
class OptimizedNN(nn.Module):
    
    def __init__(self):
        super(OptimizedNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3,stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3,stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        # add an additional conv layer to further extract features
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3,stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(64*6*8, 200)
        # apply dropout to reduce overfitting
        self.dropout1 = nn.Dropout(p=0.55)
        self.fc2 = nn.Linear(200, 5)
        # apply another dropout
        self.dropout2 = nn.Dropout(p=0.2)
        
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0),-1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        out = self.fc2(x)
        return out


# In[243]:


def train_val_NN(neural_network, train_loader, validation_loader, loss_function, optimizer,num_epochs):
    """
    Runs experiment on the model neural network given a train loader, loss function and optimizer and find validation 
    accuracy for each epoch given the validation_loader.

    Args:
        neural_network (NN model that extends torch.nn.Module): For example, it should take an instance of either
                                                                FeedForwardNN or ConvolutionalNN,
        train_loader (DataLoader),
        validation_loader (DataLoader),
        loss_function (torch.nn.CrossEntropyLoss),
        optimizer (optim.SGD)
        num_epochs (number of iterations)
    Returns:
        tuple: First position, training accuracies of each epoch formatted in an array of shape (num_epochs,1).
               Second position, training loss of each epoch formatted in an array of shape (num_epochs,1).
               third position, validation accuracy of each epoch formatted in an array of shape (num_epochs,1).
               
    """
    train_accuracy = np.zeros((num_epochs,1))
    val_accuracy = np.zeros((num_epochs,1))
    train_loss = np.zeros((num_epochs,1))
    
    i = 0
    for epoch in range(num_epochs):
        
        total_train_loss = 0
        # training
        correct = 0
        for inputs, labels in train_loader:
            
            
            optimizer.zero_grad()
            
            outputs = neural_network(inputs)
            
            loss = loss_function(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
            
            total_train_loss += loss.item()
             
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        
        train_accuracy[i] = correct/len(train_loader.dataset)   #avg accuracy over batches for this epoch
        train_loss[i] = total_train_loss   
        
        # validation
        correct = 0
        for inputs, labels in validation_loader:
            
         
            #Forward pass
            val_outputs = neural_network(inputs)
            
            _, predicted = torch.max(val_outputs.data, 1)
            correct += (predicted == labels).sum().item()
        
        val_accuracy[i] = correct/len(validation_loader.dataset)   #avg accuracy over batches for this epoch
        
        i+=1
    
    return (train_accuracy, train_loss, val_accuracy)


# In[310]:


def test_NN(neural_network, test_loader):
  
    """
    Runs experiment on the model neural network given a test loader, loss function and optimizer.

    Args:
        neural_network (NN model that extends torch.nn.Module): For example, it should take an instance of either
                                                                FeedForwardNN or ConvolutionalNN,
        test_loader (DataLoader), (make sure the loader is not shuffled)
    Returns:
    """
    
    Preds = torch.LongTensor()
    for i, data in enumerate(test_loader):
        outputs = neural_network(data[0]) 
        y_pred = outputs.data.max(1, keepdim=True)[1]
        Preds = torch.cat((Preds, y_pred), dim=0)
        
    return Preds


## In[262]:
#
#
## Run Baseline FeedForward
#images, labels = extract_data('data/images_train.npy', 'data/labels_train.npy')
#x_train, y_train, x_val, y_val = create_validation(images, labels)
    
#data = Dataset(x_train,y_train)
#x_train = data.x_data
#y_train = data.y_data    

#data = Dataset(x_val, y_val)
#x_val = data.x_data
#y_val = data.y_data 
    
#train_set = torch.utils.data.TensorDataset(x_train,y_train)
#val_set = torch.utils.data.TensorDataset(x_val,y_val)
#train_loader = DataLoader(train_set, batch_size = int(2601/40))
#val_loader = DataLoader(val_set, batch_size = int(2601/40))
#
#num_epochs = 40
#FF = FeedForwardNN()
#
#loss_function = nn.CrossEntropyLoss()
#optimizer = optim.Adagrad(FF.parameters(), lr=0.001)
#
#(ff_train_accuracy,ff_train_loss,ff_val_accuracy) = train_val_NN(FF, train_loader, val_loader, loss_function, optimizer,num_epochs)
#print(ff_val_accuracy)


# In[263]:



## plotting
#
#epochs = range(1,num_epochs+1)
#f, ax = plt.subplots(1)
#ax.plot(epochs, ff_train_loss, 'g')
#ax.set_ylim(bottom=0)
#plt.title('feedforward nn \n')
#plt.xlabel('epochs')
#plt.legend(['train_loss'])
#plt.show(f)
#
#epochs = range(1,num_epochs+1)
#f, ax = plt.subplots(1)
#ax.plot(epochs, ff_train_accuracy)
#ax.set_ylim(bottom=0)
#plt.title('feedforward nn \n')
#plt.xlabel('epochs')
#plt.legend(['train_accuracy'])
#plt.show(f)
#
#
## In[259]:
#
#
## Run Baseline CNN
#images, labels = extract_data('data/images_train.npy', 'data/labels_train.npy')
#x_train, y_train, x_val, y_val = create_validation(images, labels)
    
#data = Dataset(x_train,y_train)
#x_train = data.x_data
#y_train = data.y_data    

#data = Dataset(x_val, y_val)
#x_val = data.x_data
#y_val = data.y_data 
    
#train_set = torch.utils.data.TensorDataset(x_train,y_train)
#val_set = torch.utils.data.TensorDataset(x_val,y_val)
#train_loader = DataLoader(train_set, batch_size = int(2601/40))
#val_loader = DataLoader(val_set, batch_size = int(2601/40))
#
#num_epochs = 40
#CNN = ConvolutionalNN()
#
#loss_function = nn.CrossEntropyLoss()
#optimizer = optim.Adagrad(CNN.parameters(), lr=0.001)
#
#(cnn_train_accuracy,cnn_train_loss,cnn_val_accuracy) = train_val_NN(CNN, train_loader, val_loader, loss_function, optimizer,num_epochs)
#print(cnn_val_accuracy)
#
#
## In[264]:
#
#
#
#
## plotting
#
#epochs = range(1,num_epochs+1)
#f, ax = plt.subplots(1)
#ax.plot(epochs, cnn_train_loss, 'g')
#ax.set_ylim(bottom=0)
#plt.title('Baseline CNN \n')
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.legend(['loss'])
#plt.show(f)
#
#f, ax = plt.subplots(1)
#ax.plot(epochs, cnn_train_accuracy, 'b')
#ax.set_ylim(bottom=0)
#plt.title('Baseline CNN  \n')
#plt.xlabel('epochs')
#plt.ylabel('accuracy')
#plt.legend(['train_accuracy'])
#plt.show(f)
#
#
## In[251]:
#
#
## Run Baseline CNN on Normilized Images
## Run Baseline FeedForward
#images, labels = extract_data('data/images_train.npy', 'data/labels_train.npy')
#i=0
##normalize images
#for img in images:
#    norm_img = normalize_image(img)
#    images[i] = norm_img
#    i+=1
#    
#x_train, y_train, x_val, y_val = create_validation(images, labels)

#data = Dataset(x_train,y_train)
#x_train = data.x_data
#y_train = data.y_data    

#data = Dataset(x_val, y_val)
#x_val = data.x_data
#y_val = data.y_data   
    
#train_set = torch.utils.data.TensorDataset(x_train,y_train)
#val_set = torch.utils.data.TensorDataset(x_val,y_val)
#train_loader = DataLoader(train_set, batch_size = int(2601/40))
#val_loader = DataLoader(val_set, batch_size = int(2601/40))
#
#num_epochs = 40
#CNN_norm = ConvolutionalNN()
#
#loss_function = nn.CrossEntropyLoss()
#optimizer = optim.Adagrad(CNN_norm.parameters(), lr=0.001)
#
#(cnn_norm_train_accuracy,cnn_norm_train_loss,cnn_norm_val_accuracy) = train_val_NN(CNN_norm, train_loader, val_loader, loss_function, optimizer,num_epochs)
#
#
## In[267]:
#
#
#
## plotting
#
#
#f, ax = plt.subplots(1)
#ax.plot(epochs, cnn_norm_train_accuracy, 'b')
#ax.plot(epochs, cnn_train_accuracy, 'g')
#ax.set_ylim(bottom=0)
#plt.title('CNN with normalized images  \n training accuracy')
#plt.xlabel('epochs')
#plt.ylabel('accuracy')
#plt.legend(['normalized','raw'])
#plt.show(f)
#
#f, ax = plt.subplots(1)
#ax.plot(epochs, cnn_norm_val_accuracy, 'b')
#ax.plot(epochs, cnn_val_accuracy, 'g')
#ax.set_ylim(bottom=0)
#plt.title('CNN with normalized images  \n validation accuracy')
#plt.xlabel('epochs')
#plt.ylabel('accuracy')
#plt.legend(['normalized','raw'])
#plt.show(f)
#
#epochs = range(1,num_epochs+1)
#f, ax = plt.subplots(1)
#ax.plot(epochs, cnn_norm_train_loss, 'r')
#ax.plot(epochs, cnn_train_loss, 'y')
#ax.set_ylim(bottom=0)
#plt.title('CNN with normalized images \n training loss')
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.legend(['normalized','raw'])
#plt.show(f)
#
#
## In[313]:
#
#
## Choose from one of the above models and improve its performance
#images, labels = extract_data('data/images_train.npy', 'data/labels_train.npy')
#i=0
##normalize images
#for img in images:
#    norm_img = normalize_image(img)
#    images[i] = norm_img
#    i+=1
#    
#x_train, y_train, x_val, y_val = create_validation(images, labels)
    
#data = Dataset(x_train,y_train)
#x_train = data.x_data
#y_train = data.y_data    

#data = Dataset(x_val, y_val)
#x_val = data.x_data
#y_val = data.y_data  
    
#train_set = torch.utils.data.TensorDataset(x_train,y_train)
#val_set = torch.utils.data.TensorDataset(x_val,y_val)
#train_loader = DataLoader(train_set, batch_size = 32)
#val_loader = DataLoader(val_set, batch_size = 32)
#
#
#num_epochs = 40
#opt_cnn = OptimizedNN()
#
#loss_function = nn.CrossEntropyLoss()
#optimizer = optim.Adam(opt_cnn.parameters(), lr=0.001)
#
#(opt_cnn_train_accuracy,opt_cnn_train_loss,opt_cnn_val_accuracy) = train_val_NN(opt_cnn, train_loader, val_loader, loss_function, optimizer,num_epochs)
#
#
## In[314]:
#
#
## plotting
#
#epochs = range(1,num_epochs+1)
#f, ax = plt.subplots(1)
#ax.plot(epochs, opt_cnn_train_loss, 'g')
#ax.set_ylim(bottom=0)
#plt.title('Optimized CNN \n loss')
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.legend(['loss'])
#plt.show(f)
#
#f, ax = plt.subplots(1)
#ax.plot(epochs, opt_cnn_train_accuracy, 'b')
#ax.set_ylim(bottom=0)
#plt.title('Optimized CNN  \n training accuracy')
#plt.xlabel('epochs')
#plt.ylabel('accuracy')
#plt.legend(['train_accuracy'])
#plt.show(f)
#
#f, ax = plt.subplots(1)
#ax.plot(epochs, opt_cnn_val_accuracy, 'r')
#ax.set_ylim(bottom=0)
#plt.title('Optimized CNN  \n Validation accuracy')
#plt.xlabel('epochs')
#plt.ylabel('accuracy')
#plt.legend(['validation_accuracy'])
#plt.show(f)
#

