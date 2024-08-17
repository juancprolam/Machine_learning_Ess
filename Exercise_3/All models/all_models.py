### All models

## Plot options
from cycler import cycler
from matplotlib import rcParams
rcParams['axes.grid'] = True
rcParams['grid.linestyle'] = '--'
rcParams['grid.alpha'] = 0.5
rcParams['axes.labelsize'] = 20
rcParams['axes.prop_cycle'] = cycler('color', ['#C61A27', '#3891A6', \
                                               '#F79D65', '#FDE74C'])
rcParams['axes.titlesize'] = 22
rcParams['figure.figsize'] = (12, 7)
rcParams['figure.titlesize'] = 26
rcParams['font.size'] = 16
rcParams['image.cmap'] = 'magma'
rcParams['lines.markeredgewidth'] = 2
rcParams['lines.markerfacecolor'] = 'white'
rcParams['markers.fillstyle'] = 'none'

## Packages
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.nn.functional import conv2d, max_pool2d, cross_entropy

# Tools
import time
from tqdm import tqdm
import os
import glob

## Data
batch_size = 100

# transform images into normalized tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_dataset = datasets.MNIST(
    "./",
    download=True,
    train=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    "./",
    download=True,
    train=False,
    transform=transform,
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

## Task 1: Intro (ReLU model)
def init_weights(shape):
    # Kaiming He initialization (a good initialization is important)
    # https://arxiv.org/abs/1502.01852
    std = np.sqrt(2. / shape[0])
    w = torch.randn(size=shape) * std
    w.requires_grad = True
    return w


def rectify(x):
    # Rectified Linear Unit (ReLU)
    return torch.max(torch.zeros_like(x), x)


class RMSprop(optim.Optimizer):
    """
    This is a reduced version of the PyTorch internal RMSprop optimizer
    It serves here as an example
    """
    def __init__(self, params, lr=1e-3, alpha=0.5, eps=1e-8):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(RMSprop, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # state initialization
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                # update running averages
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                avg = square_avg.sqrt().add_(group['eps'])

                # gradient update
                p.data.addcdiv_(grad, avg, value=-group['lr'])


# define the neural network
def model(x, w_h, w_h2, w_o):
    h = rectify(x @ w_h)
    h2 = rectify(h @ w_h2)
    pre_softmax = h2 @ w_o
    return pre_softmax


# initialize weights

# input shape is (B, 784)
w_h = init_weights((784, 625))
# hidden layer with 625 neurons
w_h2 = init_weights((625, 625))
# hidden layer with 625 neurons
w_o = init_weights((625, 10))
# output shape is (B, 10)

optimizer = RMSprop(params=[w_h, w_h2, w_o])

def calculate_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    return correct / target.size(0)

def find_latest_checkpoint(checkpoint_dir, 
                           root = "checkpoint_intro_epoch_*.pth"):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, root))
    
    if not checkpoint_files:
        return None
    
    checkpoint_files.sort(key = os.path.getmtime)
    return checkpoint_files[-1]

# Remember to implement a checkpoint!!
# Especially since my laptop takes longer than 1 hour to run this
def load_checkpoint(filepath):
    # Saved checkpoint in case of crash
    checkpoint = torch.load(filepath)
    start_epoch = checkpoint['epoch']
    w_h, w_h2, w_o = checkpoint['model_state_dict']
    optimizer = RMSprop(params = [w_h, w_h2, w_o])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_loss = checkpoint['train_loss']
    test_loss = checkpoint['test_loss']
    train_accuracy = checkpoint['train_accuracy']
    test_accuracy = checkpoint['test_accuracy']
    return (start_epoch, w_h, w_h2, w_o, optimizer, 
            train_loss, test_loss, train_accuracy, test_accuracy)


# Can you load from checkpoint?
checkpoint_dir = '.'
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    (start_epoch, w_h, w_h2, w_o, optimizer, 
     train_loss, test_loss, train_accuracy, test_accuracy) \
        = load_checkpoint(latest_checkpoint)
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    print("Starting from scratch...")


n_epochs = 100


# put this into a training loop over 100 epochs
for epoch in tqdm(range(start_epoch + 1, n_epochs + 1)):
    train_loss_this_epoch = []
    train_correct = 0
    
    for idx, batch in enumerate(train_dataloader):
        x, y = batch

        # our model requires flattened input
        x = x.reshape(batch_size, 784)
        # feed input through model
        noise_py_x = model(x, w_h, w_h2, w_o)

        # reset the gradient
        optimizer.zero_grad()

        # the cross-entropy loss function already contains the softmax
        loss = cross_entropy(noise_py_x, y, reduction="mean")

        train_loss_this_epoch.append(float(loss))

        # compute the gradient
        loss.backward()
        # update weights
        optimizer.step()
        
        # Calculate accuracy
        train_correct += (torch.argmax(noise_py_x, dim=1) == y).sum().item()

    train_loss.append(np.mean(train_loss_this_epoch))
    train_accuracy.append(train_correct / len(train_dataset))

    # test periodically
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}")
        print(f"Mean Train Loss: {train_loss[-1]:.2e}")
        print(f"Train Accuracy: {train_accuracy[-1]:.2%}")
        test_loss_this_epoch = []
        test_correct = 0

        # no need to compute gradients for validation
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                x, y = batch
                x = x.reshape(batch_size, 784)
                noise_py_x = model(x, w_h, w_h2, w_o)

                loss = cross_entropy(noise_py_x, y, reduction="mean")
                test_loss_this_epoch.append(float(loss))
                # Calculate accuracy
                test_correct += (torch.argmax(noise_py_x, dim=1) == y).sum().item()


        test_loss.append(np.mean(test_loss_this_epoch))
        test_accuracy.append(test_correct / len(test_dataset))

        print(f"Mean Test Loss:  {test_loss[-1]:.2e}")
        print(f"Test Accuracy: {test_accuracy[-1]:.2%}")
        
        # Save model checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': [w_h, w_h2, w_o],
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }, f'checkpoint_dropout_epoch_{epoch}.pth')

        
# Loss plot
intro_train_loss = train_loss
intro_test_loss = test_loss
plt.figure()
plt.plot(np.arange(n_epochs), 
         train_loss, 
         label = "Train")
plt.plot(np.arange(1, n_epochs + 1, 10), 
         test_loss, 
         label = "Test")
plt.title("Train and Test Loss over Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Loss_intro.png')

# Accuracy plot
intro_train_accuracy = train_accuracy
intro_test_accuracy = test_accuracy
plt.figure()
plt.plot(np.arange(n_epochs), 
         train_accuracy, 
         label = "Train")
plt.plot(np.arange(1, n_epochs + 1, 10), 
         test_accuracy, 
         label = "Test")
plt.title("Train and Test Accuracy over Training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Accuracy_intro.png')

## Task 2: Dropout model
def init_weights(shape):
    # Kaiming He initialization (a good initialization is important)
    # https://arxiv.org/abs/1502.01852
    std = np.sqrt(2. / shape[0])
    w = torch.randn(size = shape) * std
    w.requires_grad = True
    return w


def rectify(x):
    # Rectified Linear Unit (ReLU)
    return torch.max(torch.zeros_like(x), x)


# Return weighted dropout
# Apparently, torch doesn't like playing with other kids. Use torch.rand
# instead of python's other methods, such as numpy random or scipy binom.
def dropout(X, p_drop = 0.3):
    if p_drop < 0 or p_drop >= 1:
        return X 
    
    mask = (torch.rand(X.shape) < 1 - p_drop).float()
    X = X * mask / (1 - p_drop)
    return X


class RMSprop(optim.Optimizer):
    """
    This is a reduced version of the PyTorch internal RMSprop optimizer
    It serves here as an example
    """
    def __init__(self, params, lr = 1e-4, alpha = 0.5, eps = 1e-8):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super(RMSprop, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # state initialization
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                # update running averages
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                avg = square_avg.sqrt().add_(group['eps'])

                # gradient update
                p.data.addcdiv_(grad, avg, value=-group['lr'])
                
# define the neural network
def model(x, w_h, w_h2, w_o):
    h = rectify(x @ w_h)
    h2 = rectify(h @ w_h2)
    pre_softmax = h2 @ w_o
    return pre_softmax


def dropout_model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(X @ w_h)
    
    h = dropout(h, p_drop_hidden)
    h2 = rectify(h @ w_h2)
    
    h2 = dropout(h2, p_drop_hidden)
    pre_softmax = h2 @ w_o
    return pre_softmax


def calculate_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    return correct / target.size(0)

# Remember to implement a checkpoint!!
# Especially since my laptop takes longer than 1 hour to run this
def load_checkpoint(filepath):
    # Saved checkpoint in case of crash
    checkpoint = torch.load(filepath)
    start_epoch = checkpoint['epoch']
    w_h, w_h2, w_o = checkpoint['model_state_dict']
    optimizer = RMSprop(params = [w_h, w_h2, w_o])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_loss = checkpoint['train_loss']
    test_loss = checkpoint['test_loss']
    train_accuracy = checkpoint['train_accuracy']
    test_accuracy = checkpoint['test_accuracy']
    return (start_epoch, w_h, w_h2, w_o, optimizer, 
            train_loss, test_loss, train_accuracy, test_accuracy)

def find_latest_checkpoint(checkpoint_dir, 
                           root = "checkpoint_dropout_epoch_*.pth"):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, root))
    
    if not checkpoint_files:
        return None
    
    checkpoint_files.sort(key = os.path.getmtime)
    return checkpoint_files[-1]


# initialize weights

# input shape is (B, 784)
w_h = init_weights((784, 625))
# hidden layer with 625 neurons
w_h2 = init_weights((625, 625))
# hidden layer with 625 neurons
w_o = init_weights((625, 10))
# output shape is (B, 10)

# dropout probabilities
p_drop_input = 0.5
p_drop_hidden = 0.5

optimizer = RMSprop(params=[w_h, w_h2, w_o])
            
# Can you load from checkpoint?
checkpoint_dir = '.'
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    (start_epoch, w_h, w_h2, w_o, optimizer, 
     train_loss, test_loss, train_accuracy, test_accuracy) \
        = load_checkpoint(latest_checkpoint)
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    print("Starting from scratch...")

n_epochs = 100

# Mark this as the start_time:
start_time = time.time()

# put this into a training loop over 100 epochs
for epoch in tqdm(range(start_epoch + 1, n_epochs + 1)):
    train_loss_this_epoch = []
    train_correct = 0
    
    for idx, batch in enumerate(train_dataloader):
        x, y = batch
        # our model requires flattened input
        x = x.reshape(batch_size, 784)
        # feed input through model
        noise_py_x = dropout_model(x, w_h, w_h2, w_o,
                                   p_drop_input, p_drop_hidden)

        # reset the gradient
        optimizer.zero_grad()

        # the cross-entropy loss function already contains the softmax
        loss = cross_entropy(noise_py_x, y, reduction = "mean")
        train_loss_this_epoch.append(float(loss))

        # compute the gradient
        loss.backward()
        # update weights
        optimizer.step()
        
        # Calculate accuracy
        train_correct += (torch.argmax(noise_py_x, dim=1) == y).sum().item()

    train_loss.append(np.mean(train_loss_this_epoch))
    train_accuracy.append(train_correct / len(train_dataset))

    # test periodically
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}")
        print(f"Mean Train Loss: {train_loss[-1]:.2e}")
        print(f"Train Accuracy: {train_accuracy[-1]:.2%}")
        test_loss_this_epoch = []
        test_correct = 0

        # no need to compute gradients for validation
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                x, y = batch
                x = x.reshape(batch_size, 784)
                noise_py_x = model(x, w_h, w_h2, w_o)

                loss = cross_entropy(noise_py_x, y, reduction = "mean")
                test_loss_this_epoch.append(float(loss))
                # Calculate accuracy
                test_correct += (torch.argmax(noise_py_x, dim=1) == y).sum().item()

        test_loss.append(np.mean(test_loss_this_epoch))
        test_accuracy.append(test_correct / len(test_dataset))

        print(f"Mean Test Loss:  {test_loss[-1]:.2e}")
        print(f"Test Accuracy: {test_accuracy[-1]:.2%}")
        
        # Save model checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': [w_h, w_h2, w_o],
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }, f'checkpoint_dropout_epoch_{epoch}.pth')
        
    end_time = time.time()

# Loss plot
dropout_train_loss = train_loss
dropout_test_loss = test_loss
plt.figure()
plt.plot(np.arange(1, len(train_loss) + 1), 
         train_loss, 
         label = "Train")
plt.plot(np.arange(1, len(test_loss) * 10 + 1, 10), 
         test_loss, 
         label = "Test")
plt.title("Train and Test Loss over Training\nDropout model")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Loss_dropout.png')

# Accuracy plot
dropout_train_accuracy = train_accuracy
dropout_test_accuracy = test_accuracy
plt.figure()
plt.plot(np.arange(n_epochs), 
         train_accuracy, 
         label = "Train")
plt.plot(np.arange(1, n_epochs + 1, 10), 
         test_accuracy, 
         label = "Test")
plt.title("Train and Test Accuracy over Training\nDropout model")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Accuracy_dropout.png')

## Task 3: Parametric model
def init_weights(shape):
    # Kaiming He initialization (a good initialization is important)
    # https://arxiv.org/abs/1502.01852
    std = np.sqrt(2. / shape[0])
    w = torch.randn(size = shape) * std
    w.requires_grad = True
    return w

def rectify(x):
    # Rectified Linear Unit (ReLU)
    return torch.max(torch.zeros_like(x), x)


# If x > 0: return x, else: return a * x
# Use min, max functions for case discrimination
def PReLU(x, a):
    # Parametric ReLU
    return (torch.max(x, torch.zeros_like(x)) 
            + a * torch.min(x, torch.zeros_like(x)))

# Return weighted dropout
# Apparently, torch doesn't like playing with other kids. Use torch.rand
# instead of python's other methods, such as numpy random or scipy binom.
def dropout(X, p_drop = 0.3):
    if p_drop < 0 or p_drop >= 1:
        return X 
    
    mask = (torch.rand(X.shape) < 1 - p_drop).float()
    X = X * mask / (1 - p_drop)
    return X


class RMSprop(optim.Optimizer):
    """
    This is a reduced version of the PyTorch internal RMSprop optimizer
    It serves here as an example
    """
    def __init__(self, params, lr = 1e-4, alpha = 0.5, eps = 1e-8):
        defaults = dict(lr = lr, alpha = alpha, eps = eps)
        super(RMSprop, self).__init__(params, defaults)

    # This automatically calculates the gradient and updates for every
    # learnable parameter.
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # state initialization
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                # update running averages
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                avg = square_avg.sqrt().add_(group['eps'])

                # gradient update
                p.data.addcdiv_(grad, avg, value=-group['lr'])
                
# define the neural network
def model(x, w_h, w_h2, w_o):
    h = rectify(x @ w_h)
    h2 = rectify(h @ w_h2)
    pre_softmax = h2 @ w_o
    return pre_softmax


def dropout_model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(X @ w_h)
    
    h = dropout(h, p_drop_hidden)
    h2 = rectify(h @ w_h2)
    
    h2 = dropout(h2, p_drop_hidden)
    pre_softmax = h2 @ w_o
    return pre_softmax

# Same as ReLU model, but with PReLU
def parametric_model(X, w_h, w_h2, w_o, a):
    h = PReLU(X @ w_h, a)
    h2 = PReLU(h @ w_h2, a)
    pre_softmax = h2 @ w_o
    return pre_softmax


def calculate_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    return correct / target.size(0)

# Remember to implement a checkpoint!!
# Especially since my laptop takes longer than 1 hour to run this
def load_checkpoint(filepath):
    # Saved checkpoint in case of crash
    checkpoint = torch.load(filepath)
    start_epoch = checkpoint['epoch']
    w_h, w_h2, w_o, a = checkpoint['model_state_dict']
    optimizer = RMSprop(params = [w_h, w_h2, w_o, a])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_loss = checkpoint['train_loss']
    test_loss = checkpoint['test_loss']
    train_accuracy = checkpoint['train_accuracy']
    test_accuracy = checkpoint['test_accuracy']
    return (start_epoch, w_h, w_h2, w_o, optimizer, 
            train_loss, test_loss, train_accuracy, test_accuracy)

def find_latest_checkpoint(checkpoint_dir, 
                           root = "checkpoint_parametric_epoch_*.pth"):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, root))
    
    if not checkpoint_files:
        return None
    
    checkpoint_files.sort(key = os.path.getmtime)
    return checkpoint_files[-1]


# initialize weights

# input shape is (B, 784)
w_h = init_weights((784, 625))
# hidden layer with 625 neurons
w_h2 = init_weights((625, 625))
# hidden layer with 625 neurons
w_o = init_weights((625, 10))
# output shape is (B, 10)

# Initialize a as a 0x0 tensor with gradient requirement
a = torch.tensor(0.25, requires_grad = True)


# REMEMBER: Include "a" from PReLU as a learnable param
optimizer = RMSprop(params = [w_h, w_h2, w_o, a])
            
# Can you load from checkpoint?
checkpoint_dir = '.'
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    (start_epoch, w_h, w_h2, w_o, optimizer, 
     train_loss, test_loss, train_accuracy, test_accuracy) \
        = load_checkpoint(latest_checkpoint)
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    print("Starting from scratch...")


n_epochs = 100


# Mark this as the start_time:
start_time = time.time()

# put this into a training loop over 100 epochs
for epoch in tqdm(range(start_epoch + 1, n_epochs + 1)):
    train_loss_this_epoch = []
    train_correct = 0
    for idx, batch in enumerate(train_dataloader):
        x, y = batch
        # our model requires flattened input
        x = x.reshape(batch_size, 784)
        # feed input through model
        noise_py_x = parametric_model(x, w_h, w_h2, w_o, a)

        # reset the gradient
        optimizer.zero_grad()

        # the cross-entropy loss function already contains the softmax
        loss = cross_entropy(noise_py_x, y, reduction = "mean")

        train_loss_this_epoch.append(float(loss))

        # compute the gradient
        loss.backward()
        # update weights
        optimizer.step()
        
        # Calculate accuracy
        train_correct += (torch.argmax(noise_py_x, dim = 1) == y).sum().item()

    train_loss.append(np.mean(train_loss_this_epoch))
    train_accuracy.append(train_correct / len(train_dataset))

    # test periodically
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}")
        print(f"Mean Train Loss: {train_loss[-1]:.2e}")
        print(f"Train Accuracy: {train_accuracy[-1]:.2%}")
        test_loss_this_epoch = []
        test_correct = 0

        # no need to compute gradients for validation
        with torch.no_grad():
            for idx, batch in enumerate(test_dataloader):
                x, y = batch
                x = x.reshape(batch_size, 784)
                noise_py_x = parametric_model(x, w_h, w_h2, w_o, a)

                loss = cross_entropy(noise_py_x, y, reduction = "mean")
                test_loss_this_epoch.append(float(loss))
                
                # Calculate accuracy
                test_correct += (torch.argmax(noise_py_x, dim=1) == y).sum().item()

        test_loss.append(np.mean(test_loss_this_epoch))
        test_accuracy.append(test_correct / len(test_dataset))

        print(f"Mean Test Loss:  {test_loss[-1]:.2e}")
        print(f"Test Accuracy: {test_accuracy[-1]:.2%}")
        
        # Save model checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': [w_h, w_h2, w_o, a],
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }, f'checkpoint_parametric_epoch_{epoch}.pth')
        
    end_time = time.time()
    
    
# Loss plot
parametric_train_loss = train_loss
parametric_test_loss = test_loss
plt.figure()
plt.plot(np.arange(n_epochs), 
         train_loss, 
         label = "Train")
plt.plot(np.arange(1, n_epochs + 1, 10), 
         test_loss, 
         label = "Test")
plt.title("Train and Test Loss over Training\nParametric model")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Loss_parametric.png')

# Accuracy plot
parametric_train_accuracy = train_accuracy
parametric_test_accuracy = test_accuracy
plt.figure()
plt.plot(np.arange(n_epochs), 
         train_accuracy, 
         label = "Train")
plt.plot(np.arange(1, n_epochs + 1, 10), 
         test_accuracy, 
         label = "Test")
plt.title("Train and Test Accuracy over Training\nParametric model")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Accuracy_parametric.png')

## Model comparison
plt.figure()
plt.title('Model comparison: Test accuracy')
plt.plot(np.arange(1, n_epochs + 1, 10), 
         intro_test_accuracy, 
         label = "Intro")
plt.plot(np.arange(1, n_epochs + 1, 10), 
         dropout_test_accuracy, 
         label = "Dropout")
plt.plot(np.arange(1, n_epochs + 1, 10), 
         parametric_test_accuracy, 
         label = "Parametric")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc = 'best')
plt.savefig('Model_comparison.png')