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

## Task 1: Intro
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

plt.rc("figure", dpi=100)

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
        }, f'checkpoint_intro_epoch_{epoch}.pth')

## Plots
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
