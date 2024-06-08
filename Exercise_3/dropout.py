# Math and plots
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import scipy

# PyTorch
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.nn.functional import conv2d, max_pool2d, cross_entropy

# Tools
import time
from tqdm import tqdm

plt.rc("figure", dpi = 200)

batch_size = 100

# transform images into normalized tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5,), std = (0.5,))
])

train_dataset = datasets.MNIST(
    "./",
    download=True,
    train=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    "./",
    download = True,
    train = False,
    transform = transform,
)

train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 1,
    pin_memory = True,
)

test_dataloader = DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 1,
    pin_memory = True,
)

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
def dropout(X, p_drop=0.5):
    retain_prob = 1 - p_drop
    mask = (torch.rand(X.shape) < retain_prob).float()
    X = X * mask / retain_prob
    return X


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


def dropout_model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
#    print('X:', X.shape, dropout(X).shape)
    h = rectify(X @ w_h)
    
    h = dropout(h, p_drop_hidden)
#    print('h:', h.shape, dropout(h).shape)
    h2 = rectify(h @ w_h2)
    
    h2 = dropout(h2, p_drop_hidden)
#    print('h2:', h2.shape, dropout(h2).shape)
    pre_softmax = h2 @ w_o
    return pre_softmax


def calculate_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    return correct / target.size(0)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    start_epoch = checkpoint['epoch']
    w_h, w_h2, w_o = checkpoint['model_state_dict']
    optimizer = RMSprop(params=[w_h, w_h2, w_o])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_loss = checkpoint['train_loss']
    test_loss = checkpoint['test_loss']
    train_accuracy = checkpoint['train_accuracy']
    test_accuracy = checkpoint['test_accuracy']
    return (start_epoch, w_h, w_h2, w_o, optimizer, 
            train_loss, test_loss, train_accuracy, test_accuracy)


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
            
# Try to load from checkpoint
try:
    start_epoch, w_h, w_h2, w_o, optimizer, train_loss, test_loss, train_accuracy, test_accuracy = load_checkpoint('checkpoint_epoch_30.pth')
    print(f"Resuming from epoch {start_epoch}...")
except FileNotFoundError:
    start_epoch = 0
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    print("Starting from scratch...")


n_epochs = 100

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

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
        }, f'checkpoint_epoch_{epoch}.pth')
        
    end_time = time.time()
    
    
plt.plot(np.arange(n_epochs + 1), 
         train_loss, 
         label = "Train")
plt.plot(np.arange(1, n_epochs + 2, 10), 
         test_loss, 
         label = "Test")
plt.title("Train and Test Loss over Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('Loss_dropout.png')
plt.legend()
