#%%

# Load and normalize CIFAR10
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import random
import numpy as np
import os

random_seed = 113
batch_size = 32

os.environ["PYTHONHASHSEED"] = str(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(random_seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
# if device == "cuda":
#     torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)

testset = CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# %%
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(make_grid(images))
# print labels
print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from activations.activations import molu

x = torch.tensor(3)
print(type(x))


print(torch.nn.ReLU()(x))
#%%
torch.nn.functional.relu(x)
#%%
# Define a convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NewActivationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(molu(self.conv1(x)))
        x = self.pool(molu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = molu(self.fc1(x))
        x = molu(self.fc2(x))
        x = self.fc3(x)
        return x


# net = Net()
# new_net = NewActivationNet()

#%%
import torch.nn.functional as F
from models.resnet import ResNet18, M_ResNet18

# Define a loss function and optimizer
import torch.optim as optim

epochs = 3
learning_rate = 0.001
momentum = 0.9

net = ResNet18()
net = net.to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

m_net = M_ResNet18()
m_net = m_net.to(device)
m_optimizer = optim.SGD(m_net.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(trainloader.dataset) for i in range(epochs + 1)]
corrects = []

m_train_losses = []
m_train_counter = []
m_test_losses = []
m_test_counter = [i * len(trainloader.dataset) for i in range(epochs + 1)]
m_corrects = []


def train(epoch):
    net.train()
    train_loss = 0
    for batch_idx, (input, target) in enumerate(trainloader):
        input, target = input.to(device), target.to(device)

        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        if batch_idx % 2000 == 1999:  # print every 2000 mini-batches
            train_losses.append(train_loss.item())
            print(
                f"Train Epoch:{epoch}, batch index:{batch_idx + 1}, loss: {train_loss / 2000:.4f}"
            )
            torch.save(net.state_dict(), "./results/cifar_net.pth")
            torch.save(optimizer.state_dict(), "./results/cifar_net.pth")


def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for input, target in testloader:
            input, target = input.to(device), target.to(device)
            output = net(input)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)
    corrects.append(correct)
    print(
        f"Test set: Avg. loss: {test_loss:.4f}, Accuracy: {100.0 * correct / len(testloader.dataset):.2f}%"
    )


def m_train(epoch):
    m_net.train()
    m_train_loss = 0
    for batch_idx, (m_input, target) in enumerate(trainloader):
        m_input, target = m_input.to(device), target.to(device)

        m_optimizer.zero_grad()

        # forward + backward + optimize
        m_output = m_net(m_input)
        m_loss = F.nll_loss(m_output, target)
        m_loss.backward()
        m_optimizer.step()

        # print statistics
        m_train_loss += m_loss.item()
        if batch_idx % 2000 == 1999:  # print every 2000 mini-batches
            m_train_losses.append(m_train_loss.item())
            print(
                f"Train Epoch:{epoch}, batch index:{batch_idx + 1}, loss: {m_train_loss / 2000:.4f}"
            )
            torch.save(m_net.state_dict(), "./results/cifar_m_net.pth")
            torch.save(m_optimizer.state_dict(), "./results/cifar_m_net.pth")


def m_test():
    m_net.eval()
    m_test_loss = 0
    m_correct = 0
    with torch.no_grad():
        for m_input, target in testloader:
            m_input, target = m_input.to(device), target.to(device)
            m_output = m_net(m_input)
            m_test_loss += F.nll_loss(m_output, target, size_average=False).item()
            m_pred = m_output.data.max(1, keepdim=True)[1]
            m_correct += m_pred.eq(target.data.view_as(m_pred)).sum()
    m_test_loss /= len(testloader.dataset)
    m_test_losses.append(m_test_loss)
    m_corrects.append(m_correct)
    print(
        f"Test set: Avg. loss: {m_test_loss:.4f}, Accuracy: {100.0 * m_correct / len(testloader.dataset):.2f}%"
    )


#%%
# import time

# start_time = time.time()
# Train the network
test()
for epoch in range(1, epochs + 1):
    train(epoch)
    test()  # loop over the dataset multiple times

# total_time = time.time() - start_time
# print(f" Duration: {total_time/60} mins")

# batch 32
# 3 epochs
# Test set: Avg. loss: 0.9021, Accuracy: 68.01%
# Test set: Avg. loss: 0.7579, Accuracy: 73.94%
# Test set: Avg. loss: 0.6791, Accuracy: 76.80%

# 10 epochs
# Test set: Avg. loss: 0.9021, Accuracy: 68.01%
# Test set: Avg. loss: 0.7579, Accuracy: 73.94%
# Test set: Avg. loss: 0.6791, Accuracy: 76.80%
# Test set: Avg. loss: 0.6834, Accuracy: 77.64%
# Test set: Avg. loss: 0.7301, Accuracy: 78.01%
# Test set: Avg. loss: 0.7879, Accuracy: 77.80%
# Test set: Avg. loss: 0.7952, Accuracy: 78.41%
# Test set: Avg. loss: 0.8182, Accuracy: 79.04%
# Test set: Avg. loss: 0.9138, Accuracy: 78.25%
# Test set: Avg. loss: 0.7859, Accuracy: 81.13%

# batch 64
# 3 epochs
# Test set: Avg. loss: 1.0751, Accuracy: 61.89%
# Test set: Avg. loss: 0.8413, Accuracy: 70.10%
# Test set: Avg. loss: 0.7413, Accuracy: 74.27%
#%%
# import time

# start_time = time.time()
# Train the network
m_test()
for epoch in range(1, epochs + 1):
    m_train(epoch)
    m_test()  # loop over the dataset multiple times

# total_time = time.time() - start_time
# print(f" Duration: {total_time/60} mins")

# 32 batch
# 4 epochs
# 1, 1
# Test set: Avg. loss: 0.8627, Accuracy: 68.98%
# Test set: Avg. loss: 0.7092, Accuracy: 75.53%
# Test set: Avg. loss: 0.6331, Accuracy: 78.26%
# Test set: Avg. loss: 0.7129, Accuracy: 77.30%

# 10 epochs
# Test set: Avg. loss: 0.8627, Accuracy: 68.98%
# Test set: Avg. loss: 0.7092, Accuracy: 75.53%
# Test set: Avg. loss: 0.6331, Accuracy: 78.26%
# Test set: Avg. loss: 0.7129, Accuracy: 77.30%
# Test set: Avg. loss: 0.6951, Accuracy: 79.29%
# Test set: Avg. loss: 0.7547, Accuracy: 79.04%
# Test set: Avg. loss: 0.7569, Accuracy: 79.83%
# Test set: Avg. loss: 0.8708, Accuracy: 78.88%
# Test set: Avg. loss: 0.8648, Accuracy: 79.84%
# Test set: Avg. loss: 0.7960, Accuracy: 80.97%
#%%
from torchinfo import summary

examples = enumerate(testloader)
batch_idx, (example_data, example_targets) = next(examples)

example_data.shape
#%%
summary(net, (32, 3, 32, 32))
#%%
batch_idx = iter(testloader)
images, labels = next(batch_idx)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# print images
img_grid = make_grid(images)
imshow(img_grid)
print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(32)))

#%%
#
#
#
#
#
# Define a loss function and optimizer
# import torch.nn.functional as F
# from models.resnet import ResNet18
# import torch.optim as optim

# learning_rate = 0.001
# momentum = 0.9

# net = ResNet18()
# net = net.to(device)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# # Train the network
# for epoch in range(5):  # loop over the dataset multiple times

#     train_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data.to(device), labels.to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = F.nll_loss(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         train_loss += loss.item()
#         if i % 2000 == 1999:  # print every 2000 mini-batches
#             print(f"[{epoch + 1}, {i + 1:5d}] loss: {train_loss / 2000:.4f}")
#             train_loss = 0.0

# print("Finished Training")
# PATH = "./results/cifar_new_net.pth"
# torch.save(net.state_dict(), PATH)
# %%
#
#
#
#
# #
# net = ResNet18()
# net = net.to(device)
# net.load_state_dict(torch.load(PATH))
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)

# correct = 0
# total = 0
# # since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # calculate outputs by running images through the network
#         outputs = net(images)
#         # the class with the highest energy is what we choose as prediction
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(
#     f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
# )

# # 3 epochs
# # 2, 2, Accuracy of the network on the 10000 test images: 61 %

# %%
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data, label in testloader:
        images, labels = data.to(device), label.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.2f} %")
# 32 batch
# 4 epochs
# Accuracy for class: plane is 84.40 %
# Accuracy for class: car   is 89.50 %
# Accuracy for class: bird  is 76.50 %
# Accuracy for class: cat   is 62.70 %
# Accuracy for class: deer  is 80.40 %
# Accuracy for class: dog   is 47.40 %
# Accuracy for class: frog  is 83.10 %
# Accuracy for class: horse is 86.50 %
# Accuracy for class: ship  is 81.40 %
# Accuracy for class: truck is 89.10 %

# 64 batch
# 3 epochs
# Accuracy for class: plane is 81.60 %
# Accuracy for class: car   is 89.60 %
# Accuracy for class: bird  is 59.20 %
# Accuracy for class: cat   is 45.80 %
# Accuracy for class: deer  is 71.60 %
# Accuracy for class: dog   is 71.70 %
# Accuracy for class: frog  is 79.20 %
# Accuracy for class: horse is 83.30 %
# Accuracy for class: ship  is 77.50 %
# Accuracy for class: truck is 83.20 %

# %%
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data, label in testloader:
        images, labels = data.to(device), label.to(device)
        outputs = m_net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:5s} is {accuracy:.2f} %")

# 4 epochs
# Accuracy for class: plane is 83.10 %
# Accuracy for class: car   is 94.20 %
# Accuracy for class: bird  is 61.90 %
# Accuracy for class: cat   is 43.90 %
# Accuracy for class: deer  is 71.70 %
# Accuracy for class: dog   is 76.20 %
# Accuracy for class: frog  is 85.70 %
# Accuracy for class: horse is 81.10 %
# Accuracy for class: ship  is 87.40 %
# Accuracy for class: truck is 87.80 %

# # 10 epochs
# Accuracy for class: plane is 85.20 %
# Accuracy for class: car   is 94.00 %
# Accuracy for class: bird  is 73.40 %
# Accuracy for class: cat   is 68.00 %
# Accuracy for class: deer  is 71.30 %
# Accuracy for class: dog   is 73.00 %
# Accuracy for class: frog  is 82.80 %
# Accuracy for class: horse is 86.20 %
# Accuracy for class: ship  is 90.70 %
# Accuracy for class: truck is 85.10 %

# %%
img = images[0].view(1, 3072)

# we are turning off the gradients
with torch.no_grad():
    model_prediction = net.forward(img)
probabilities = F.softmax(model_prediction, dim=1).detach().cpu().numpy().squeeze()

print(probabilities)
#%%
img_grid = make_grid(images)
imshow(img_grid)
net = ResNet18()
net.load_state_dict(torch.load(PATH))
fig, (ax1, ax2) = plt.subplots(figsize=(6, 8), ncols=2)
img = img.view(3, 32, 32)

ax1.axis("off")
ax2.barh(np.arange(10), probabilities, color="r")
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(10))
ax2.set_yticklabels(**classes)
ax2.set_title("Class Probability")
ax2.set_xlim(0, 1.1)

plt.tight_layout()

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(range(4), test_losses)
# %%
train_losses
# %%
test_losses
# %%
