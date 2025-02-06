#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tqdm')


# In[2]:


import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm


# In[3]:


import torchvision.transforms as transforms


# In[7]:


import sys
import os

file_path = os.path.abspath("C:/Users/Alma/Downloads/")  # Get absolute path
sys.path.append(file_path)


# In[17]:


with open("C:/Users/Alma/Downloads/Medmamba.py", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Remove hidden characters
clean_lines = [line.replace("\t", "    ").rstrip() + "\n" for line in lines]

with open("C:/Users/Alma/Downloads/Medmamba_fixed.py", "w", encoding="utf-8") as file:
    file.writelines(clean_lines)

print("Fixed file saved as Medmamba_fixed.py")


# In[18]:


file_path = "C:/Users/Alma/Downloads/Medmamba_fixed.py"  # Use the cleaned file
spec = importlib.util.spec_from_file_location("Medmamba", file_path)
medmamba = importlib.util.module_from_spec(spec)
sys.modules["Medmamba"] = medmamba
spec.loader.exec_module(medmamba)


# In[14]:


import Medmamba


# In[54]:


from Medmamba import VSSM as medmamba  # import model


# In[55]:


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = datasets.ImageFolder(root=r"\Users\Alma\Downloads\Covid19-Pneumonia-Normal Chest X-Ray Images Dataset",
                                         transform=data_transform["train"])

    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=r"\Users\Alma\Downloads\valid_test",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))


# In[56]:


# Define the number of classes (e.g., 10 classes for a classification task)
num_classes = 10  # Modify based on your task

# Now initialize the model
net = VSSM(num_classes=num_classes)


# In[57]:


net = VSSM(num_classes=num_classes)


# In[58]:


# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Now, move your model to the device
net.to(device)

# Then you can proceed with your loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


# In[59]:


from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define any necessary transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Modify as per your dataset
])

# Assuming you're using a dataset like CIFAR-10 or MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create the DataLoader for training data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Now you can use train_loader in your training loop
train_steps = len(train_loader)


# In[60]:


import torchvision
import torchvision.transforms as transforms


# In[61]:


transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor first
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Manually repeat channels
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
images, labels = next(iter(train_loader))
print(images.shape)  # Should print torch.Size([64, 3, 28, 28])


# In[62]:


def selective_scan_fn(x: torch.Tensor):
    return x  # Simply returns the input unchanged for now


# In[63]:


net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

epochs = 10
best_acc = 0.0
model_name="Medmamba"
save_path = './{}Net.pth'.format(model_name)
train_steps = len(train_loader)
for epoch in range(epochs):
        # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

            # print statistics
        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()

