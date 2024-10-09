import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv
import argparse
import yaml
import arch
import optimizers

import os
import cv2
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from torch.cuda.amp import autocast, GradScaler
from arch import ConvNet

    
    
class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, path_dir1:str, path_dir2:str):
        super().__init__()
        
        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2
        
        
        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))
        
    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)
    
    def __getitem__(self, idx):
        
        if idx < len(self.dir1_list):
            class_id = 0
            img_path = os.path.join(self.path_dir1, self.dir1_list[idx])
        else:
            class_id = 1
            idx -= len(self.dir1_list)
            img_path = os.path.join(self.path_dir2, self.dir2_list[idx])
            
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # bgr to rgb
        img = img.astype(np.float32)
        img = img / 255.0
        
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        
        img = img.transpose([2, 0, 1])
        
        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)
        
        
        return {'img': t_img, 'label': t_class_id}

  
parser = argparse.ArgumentParser()
parser.add_argument('-option', type=str, required=True, help='Path to the option file') 
args = parser.parse_args()  
print(args.option)
option_path = args.option
    
option_path = 'ResNet_1/config.yaml'

with open(option_path, 'r') as file_option:
    option = yaml.safe_load(file_option)
    
    

train_dogs_path = option['train_dogs_path']
train_cats_path = option['train_cats_path']

test_dogs_path = option['test_dogs_path']
test_cats_path = option['test_cats_path']

train_ds_catdogs = Dataset2class(train_dogs_path, train_cats_path)
test_ds_catdogs = Dataset2class(test_dogs_path, test_cats_path)


batch_size = option['batch_size']


train_loader = torch.utils.data.DataLoader(
    train_ds_catdogs, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

test_loader = torch.utils.data.DataLoader(
   test_ds_catdogs, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
   )


option_network = option['network']
model = arch.get_network(option_network)

for sample in train_loader:
    img = sample['img']
    label = sample['label']
    model(img)
    break

loss_fn = nn.CrossEntropyLoss()
option_optimizer = option['optimizer']
optimizer = optimizers.get_optimizer(model.parameters(), option_optimizer)
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, 
    gamma=0.6
)

def accuracy(pred, label):
    answer = F.softmax(pred.detach(), dim=1).numpy().argmax(axis=1) == label.numpy().argmax(axis=1)
    return answer.mean()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.float().to(DEVICE)
loss_fn = loss_fn.to(DEVICE)

use_amp = True
scaler = torch.amp.GradScaler('cuda')

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

epochs = 10
loss_epochs_list = []
acc_epochs_list = []

for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    pbar = tqdm(train_loader)
    for sample in pbar:
        img, label = sample['img'], sample['label']
        label = F.one_hot(label, 2).float()
        img = img.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            img = img.float()
            pred = model(img)
            loss = loss_fn(pred, label)
            
        scaler.scale(loss).backward()
        loss_item = loss.item()
        loss_val += loss_item
        
        scaler.step(optimizer)
        scaler.update()
        
        acc_current = accuracy(pred.cpu().float(), label.cpu().float())
        acc_val += acc_current
        
        pbar.set_description(f'loss: {loss_item:.4f}\taccuracy: {acc_current:3f}')
    scheduler.step()
    loss_epochs_list.append(loss_val / len(train_loader))
    acc_epochs_list.append(acc_val / len(train_loader))
    print(loss_epochs_list[-1])
    print(acc_epochs_list[-1])

loss_epochs_list_resnet_v3 = loss_epochs_list.copy()
acc_epochs_list_resnet_v3 = acc_epochs_list.copy()

plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
plt.title('Loss')
plt.xlabel('Epochs')
# plt.plot(loss_epochs_list_resnet_v1, color='blue')
# plt.plot(loss_epochs_list_resnet_v2, color='red')
plt.plot(loss_epochs_list_resnet_v3, color='green')
plt.legend(['v1', 'v2', 'v3'])

plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.xlabel('Epochs')
# plt.plot(acc_epochs_list_resnet_v1, color='blue')
# plt.plot(acc_epochs_list_resnet_v2, color='red')
plt.plot(acc_epochs_list_resnet_v3, color='green')
plt.legend(['v1', 'v2', 'v3'])
plt.show()

