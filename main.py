import torch
from torch import nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit
from tqdm import tqdm

from ViT.vit import ViT



RANDOM_SEED = 42

BATCH_SIZE = 512
EPOCHS = 4


LEARNING_RATE = 1e-4
NUM_CLASSES = 10
PATHCH_SIZE = 4
IMG_SIZE = 28
IN_CHANNELS = 1
NUM_HEADS = 8
DROPOUT = 0.001
HIDDEN_DIM = 768
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
NUM_LAYERS = 4
EMBED_DIM = (PATHCH_SIZE ** 2) * IN_CHANNELS    #4**2 * 1 = 16
NUM_PATCHES = (IMG_SIZE // PATHCH_SIZE) ** 2    #28//4 **2 =  49
BLOCK_SIZE = NUM_PATCHES + 1
FORWARD_EXPANSION = 16


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"



class MNISTTrainDataset(Dataset):
    def __init__(self, images, labels, indicies):
        self.images = images
        self.labels = labels
        self.indicies = indicies
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        label = self.labels[idx]
        index = self.indicies[idx]
        image = self.transform(image)
        
        return {"image": image, "label": label, "index": index}



class MNISTValDataset(Dataset):
    def __init__(self, images, labels, indicies):
        self.images = images
        self.labels = labels
        self.indicies = indicies
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        label = self.labels[idx]
        index = self.indicies[idx]
        image = self.transform(image)
        
        return {"image": image, "label": label, "index": index}
    
    
    
class MNISTSubmitDataset(Dataset):
    def __init__(self, images, indicies):
        self.images = images
        self.indicies = indicies
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape((28, 28)).astype(np.uint8)
        index = self.indicies[idx]
        image = self.transform(image)
        
        return {"image": image, "index": index}
        





def main():
    train_df = pd.read_csv("digit-recognizer/train.csv")
    test_df = pd.read_csv("digit-recognizer/test.csv")
    submission_df = pd.read_csv("digit-recognizer/sample_submission.csv")
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=RANDOM_SEED, shuffle=True)
    
    model = ViT(NUM_PATCHES, NUM_CLASSES, PATHCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT, IN_CHANNELS, FORWARD_EXPANSION, BLOCK_SIZE).to(device)
    x = torch.randn(512, 1, 28, 28)

    
    
    train_dataset = MNISTTrainDataset(train_df.iloc[:, 1:].values.astype(np.uint8), train_df.iloc[:, 0].values, train_df.index.values)
    val_dataset = MNISTValDataset(val_df.iloc[:, 1:].values.astype(np.uint8), val_df.iloc[:, 0].values, val_df.index.values)
    test_dataset = MNISTSubmitDataset(test_df.values.astype(np.uint8), test_df.index.values)
    
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    val_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)
    
    start = timeit.default_timer()
    for epoch in tqdm(range(EPOCHS), position=0, leave=True):
        model.train()
        train_labels = []
        train_preds = []
        train_running_loss = 0
        
        for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_label["image"].float().to(device)
            label = img_label["label"].type(torch.uint8).to(device)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim=1)
            
            train_labels.extend(label.cpu().detach())
            train_preds.extend(y_pred_label.cpu().detach())
            
            loss = criterion(y_pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
            
        train_loss = train_running_loss / (idx + 1)
        
        
        model.eval()
        val_labels = []
        val_preds = []
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_label in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img_label["image"].float().to(device)
                label = img_label["label"].type(torch.uint8).to(device)
                y_pred = model(img)
                y_pred_label = torch.argmax(y_pred, dim=1)
                
                val_labels.extend(label.cpu().detach())
                val_preds.extend(y_pred_label.cpu().detach())
                
                loss = criterion(y_pred, label)
                val_running_loss += loss.item()

        val_loss = val_running_loss / (idx + 1)
        
        print("-"*30)
        print(f"Train Loss EPOCH{epoch+1}: {train_loss:.4f}")
        print(f"Val Loss EPOCH{epoch+1}: {val_loss:.4f}")        
        print(f"Train Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
        print(f"Valid Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
        print("-"*30)
        
    stop = timeit.default_timer()
    print(f"Training Time: {stop-start:.2f}s")        
          
    
if __name__ == '__main__':
    main()