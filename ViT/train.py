import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import VisionTransformer
from tqdm import tqdm

transforms = transforms.Compose(
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
)

BATCH_SIZE = 8

IMAGE_SIZE = 28
IN_CHANNEL = 1
PATCH_SIZE = 4
HEAD_NUM = 4
EMBED_DIM = (IMAGE_SIZE // PATCH_SIZE) ** 2
MLP_DIM = 256
NUM_LAYERS = 4
NUM_CLASSES = 10
DROPOUT = 0.01

EPOCHES = 10
LEARNING_RATE = 3E-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = datasets.MNIST(
    root = 'Foundational_Model/ViT/dataset', train=True, download=True, transform=transforms
)
val_set = datasets.MNIST(
    root='Foundational_Model/ViT/dataset', train=False, download=True, transform=transforms
)

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

model = VisionTransformer(
    image_size=IMAGE_SIZE,
    in_channels= IN_CHANNEL,
    patch_size=PATCH_SIZE,
    head_num=HEAD_NUM,
    embed_dim=EMBED_DIM,
    mlp_dim=MLP_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHES):
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader, position=0, leave=True):
        images, labels = images.to(device), labels.to(device)
        preds = model(images)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    model.eval()
    correct = 0
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, position=0, leave=True):
            images, labels = images.to(device), labels.to(device) 
            preds_val = model(images)

            val_loss = criterion(preds_val, labels)
            preds_arg = preds_val.argmax(dim=1)

            correct += (preds_arg==labels).sum().item()
            total_val_loss += val_loss.item()

    
    print(f"Epoch {epoch+1} | Traing Loss: {total_loss/len(train_loader):.4f} | Val Loss: {total_val_loss/len(val_loader):.4f} | "
          f"Acc: {correct/len(val_set)*100:.2f}%")