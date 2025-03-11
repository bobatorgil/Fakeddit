# Import statements
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import v2

dataset_path = '/home/iai/Desktop/ohw/Fakeddit/dataset/multimodal_train.tsv'
df = pd.read_csv(dataset_path, sep='\t')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")
    
# Drop unnecessary columns
df.drop(['6_way_label', '3_way_label', 'title'], axis=1, inplace=True)

# Splitting the dataset
df, df_backup = train_test_split(
    df,
    test_size=0.05,
    shuffle=True,
    stratify=df["2_way_label"]
)

# Reset indexes
df.reset_index(drop=True, inplace=True)

# Check for null values
print(df['clean_title'].isnull().sum())
print(df['id'].isnull().sum())
print(df['hasImage'].isnull().sum())

# Check for how many rows the column hasImage would be False
print(df['hasImage'].value_counts())

# Plot the distribution of 2_way_label
from matplotlib import pyplot as plt

df['2_way_label'].plot(kind='hist', bins=20, title='2_way_label')
plt.show()

# Replace NaN values with empty strings
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

# Define the image directory
image_dir = '/home/iai/Desktop/ohw/Fakeddit/dataset/images'

# Image transformations
image_transforms = v2.Compose([
    v2.Resize(size=256),
    v2.CenterCrop(size=224),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_id = self.dataframe.loc[index, 'id']
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        # Check if the image file is valid
        try:
            # Attempt to open the image file
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            label = self.dataframe.loc[index, '2_way_label']
            return image, label
        except (IOError, UnidentifiedImageError):
            # If the image file is invalid, skip it
            print(f"Skipping invalid image: {image_path}")
            # Instead of returning None, None, return the image and label from the previous index
            # if index > 0 else return the image and label from the next index
            new_index = index - 1 if index > 0 else index + 1
            return self.__getitem__(new_index)


# Create the dataset and dataloader
dataset = ImageDataset(df, image_dir, transform=image_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load the pre-trained ResNet50 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Modify the final fully connected layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 output classes
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3

# Open a file to save the results
results_path = '/home/iai/Desktop/ohw/Fakeddit/dataset/results.txt'
with open(results_path, 'w') as f:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            if images is None or labels is None:
                # Skip this iteration if the image is invalid
                continue
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Evaluation
        model.eval()  # Set the model to evaluation mode
        all_predictions = []
        all_labels = []

        with torch.no_grad():  # Disable gradient calculations during evaluation
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predictions = torch.max(outputs, 1)  # Get predicted class labels

                all_predictions.extend(predictions.cpu().numpy())  # Store predictions
                all_labels.extend(labels.cpu().numpy())  # Store true labels

        # Calculate accuracy and F1 score
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        # Save results to a file
        f.write(f"Epoch {epoch+1}/{num_epochs}\n")
        f.write(f"Loss: {epoch_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write('\n')

print("Training complete.")