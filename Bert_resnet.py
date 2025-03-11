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
from transformers import BertTokenizer, BertModel
from torch.cuda.amp import GradScaler, autocast

# Define paths
dataset_path = '/home/iai/Desktop/ohw/Fakeddit/dataset/multimodal_train.tsv'
image_dir = '/home/iai/Desktop/ohw/Fakeddit/dataset/images'

# Load dataset
df = pd.read_csv(dataset_path, sep='\t')

# Filter the DataFrame to include only rows with images
df = df[df['hasImage'] == True]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Replace NaN values with empty strings
df = df.replace(np.nan, '', regex=True)
df.fillna('', inplace=True)

# Image transformations
image_transforms = v2.Compose([
    v2.Resize(size=256),
    v2.CenterCrop(size=224),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the dataset class
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_id = row['id']
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        text = row['clean_title']
        label = row['2_way_label']

        # Process image
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except (IOError, UnidentifiedImageError):
            print(f"Skipping invalid image: {image_path}")
            new_index = index - 1 if index > 0 else index + 1
            return self.__getitem__(new_index)

        # Process text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create the dataset and dataloader
dataset = MultimodalDataset(df, image_dir, transform=image_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  # Use multiple workers

# Define the multimodal model
class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc = nn.Linear(self.bert.config.hidden_size + 2048, 2)  # 2 output classes

    def forward(self, bert_input_ids, attention_mask, images):
        bert_outputs = self.bert(input_ids=bert_input_ids, attention_mask=attention_mask)
        bert_pooled_output = bert_outputs.pooler_output

        resnet_outputs = self.resnet(images)

        combined = torch.cat((bert_pooled_output, resnet_outputs), dim=1)
        return self.fc(combined)

# Initialize the model, loss function, and optimizer
model = MultimodalModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()  # For mixed precision training

# Training loop
num_epochs = 3

# Open a file to save the results
results_path = '/home/iai/Desktop/ohw/Fakeddit/dataset/results2.txt'
with open(results_path, 'w', encoding='utf-8') as f:
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast():  # Mixed precision training
                outputs = model(bert_input_ids=input_ids, attention_mask=attention_mask, images=images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Evaluation
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(bert_input_ids=input_ids, attention_mask=attention_mask, images=images)
                _, predictions = torch.max(outputs, 1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

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