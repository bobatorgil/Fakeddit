# Import statements
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import v2
from transformers import BertTokenizer, BertModel

# Paths to datasets
train_dataset_path = '/home/iai3/Desktop/ohw/Fakeddit/dataset/multimodal_train.tsv'
validate_dataset_path = '/home/iai3/Desktop/ohw/Fakeddit/dataset/multimodal_validate.tsv'
test_dataset_path = '/home/iai3/Desktop/ohw/Fakeddit/dataset/multimodal_test_public.tsv'

# Load datasets
train_df = pd.read_csv(train_dataset_path, sep='\t')
validate_df = pd.read_csv(validate_dataset_path, sep='\t')
test_df = pd.read_csv(test_dataset_path, sep='\t')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")
    
# Drop unnecessary columns
train_df.drop(['6_way_label', '3_way_label', 'title'], axis=1, inplace=True)
validate_df.drop(['6_way_label', '3_way_label', 'title'], axis=1, inplace=True)
test_df.drop(['6_way_label', '3_way_label', 'title'], axis=1, inplace=True)

# Replace NaN values with empty strings
train_df = train_df.replace(np.nan, '', regex=True)
train_df.fillna('', inplace=True)
validate_df = validate_df.replace(np.nan, '', regex=True)
validate_df.fillna('', inplace=True)
test_df = test_df.replace(np.nan, '', regex=True)
test_df.fillna('', inplace=True)

# Define the image directory
image_dir = '/home/iai3/Desktop/ohw/Fakeddit/dataset/images'

# Construct image paths
train_df['image_path'] = image_dir + '/' + train_df['id'].astype(str) + '.jpg'
validate_df['image_path'] = image_dir + '/' + validate_df['id'].astype(str) + '.jpg'
test_df['image_path'] = image_dir + '/' + test_df['id'].astype(str) + '.jpg'

# Filter the DataFrame to include only rows with existing images
train_df = train_df[train_df['image_path'].apply(lambda x: os.path.exists(x))]
validate_df = validate_df[validate_df['image_path'].apply(lambda x: os.path.exists(x))]
test_df = test_df[test_df['image_path'].apply(lambda x: os.path.exists(x))]

# Reset index after filtering
train_df.reset_index(drop=True, inplace=True)
validate_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Debugging statements
print(f"Number of training samples: {len(train_df)}")
print(f"Number of validation samples: {len(validate_df)}")
print(f"Number of test samples: {len(test_df)}")

# Image transformations
image_transforms = v2.Compose([
    v2.Resize(size=256),
    v2.CenterCrop(size=224),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MultimodalDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, tokenizer=None, max_length=128):
        self.dataframe = dataframe
        self.image_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_id = self.dataframe.loc[index, 'id']
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        text = self.dataframe.loc[index, 'clean_title']
        label = self.dataframe.loc[index, '2_way_label']

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
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create the datasets and dataloaders
train_dataset = MultimodalDataset(train_df, image_dir, transform=image_transforms, tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

validate_dataset = MultimodalDataset(validate_df, image_dir, transform=image_transforms, tokenizer=tokenizer)
validate_dataloader = DataLoader(validate_dataset, batch_size=32, shuffle=False)

test_dataset = MultimodalDataset(test_df, image_dir, transform=image_transforms, tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pre-trained ResNet50 model
resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Identity()  # Remove the final fully connected layer

# Load the pre-trained BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')

class MultimodalModel(nn.Module):
    def __init__(self, resnet_model, bert_model, num_ftrs, num_classes=2):
        super(MultimodalModel, self).__init__()
        self.resnet_model = resnet_model
        self.bert_model = bert_model
        self.fc = nn.Linear(num_ftrs + bert_model.config.hidden_size, num_classes)

    def forward(self, image, input_ids, attention_mask):
        # Get image features
        image_features = self.resnet_model(image)

        # Get text features
        bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token representation

        # Concatenate features
        combined_features = torch.cat((image_features, text_features), dim=1)

        # Apply final classification layer
        output = self.fc(combined_features)
        return output

# Initialize the multimodal model
model = MultimodalModel(resnet_model, bert_model, num_ftrs)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping
num_epochs = 20
patience = 3
best_val_accuracy = 0.0
patience_counter = 0

# Open a file to save the results
results_path = '/home/iai3/Desktop/ohw/Fakeddit/dataset/results.txt'
with open(results_path, 'w') as f:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_dataloader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Evaluation on the validation set
        model.eval()  # Set the model to evaluation mode
        all_predictions = []
        all_labels = []

        with torch.no_grad():  # Disable gradient calculations during evaluation
            for batch in validate_dataloader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images, input_ids, attention_mask)
                _, predictions = torch.max(outputs, 1)  # Get predicted class labels

                all_predictions.extend(predictions.cpu().numpy())  # Store predictions
                all_labels.extend(labels.cpu().numpy())  # Store true labels

        # Calculate accuracy and F1 score
        val_accuracy = accuracy_score(all_labels, all_predictions)
        val_f1 = f1_score(all_labels, all_predictions)

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}")

        # Save results to a file
        f.write(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}")

        # Save results to a file
        f.write(f"Epoch {epoch+1}/{num_epochs}\n")
        f.write(f"Loss: {epoch_loss:.4f}\n")
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Validation F1 Score: {val_f1:.4f}\n")
        f.write('\n')

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

# Final evaluation on the test set
model.eval()  # Set the model to evaluation mode
all_predictions = []
all_labels = []

with torch.no_grad():  # Disable gradient calculations during evaluation
    for batch in test_dataloader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(images, input_ids, attention_mask)
        _, predictions = torch.max(outputs, 1)  # Get predicted class labels

        all_predictions.extend(predictions.cpu().numpy())  # Store predictions
        all_labels.extend(labels.cpu().numpy())  # Store true labels

# Calculate accuracy and F1 score
test_accuracy = accuracy_score(all_labels, all_predictions)
test_f1 = f1_score(all_labels, all_predictions)

print(f"Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")

# Save test results to a file
with open(results_path, 'a') as f:
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test F1 Score: {test_f1:.4f}\n")

print("Training complete.")