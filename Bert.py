# Import statements
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, BertForSequenceClassification, BertTokenizer
from matplotlib import pyplot as plt
import csv

# Load the dataset
dataset_path = '/home/iai/Desktop/ohw/Fakeddit/dataset/multimodal_train.tsv'
df = pd.read_csv(dataset_path, sep='\t')

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
df['2_way_label'].plot(kind='hist', bins=20, title='2_way_label')
plt.show()

# Define a custom dataset class
class FakedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare the dataset
MAX_LEN = 128
BATCH_SIZE = 16

train_texts = df['clean_title'].tolist()
train_labels = df['2_way_label'].tolist()

train_dataset = FakedditDataset(
    texts=train_texts,
    labels=train_labels,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

# Training loop
EPOCHS = 3

# Open a file to save the results
with open('training_results.txt', 'w', encoding='utf-8') as f, open('training_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Epoch', 'Loss', 'Accuracy', 'F1 Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct_predictions = 0
        all_labels = []
        all_preds = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            correct_predictions += torch.sum(torch.argmax(logits, dim=1) == labels)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions.double() / len(train_loader.dataset)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Print metrics
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(f'Loss: {avg_loss}')
        print(f'Accuracy: {accuracy}')
        print(f'F1 Score: {f1}')

        # Save metrics to file
        f.write(f'Epoch {epoch + 1}/{EPOCHS}\n')
        f.write(f'Loss: {avg_loss}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write('\n')

        # Write metrics to CSV file
        writer.writerow({'Epoch': epoch + 1, 'Loss': avg_loss, 'Accuracy': accuracy.item(), 'F1 Score': f1})

print("Training complete.")