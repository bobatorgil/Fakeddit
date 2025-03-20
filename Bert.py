# Import statements
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, BertForSequenceClassification, BertTokenizer
import csv
import time
from torch.cuda.amp import GradScaler, autocast

# Load the training dataset
train_dataset_path = '/home/iai3/Desktop/ohw/Fakeddit/dataset/multimodal_train.tsv'
train_df = pd.read_csv(train_dataset_path, sep='\t')

# Load the validation dataset
validate_dataset_path = '/home/iai3/Desktop/ohw/Fakeddit/dataset/multimodal_validate.tsv'
validate_df = pd.read_csv(validate_dataset_path, sep='\t')

# Load the test dataset
test_dataset_path = '/home/iai3/Desktop/ohw/Fakeddit/dataset/multimodal_test_public.tsv'
test_df = pd.read_csv(test_dataset_path, sep='\t')

# Drop unnecessary columns
train_df.drop(['6_way_label', '3_way_label', 'title'], axis=1, inplace=True)
validate_df.drop(['6_way_label', '3_way_label', 'title'], axis=1, inplace=True)
test_df.drop(['6_way_label', '3_way_label', 'title'], axis=1, inplace=True)

# Reset indexes
train_df.reset_index(drop=True, inplace=True)
validate_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# Check for null values
print(train_df['clean_title'].isnull().sum())
print(train_df['id'].isnull().sum())
print(train_df['hasImage'].isnull().sum())

# Check for how many rows the column hasImage would be False
print(train_df['hasImage'].value_counts())

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
BATCH_SIZE = 8

train_texts = train_df['clean_title'].tolist()
train_labels = train_df['2_way_label'].tolist()

validate_texts = validate_df['clean_title'].tolist()
validate_labels = validate_df['2_way_label'].tolist()

test_texts = test_df['clean_title'].tolist()
test_labels = test_df['2_way_label'].tolist()

train_dataset = FakedditDataset(
    texts=train_texts,
    labels=train_labels,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

validate_dataset = FakedditDataset(
    texts=validate_texts,
    labels=validate_labels,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_dataset = FakedditDataset(
    texts=test_texts,
    labels=test_labels,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)  # Use multiple workers
validate_loader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Define the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
scaler = GradScaler()  # For mixed precision training

# Training loop with early stopping
MAX_EPOCHS = 20
PATIENCE = 3
best_val_accuracy = 0
patience_counter = 0

# Open a file to save the results
with open('training_results.txt', 'w', encoding='utf-8') as f, open('training_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Epoch', 'Loss', 'Accuracy', 'F1 Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    start_time = time.time()  # Start time for training

    for epoch in range(MAX_EPOCHS):
        print(f"Starting epoch {epoch + 1}/{MAX_EPOCHS}")
        model.train()
        total_loss = 0
        correct_predictions = 0
        all_labels = []
        all_preds = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}", leave=False):
            input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
            attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()
            with autocast():  # Mixed precision training
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            correct_predictions += torch.sum(torch.argmax(outputs.logits, dim=1) == labels)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions.double() / len(train_loader.dataset)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Print metrics
        print(f'Epoch {epoch + 1}/{MAX_EPOCHS}')
        print(f'Loss: {avg_loss}')
        print(f'Accuracy: {accuracy}')
        print(f'F1 Score: {f1}')

        # Save metrics to file
        f.write(f'Epoch {epoch + 1}/{MAX_EPOCHS}\n')
        f.write(f'Loss: {avg_loss}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write('\n')

        # Write metrics to CSV file
        writer.writerow({'Epoch': epoch + 1, 'Loss': avg_loss, 'Accuracy': accuracy.item(), 'F1 Score': f1})

        # Validation
        model.eval()
        val_correct_predictions = 0
        val_all_labels = []
        val_all_preds = []

        with torch.no_grad():
            for batch in tqdm(validate_loader, desc="Validating"):
                input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
                attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
                labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, predictions = torch.max(outputs.logits, dim=1)

                val_correct_predictions += torch.sum(predictions == labels)
                val_all_labels.extend(labels.cpu().numpy())
                val_all_preds.extend(predictions.cpu().numpy())

        val_accuracy = val_correct_predictions.double() / len(validate_loader.dataset)
        val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted')

        print(f'Validation Accuracy: {val_accuracy}')
        print(f'Validation F1 Score: {val_f1}')

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

    end_time = time.time()  # End time for training
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")

print("Training complete.")

# Load the best model
model.load_state_dict(torch.load('best_model.pt'))

# Evaluation on the test set
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to('cuda' if torch.cuda.is_available() else 'cpu')
        attention_mask = batch['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu')
        labels = batch['label'].to('cuda' if torch.cuda.is_available() else 'cpu')

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predictions = torch.max(outputs.logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy and F1 score
accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions, average='weighted')

print(f'Test Accuracy: {accuracy}')
print(f'Test F1 Score: {f1}')

# Save test results to a file
with open('test_results.txt', 'w', encoding='utf-8') as f:
    f.write(f'Test Accuracy: {accuracy}\n')
    f.write(f'Test F1 Score: {f1}\n')

# Save training time to a file
with open('training_time.txt', 'w', encoding='utf-8') as f:
    f.write(f'Total training time: {training_time:.2f} seconds\n')