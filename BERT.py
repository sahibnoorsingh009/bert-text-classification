import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Load dataset
file_path = '/Users/sahibnoorsingh/Desktop/css_project/OLID.csv'
df = pd.read_csv(file_path)

# Encode labels
label_mapping = {'NOT': 0, 'OFF': 1}  # Adjust based on dataset labels
df['subtask_a'] = df['subtask_a'].map(label_mapping)

# Remove NaN values
df = df.dropna(subset=['cleaned_tweet'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_tweet'], df['subtask_a'], test_size=0.2, random_state=1)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

# Convert labels to PyTorch tensors
train_labels = torch.tensor(y_train.values)
test_labels = torch.tensor(y_test.values)
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Convert data to PyTorch dataset
train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Convert data to PyTorch dataset
train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

from transformers import BertForSequenceClassification

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",          # Save model here
    evaluation_strategy="epoch",      # Evaluate at each epoch
    per_device_train_batch_size=8,    # Batch size for training
    per_device_eval_batch_size=8,     # Batch size for evaluation
    num_train_epochs=3,               # Number of epochs
    logging_dir="./logs",             # Log directory
    logging_steps=500,                # Log every 500 steps
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
‚
