import pandas as pd
import numpy as np
import torch
# import matplotlib.pyplot as plt
# import sklearn
# import seaborn as sns
# from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm

pd.set_option('display.max_columns', None)

# Instantiate the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Define a class Dataset to transform the raw data from the DataFrame into a BERT/PyTorch-compatible format
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = list(df['label'])
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") \
                      for text in df['text']]
        # Now each instance of 'text' in the DataFrame is encoded as a 512-token BERT embedding

    def classes(self):
        return self.labels

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __len__(self):
        return len(self.labels)

    # Indexing
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

# For evaluation
class NoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") \
                      for text in df['text']]
        # Now each instance of 'text' in the DataFrame is encoded as a 512-token BERT embedding

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __len__(self):
        return len(self.texts)

    # Indexing
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        return batch_texts


# Define the NN
class FakeBERT(nn.Module):
    def __init__(self, dropout=0.3):
        super(FakeBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        # BERT hidden output is not necessary for a single linear layer, so only use the pooled ouput
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        # Apply dropout to prevent overfitting
        dropout_output = self.dropout(pooled_output)
        # Feed into the linear layer
        linear_output = self.linear(dropout_output)
        # Apply ReLU
        final_layer = self.relu(linear_output)
        return final_layer


# Define the training pipeline
def train(model, train_data, val_data, learning_rate, epochs):
    # Tokenize the training and validation text
    train, val = Dataset(train_data), Dataset(val_data)
    # Use the DataLoader to convert to iterable
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    # I'm just a poor boy, I need no sympathy but I could sure use a GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # Binary classification, so use cross entropy loss
    criterion = nn.CrossEntropyLoss()
    # Use ADAM to optimize the learning rate
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

    # Begin the training cycle
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                        | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                        | Val Loss: {total_loss_val / len(val_data): .3f} \
                        | Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluate(model, test_data):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    tests = []
    pred = []

    # If test labels are provided, use those to compute accuracy
    if 'label' in list(test_data.columns):
        test = Dataset(test_data)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

        with torch.no_grad():
            for test_input, test_label in test_dataloader:
                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
                pred.append(output.argmax(dim=1))

        print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

    # If no labels are provided, just compute predictions
    else:
        test = NoLabelDataset(test_data)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

        with torch.no_grad():
            for test_input in test_dataloader:
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                pred.append(output.argmax(dim=1))

    return list(np.array([tens.numpy() for tens in pred]).ravel())

# def main(t=False, random_seed=112):
path = input("Please input a file path: ")
df = pd.read_csv(path)
df['text'] = df['text'].str.replace(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', regex=True).str.replace(r'\s+', ' ', regex=True)
df = df.fillna('')

# Create train, validation, and test splits
random_seed = 42
np.random.seed(random_seed)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=random_seed), [int(0.8 * len(df)), int(0.9 * len(df))])

model = FakeBERT()

# if True:
#     epochs, lr = 1, 1e-6
#     train(model, df_train, df_val, lr, epochs)
#     torch.save(model.state_dict(), 'strongbaseline.pt')

# else:
model.load_state_dict(torch.load('strongbaseline.pt'))

model.eval()
pred = evaluate(model, df_test)

pred_df = pd.Series(pred, name='label')
pred_df.reset_index(drop=True).to_csv('test_predictions.csv')

