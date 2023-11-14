from setup import * 
from dgllife.utils.splitters import ScaffoldSplitter

# Try different values
HIDDEN_DIM = 100
DROPOUT = 0
NUM_EPOCHS = 10

class RNN(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(in_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(
            hidden_dim,
            hidden_dim,
            dropout=DROPOUT,
            batch_first=True,
            bidirectional=True
        )
        self.fc = torch.nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, batch):
        data = batch["data"]
        pad_mask = batch["pad_mask"]
        max_len = data.shape[1]

        # Embed each input token into a vector
        emb = self.embedding(data)

        # Compute lengths from padding mask
        lengths = pad_mask.sum(dim=1)

        # Ignore padding
        out = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths=lengths, batch_first=True, enforce_sorted=False)

        # Pass it to the LSTM (which outputs out, state). Ignore the state.
        out = self.lstm(out)[0]

        # Unpack
        out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=max_len)[0]

        # Compute the average vector for the sequence
        mask_3d = pad_mask.unsqueeze(-1).expand(out.size())
        out *= mask_3d  # Apply mask to zero out padding
        out = out.sum(1) / lengths.unsqueeze(1).float()

        # Apply the fc layer
        out = self.fc(out)
        return out

vocab = {"~": 0}
def process_sample(sample, max_length):
    smiles = sample[0]
    labels = sample[2]
    mask = sample[3]

    tok_ids = []
    for token in smiles:
        if token not in vocab:
            vocab[token] = len(vocab)
            tok_id = len(vocab)
        else:
            tok_id = vocab[token]
        tok_ids.append(tok_id)

    arr = torch.tensor(tok_ids).long()
    return {"data": arr, "labels": labels, "mask": mask}

# NON-RANDOM SPLITTING
def create_dataset():
    max_length = max(len(x[0]) for x in Tox21)
    train, _, test = ScaffoldSplitter.train_val_test_split(Tox21, frac_val=0, frac_test=0.2)
    train = list(map(lambda x: process_sample(x, max_length), train))
    test = list(map(lambda x: process_sample(x, max_length), test))
    return train, test

def create_model():
    return RNN(len(vocab) + 1, HIDDEN_DIM, 12)

def collate_fn(data):

    tok_ids = [d["data"] for d in data]
    pad_mask = [torch.ones_like(d["data"]) for d in data]
    labels = [d["labels"] for d in data]
    mask = [d["mask"] for d in data]

    tok_ids = torch.nn.utils.rnn.pad_sequence(tok_ids, batch_first=True)
    pad_mask = torch.nn.utils.rnn.pad_sequence(pad_mask, batch_first=True)
    labels = torch.stack(labels)
    mask = torch.stack(mask)

    return {"data": tok_ids, "labels": labels, "mask": mask, "pad_mask": pad_mask}


train_set, test_set = create_dataset()
model = create_model()
train_dl = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_dl = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
train_loss, train_aucs, test_aucs = train(model, train_dl, test_dl, num_epochs=NUM_EPOCHS)

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(train_loss)
ax1.set_title("Training Loss")
ax2.plot(train_aucs)
ax2.set_title("Training ROC-AUC")
ax3.plot(test_aucs)
ax3.set_title("Test ROC-AUC")

print("Final Training ROC-AUC: ", train_aucs[-1])
print("Best Training ROC-AUC: ", max(train_aucs))
print("\nFinal Testing ROC-AUC: ", test_aucs[-1])
print("Best Testing ROC-AUC: ", max(test_aucs))

# Results
# With the models ran on the random and non-random split datasets and the hidden dimensions and dropout being set at 100 and 0 respectively, it seems like the performance of the LSTM model with this non-random splitting of the data decreases: the non-random splitting has a larger training roc-auc and a significant decrease in testing roc-auc (suggesting overfitting). Thus, the performance of the RNN is better with random splitting of the data.