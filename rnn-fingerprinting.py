from setup import * 

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

def create_dataset():
    max_length = max(len(x[0]) for x in Tox21)
    dataset = list(map(lambda x: process_sample(x, max_length), Tox21))
    train, test = train_test_split(dataset, test_size=0.2, random_state=1)
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
# Comparing the RNN model with the MLP baseline with 100 hidden dimensions and 0 dropout, the RNN model had a lower training roc-auc but higher testing roc-auc. This suggests that the MLP model may be overfitting.
# While more hidden parameters seem to improve the roc_auc of both the training and testing data for the MLP, this RNN/LSTM showed overfitting in which there was a greater training roc-auc and lower testing roc-auc (not very significant). However, the RNN/LSTM model differs from the MLP model in which greater dropout values decreases both the training and testing roc-auc (compared to decreasing training and increasing testing roc-auc, less overfitting) but they both do result in lower instances in which the roc-auc drops with more epochs. Thus, improving the RNN model's testing roc-auc comes with a good mixture of a lesser number of hidden parameters and lower dropout values.