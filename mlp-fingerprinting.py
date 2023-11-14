from setup import * 

# Try different values
RADIUS = 3
HIDDEN_DIM = 100
DROPOUT = 0
NUM_EPOCHS = 10
FP_SIZE = 2048


class MLP(torch.nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.w2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, batch):
        data = batch["data"]
        data = batch["data"]
        x = self.w1(data)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


def process_sample(sample):
    smiles = sample[0]
    labels = sample[2]
    mask = sample[3]
    fpgen = AllChem.GetMorganGenerator(radius=RADIUS, fpSize=FP_SIZE)
    mol = Chem.MolFromSmiles(smiles)
    ao = AllChem.AdditionalOutput()
    ao.CollectBitInfoMap()
    fp = fpgen.GetCountFingerprint(mol, additionalOutput=ao)
    arr = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    arr = torch.tensor(arr).float()
    return {"data": arr, "labels": labels, "mask": mask}

def create_dataset():
    dataset = list(map(process_sample, Tox21))
    train, test = train_test_split(dataset, test_size=0.2, random_state=1)
    return train, test

def create_model():
    model = MLP(FP_SIZE, HIDDEN_DIM, 12)
    return model

def evaluate(model, dataloader):
    out_pred, out_labels, out_mask = [], [], []
    for batch in dataloader:
        mask = batch["mask"]
        labels = batch["labels"]
        y_pred = model(batch).sigmoid()
        out_pred.append(y_pred)
        out_labels.append(labels)
        out_mask.append(mask)

    out_pred = torch.cat(out_pred).detach().numpy()
    out_labels = torch.cat(out_labels).detach().numpy()
    out_mask = torch.cat(out_mask).bool().detach().numpy()

    aucs = []
    for i in range(12):
        preds = out_pred[:, i]
        labels = out_labels[:, i]
        mask = out_mask[:, i]
        preds = preds[mask]
        labels = labels[mask]
        aucs.append(roc_auc_score(labels, preds))

    return np.mean(aucs)

def train(model, train_dataloader, test_dataloader, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss = []
    train_aucs = []
    test_aucs = []
    for _ in tqdm(range(num_epochs), total=num_epochs):
        avg_loss = 0
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            mask = batch["mask"]
            labels = batch["labels"]
            y_pred = model(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, labels, reduction="none")
            loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        model.eval()
        with torch.no_grad():
            train_auc = evaluate(model, train_dataloader)
            test_auc = evaluate(model, test_dataloader)

        avg_loss /= len(train_dataloader)
        train_loss.append(avg_loss)
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)

    return train_loss, train_aucs, test_aucs

train_set, test_set = create_dataset()
train_dl = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
model = create_model()
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
# It looks like greater radius leads to higher training roc-auc and lower testing roc-auc so it seems like higher radius leads to overfitting of the data on the training data. I personally would use lower radius values just for the model to perform better outside the training data.
# Greater hidden dimensions also genrally leads to higher training roc-auc and higher testing roc-auc. This means that considering more hidden dimensions leads to better fitting of the model on the training dataset which also translates to higher accurracy on the testing data.
# With more epochs may come overfitting of the model to the training dataset and thus decreased roc-auc for the testing data. This is because the model is becoming more and more fit with the training data with each epoch. However, increasing the dropout prevents this as more epochs are needed to get to the optimal/highest roc-auc for the testing dataset. While these increases in dropout generally increases the roc-auc of the testing dataset in small increments (with a decrease in the roc-auc of the training dataset), it is interesting because this is most likley more computationally intensive as we are running more epochs to get to the highest roc-auc for the testing dataset rather than identifying the least amount needed to get to this optimal value. However, this increase in dropout comes with a decrease in training roc-auc so this may just be preventing overfitting in instances when a set number of epochs are used.