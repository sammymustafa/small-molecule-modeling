from setup import * 
from dgllife.utils.splitters import ScaffoldSplitter
import dgl

# Try different values
HIDDEN_DIM = 100
NUM_STEPS = 4
DROPOUT = 0
EPOCHS = 20


class GNNLayer(torch.nn.Module):

    def __init__(self, dim, dropout):
        super().__init__()
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim * 3, dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim)
        )
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim)
        )
    def message(self, edges):
        node_src = edges.src['h']
        node_dst = edges.dst['h']
        edge = edges.data['e']
        msg = self.message_mlp(torch.cat([node_src, node_dst, edge], dim=-1))
        return {'msg_h': msg}

    def forward(self, graph, nodes, edges):
        with graph.local_scope():
            # node feature
            graph.ndata['h'] = nodes

            # edge feature
            graph.edata['e'] = edges

            # Compute messages
            graph.apply_edges(self.message)
            graph.update_all(dgl.function.copy_e('msg_h', 'm'), dgl.function.sum('m', 'h_neigh'))
            h_neigh = graph.ndata['h_neigh']

            # Compute node updates
            h_new = self.node_mlp(torch.cat([nodes, h_neigh], dim=-1))
            return h_new


class GNN(torch.nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.node_fc = torch.nn.Linear(74, dim)
        self.edge_fc = torch.nn.Linear(12, dim)
        self.gnn = GNNLayer(dim, dropout)
        self.fc = torch.nn.Linear(dim, 12)

    def forward(self, batch):
        g = batch["graph"]
        nodes = self.node_fc(g.ndata["h"])
        edges = self.edge_fc(g.edata["e"])

        for i in range(NUM_STEPS):
            # Add a residual connection with helps with stability
            nodes_new = self.gnn(g, nodes, edges)
            nodes = nodes + torch.nn.functional.relu(nodes_new)

        g.ndata["h_out"] = nodes
        out = dgl.mean_nodes(g, "h_out")
        out = self.fc(out)
        return out

def process_sample(sample):
    return {"graph": sample[1], "labels": sample[2], "mask": sample[3]}

def create_dataset():
    train, _, test = ScaffoldSplitter.train_val_test_split(Tox21, frac_val=0, frac_test=0.2)
    train = list(map(process_sample, train))
    test = list(map(process_sample, test))
    return train, test

def create_model():
    return GNN(HIDDEN_DIM, DROPOUT)

def collate_fn(data):
    graph = dgl.batch([d["graph"] for d in data])
    labels = torch.stack([d["labels"] for d in data])
    mask = torch.stack([d["mask"] for d in data])
    return {"graph": graph, "labels": labels, "mask": mask}


train_set, test_set = create_dataset()
model = create_model()
train_dl = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_dl = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
train_loss, train_aucs, test_aucs = train(model, train_dl, test_dl, num_epochs=EPOCHS)

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
# With the models ran on the random and non-random split datasets and the hidden dimensions and dropout being set at 100 and 0 respectively, it seems like the performance of the GNN model with this non-random splitting of the data decreases: the non-random splitting has a larger training roc-auc and a significant decrease in testing roc-auc (suggesting overfitting). Thus, the performance of the GNN is better with random splitting of the data.