from tools.packages import *

class ChatBotModule(pl.LightningModule):
  def __init__(self, input_size, hidden_size, num_classes, lr = 1e-3):
    super().__init__()
    self.l1 = nn.Linear(input_size, hidden_size) 
    self.l2 = nn.Linear(hidden_size, hidden_size) 
    self.l3 = nn.Linear(hidden_size, num_classes)
    self.relu = nn.ReLU()

    self.accuracy = torchmetrics.Accuracy()
    self.lr = lr
    self.loss = nn.CrossEntropyLoss()
    self.save_hyperparameters()
    
  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    out = self.relu(out)
    out = self.l3(out)
    return out

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss(y_hat, y)
    self.log('train_loss', loss)
    self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)

class ChatDataset(Dataset):

    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


