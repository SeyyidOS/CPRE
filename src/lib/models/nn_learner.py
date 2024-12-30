from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import optuna


class NNLearner:
    def __init__(self, model, dataset_class, params=None, log_dir="../runs/nn"):
        self.model = model
        self.params = params if params else {
            'learning_rate': 0.0001,
            'batch_size': 512,
            'num_epochs': 20
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.log_dir = log_dir
        self.dataset_class = dataset_class

    def train(self, X_train, y_train, X_val, y_val):
        writer = SummaryWriter(self.log_dir)

        X_train["Rating"] = y_train.values
        X_val["Rating"] = y_val.values        

        train_dataset = self.dataset_class(X_train, add_empty_channel=True)
        val_dataset = self.dataset_class(X_val, add_empty_channel=True)
        train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

        best_loss = float('inf')
        for epoch in range(self.params['num_epochs']):
            self.model.train()
            epoch_loss = 0

            for fen_input, move_input, target in tqdm(train_loader, desc=f"Epoch: {epoch+1}"):
                fen_input = fen_input.permute(0, 3, 1, 2).to(self.device)  # Change to NCHW format
                move_input = move_input.to(self.device)
                target = target.unsqueeze(1).to(self.device)  # Ensure target is 2D for MSELoss

                self.optimizer.zero_grad()
                outputs = self.model(fen_input, move_input)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() 

            writer.add_scalar("Loss/Train", epoch_loss / len(train_loader), epoch)

            val_loss = self._evaluate(val_loader)
            writer.add_scalar("Loss/Validation", val_loss, epoch)

            print(f"Epoch {epoch+1}/{self.params['num_epochs']} - Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model(f"../results/models/cnn/best_model_{val_loss:.4f}")
                print(f"Best model saved with loss: {val_loss:.4f}")

        writer.close()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def evaluate(self, X_val, y_val):
        predictions = self.predict(X_val)
        mse = mean_squared_error(y_val, predictions)
        print(f"Validation MSE: {mse:.4f}")
        return mse

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)

    def optimize(self, X_train, y_train, X_val, y_val, n_trials=50, db_path="sqlite:///nn_optuna.db"):
        def objective(trial):
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            num_epochs = trial.suggest_int("num_epochs", 50, 200)

            # Temporary model instance
            model = self.model.__class__()
            model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # Convert data to DataLoader
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            for epoch in range(num_epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            val_loss = self._evaluate(val_loader, model=model)
            return val_loss

        study = optuna.create_study(direction="minimize", study_name="nn_optimization", storage=db_path, load_if_exists=True)
        study.optimize(objective, n_trials=n_trials)

        print("Best hyperparameters:", study.best_params)
        self.params.update(study.best_params)
        return study.best_params

    def _evaluate(self, data_loader, model=None):
        if model is None:
            model = self.model
        model.eval()
        total_loss = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for fen_input, move_input, target in tqdm(data_loader):
                fen_input = fen_input.permute(0, 3, 1, 2).to(self.device)  # Change to NCHW format
                move_input = move_input.to(self.device)
                target = target.unsqueeze(1).to(self.device)
                outputs = model(fen_input, move_input)
                loss = criterion(outputs, target)
                total_loss += loss.item()
        return total_loss / len(data_loader)



class ChessPuzzleRatingModel(nn.Module):
    def __init__(self, move_vocab_size=80, move_embedding_dim=16, rnn_hidden_dim=32):
        super(ChessPuzzleRatingModel, self).__init__()

        self.fen_conv1 = nn.Conv2d(13, 32, kernel_size=3, stride=1, padding=1)
        self.fen_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fen_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.move_embedding = nn.Embedding(move_vocab_size, move_embedding_dim)
        self.move_gru = nn.GRU(move_embedding_dim, rnn_hidden_dim, batch_first=True)

        self.fc1 = nn.Linear(64 * 4 * 4 + rnn_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  

    def forward(self, fen_input, move_input):
        x = F.relu(self.fen_conv1(fen_input))
        x = self.fen_pool(F.relu(self.fen_conv2(x)))
        x = x.reshape(x.size(0), -1)  

        move_embedded = self.move_embedding(move_input)
        move_embedded = move_embedded[:, :30, :] 
        _, move_hidden = self.move_gru(move_embedded)
        move_hidden = move_hidden.squeeze(0)  

        combined = torch.cat((x, move_hidden), dim=1)
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        output = self.fc3(combined)

        return output
