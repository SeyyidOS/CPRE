
from src.feature.cnn.move_encoders import encode_moves_64_squares
from src.feature.cnn.one_hot import fen_to_one_hot
from torch.utils.data import Dataset

import torch


class ChessPuzzleDataset(Dataset):
    def __init__(self, data, add_empty_channel=False, max_len=30):
        self.data = data
        self.add_empty_channel = add_empty_channel
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row['FEN']
        moves = row['Moves']
        rating = row['Rating']

        fen_one_hot = fen_to_one_hot(fen, add_empty_channel=self.add_empty_channel)
        encoded_moves = encode_moves_64_squares(moves, max_len=self.max_len)

        return torch.tensor(fen_one_hot, dtype=torch.float32), torch.tensor(encoded_moves, dtype=torch.long), torch.tensor(rating, dtype=torch.float32)
