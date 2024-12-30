import torch

char_to_index = {sq: i for i, sq in enumerate("abcdefgh12345678qrbnk")}


square_to_index = {f"{file}{rank}": i for i, (rank, file) in enumerate([(r, f) for r in "87654321" for f in "abcdefgh"])}

def encode_moves_with_char(moves_str, max_len=10, verbose=False):
    moves = moves_str.split()
    encoded = [[char_to_index[c] for c in move] for move in moves]

    padded_encoded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq)[:max_len] for seq in encoded],  
        batch_first=True, padding_value=0
    )
    
    if padded_encoded.size(1) < max_len:
        padded_encoded = torch.nn.functional.pad(
            padded_encoded, (0, max_len - padded_encoded.size(1)), value=0
        )
    
    if verbose:
        print("Encoded Moves (Character-based):\n", padded_encoded)
        
    return padded_encoded

def encode_moves_64_squares(moves_str, max_len=10, verbose=False):
    moves = moves_str.split()
    
    encoded = []
    for move in moves:
        start = move[:2]
        end = move[2:4]
        promotion = move[4:]  # Optional promotion character (e.g., 'q', 'r', 'b', 'n')
        
        if start in square_to_index and end in square_to_index:
            encoded.append((square_to_index[start], square_to_index[end]))
            if promotion:
                encoded.append((64 + "qrbnk".index(promotion.lower()),))
    
    flattened = [item for pair in encoded for item in pair]
    encoded_tensor = torch.tensor(flattened[:max_len])
    
    padded_encoded = torch.nn.functional.pad(
        encoded_tensor, (0, max_len - len(encoded_tensor)), value=0
    )

    if verbose:
        print(f"Original Moves: {moves_str}")
        print(f"Encoded (start, end, promotion): {encoded}")
        print(f"Padded Sequence: {padded_encoded.tolist()}")

    return padded_encoded
