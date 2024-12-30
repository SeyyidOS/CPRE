import matplotlib.pyplot as plt
import numpy as np

piece_to_channel = {
    'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5,  # White pieces
    'p': 6, 'r': 7, 'n': 8, 'b': 9, 'q': 10, 'k': 11  # Black pieces
}

def visualize_one_hot_encoding(one_hot, fen):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(f"One-Hot Encoding Visualization for FEN: {fen}", fontsize=14)
    channel_names = [
        "White Pawn", "White Rook", "White Knight", "White Bishop",
        "White Queen", "White King", "Black Pawn", "Black Rook",
        "Black Knight", "Black Bishop", "Black Queen", "Black King", "Empty Squares"
    ]
    
    num_channels = one_hot.shape[2]
    for i in range(num_channels):
        ax = axes[i // 4, i % 4]
        ax.imshow(one_hot[:, :, i], cmap='gray', aspect='auto')
        ax.set_title(channel_names[i] if i < len(channel_names) else f"Channel {i}")
        ax.axis('off')
    
    for i in range(num_channels, 16):
        axes[i // 4, i % 4].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    output_path = "../results/cnn/one_hot.png"
    plt.savefig(output_path)
    print(f"Chessboard one-hot encoded image saved as {output_path}")
    
    plt.close()
    


def fen_to_one_hot(fen, add_empty_channel=False, save=False, verbose=False):
    board_part = fen.split()[0]
    rows = board_part.split('/')
    
    num_channels = 12 + (1 if add_empty_channel else 0)
    one_hot = np.zeros((8, 8, num_channels), dtype=np.uint8)
    
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                channel = piece_to_channel[char]
                one_hot[i, col, channel] = 1
                col += 1
    
    if add_empty_channel:
        one_hot[:, :, -1] = (one_hot.sum(axis=2) == 0).astype(np.uint8)
    
    if verbose:
        print("One-hot encoding shape:", one_hot.shape)
    
    if save:
        visualize_one_hot_encoding(one_hot, fen)
                
    return one_hot





