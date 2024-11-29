import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader
from get_data import generate_game_positions
from chess_engine import SimpleChessEngine
import time
from tqdm import tqdm
import chess
from get_data import get_flat_board

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, n_layers=2, d_model=32, nhead=2, dropout=0.1):
        super().__init__()
        
        # Input embedding: from piece encoding (0-12) to d_model dimensions
        self.embedding = nn.Embedding(13, d_model)  # 13 because 0-12 inclusive
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                 nhead=nhead, 
                                                 dropout=dropout,
                                                 batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output layer: predict move probability for each square
        self.output = nn.Linear(d_model * 64, 4096)  # 1 score per position

    def forward(self, src):
        # src shape: (batch_size, 64)
        
        # Embed pieces
        x = self.embedding(src)  # -> (batch_size, 64, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Run through transformer
        x = self.transformer(x)  # -> (batch_size, 64, d_model)
        
        # Get move probabilities for each square
        x = x.reshape(x.size(0), -1) # -> (batch_size, 4096)
        x = self.output(x)  # -> (batch_size, 4096)
        
        return x
    
class ChessDataset(Dataset):
    def __init__(self, game_positions):
        """
        Initialize the dataset with game positions.
        
        Args:
            game_positions: List of tuples (flat_board, move, from_idx, to_idx)
        """
        self.positions = []
        self.move_targets = []
        self.legal_moves = []
        
        for board, _, from_idx, to_idx, legal_moves in game_positions:
            # Convert board to tensor
            board_tensor = torch.tensor(board, dtype=torch.long)
            
            # Convert from_idx and to_idx to single target index
            target_idx = from_idx * 64 + to_idx
            
            self.positions.append(board_tensor)
            self.move_targets.append(target_idx)

            legal_moves_tensor = torch.zeros(4096)

            for from_idx, to_idx in legal_moves:
                legal_moves_tensor[from_idx * 64 + to_idx] = 1
            
            self.legal_moves.append(legal_moves_tensor)
            
        # Stack all positions into a single tensor
        self.positions = torch.stack(self.positions)
        self.move_targets = torch.tensor(self.move_targets, dtype=torch.long)
        self.legal_moves = torch.stack(self.legal_moves)
        
    def __len__(self):
        return len(self.positions)
        
    def __getitem__(self, idx):
        return self.positions[idx], self.move_targets[idx], self.legal_moves[idx]

def create_data_loader(game_positions, batch_size=32, shuffle=True):
    """
    Create a DataLoader from game positions.
    
    Args:
        game_positions: List of tuples (flat_board, move, from_idx, to_idx)
        batch_size: Number of positions per batch
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader object
    """
    dataset = ChessDataset(game_positions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(model, num_epochs=1000, batch_size=64, learning_rate=1e-3):
    """
    Train the transformer model on continuously generated chess games.
    Each epoch is one full game.
    
    Args:
        model: The transformer model
        num_epochs: Number of games to train on
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model = model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Initialize engine
    engine = SimpleChessEngine()

    depth = 1
    
    # Training metrics
    running_loss = 0.0
    moves_processed = 0
    
    # Main training loop
    epoch = 0
    while True:
        epoch += 1
        epoch_start = time.time()
        
        # Generate 10 new game
        all_game_positions = []
        for _ in range(10):
            game_positions = generate_game_positions(engine, depth=depth)
            all_game_positions.extend(game_positions)
        
        # Create data loader for these games
        loader = create_data_loader(all_game_positions, batch_size=batch_size, shuffle=True)
        
        # Process each batch in the games
        batch_losses = []
        batch_accuracies = []
        for boards, targets, legal_moves in loader:
            # Move data to device
            boards = boards.to(device)
            targets = targets.to(device)
            legal_moves = legal_moves.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, legal_moves)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record metrics
            batch_losses.append(loss.item())
            running_loss += loss.item()
            moves_processed += boards.size(0)
            batch_accuracies.append(((torch.sigmoid(outputs) > 0.5) == legal_moves).to(torch.float).mean() * 100)
        
        # Calculate epoch statistics
        epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        epoch_acc = sum(batch_accuracies) / len(batch_accuracies)
        epoch_time = time.time() - epoch_start
        moves_per_second = len(all_game_positions) / epoch_time
        
        # Print progress every epoch
        print(f"\nEpoch {epoch+1}")
        print(f"Game length: {len(all_game_positions)} moves")
        print(f"Average loss: {epoch_loss:.4f}")
        print(f"Average acc: {epoch_acc:.4f}%")
        print(f"Moves per second: {moves_per_second:.2f}")
        print(f"Depth: {depth}")

        # if epoch_loss < 3:
        #     depth += 1

if __name__ == "__main__":
    # Create model
    model = Transformer(
        n_layers=6,
        d_model=384,
        nhead=6,
        dropout=0.1
    )
    
    # Start training
    try:
        train_model(model)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save the model
    torch.save(model.state_dict(), 'chess_transformer.pth')