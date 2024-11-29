import chess
import os
import random
from multiprocessing import Pool, cpu_count
from functools import partial

class SimpleChessEngine:
    def __init__(self):
        # Simplified piece values like the second implementation
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Keep basic Unicode pieces for display
        self.unicode_pieces = {
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
            '.': ' '
        }

    def evaluate_position(self, board):
        """Simple material count evaluation like the second implementation"""
        score = 0
        pieces = board.piece_map()
        
        for square, piece in pieces.items():
            value = self.piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score -= value  # Negative for white like second implementation
            else:
                score += value  # Positive for black like second implementation
        
        return score

    def minimax(self, board, depth, maximizing_player):
        """Simplified minimax that collects equally good moves"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board), []

        best_moves = []
        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, False)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_moves = [move]
                elif eval_score == max_eval:
                    best_moves.append(move)
            return max_eval, best_moves
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, True)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_moves = [move]
                elif eval_score == min_eval:
                    best_moves.append(move)
            return min_eval, best_moves

    def get_best_move(self, board, depth):
        is_maximizing = board.turn == chess.BLACK  # Black is maximizing like second implementation
        _, best_moves = self.minimax(board, depth, is_maximizing)
        return random.choice(best_moves) if best_moves else None

    def print_board(self, board):
        """Simplified board printing"""
        print("\n  a b c d e f g h")
        for rank in range(8):
            print(f"{8-rank}", end=" ")
            for file in range(8):
                square = chess.square(file, 7-rank)
                piece = board.piece_at(square)
                if piece is None:
                    print(".", end=" ")
                else:
                    print(piece.symbol(), end=" ")
            print()
        print()

def play_game():
    board = chess.Board()
    engine = SimpleChessEngine()
    move_history = []  # Store tuples of (move_object, uci_string)
    last_move = None  # Store the last chess.Move object
    
    while not board.is_game_over():
        # Print move history
        if move_history:
            print("\nMove History:")
            for i, (_, uci) in enumerate(move_history):
                if i % 2 == 0:
                    print(f"{(i//2)+1}. {uci}", end="")
                else:
                    print(f" ...{uci}")
            if len(move_history) % 2 != 0:
                print()

        engine.print_board(board)
        
        if board.turn == chess.WHITE:
            # Human plays white
            print("\nYour turn (White)")
            move_uci = input("Enter your move (e.g., e2e4) or 'help' for legal moves: ")
            
            if move_uci.lower() == 'help':
                print("\nLegal moves:", ' '.join(move.uci() for move in board.legal_moves))
                input("Press Enter to continue...")
                continue
                
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                    move_history.append((move, move_uci))
                    last_move = move
                else:
                    print("Illegal move! Try again.")
                    continue
            except ValueError:
                print("Invalid input! Try again.")
                continue
        else:
            # Engine plays black
            print("\nEngine is thinking...")
            move = engine.get_best_move(board, depth=3)
            board.push(move)
            move_history.append((move, move.uci()))
            last_move = move
            print(f"Engine plays: {move.uci()}")

    # Game over
    engine.print_board(board)
    print("\nGame Over!")
    print(f"Result: {board.outcome().result()}")

if __name__ == "__main__":
    play_game()