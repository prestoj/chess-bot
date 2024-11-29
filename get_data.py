import chess
from chess_engine import SimpleChessEngine

def get_flat_board(board):
    """
    Convert a board position into a flat list of 64 integers.
    Board is oriented so current player's pieces start at bottom.
    Returns pieces relative to current player:
    0 = empty
    1-6 = my pieces (PNBRQK)
    7-12 = opponent pieces (PNBRQK)
    """
    flat_board = []
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                  chess.ROOK, chess.QUEEN, chess.KING]

    # For each rank and file
    ranks = range(8) if board.turn else range(7, -1, -1)
    files = range(8) if board.turn else range(7, -1, -1)
    
    for rank in ranks:
        for file in files:
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is None:
                flat_board.append(0)
            else:
                # Get piece type index (0-5 for PNBRQK)
                piece_idx = piece_types.index(piece.piece_type)
                # If it's my piece, add 1 to get values 1-6
                # If opponent's piece, add 7 to get values 7-12
                if piece.color == board.turn:
                    flat_board.append(piece_idx + 1)
                else:
                    flat_board.append(piece_idx + 7)
    
    return flat_board

import chess

def get_move_indices(board, move):
    """
    Convert a chess move (UCI format) to indices in the flattened board representation.
    Takes into account the board rotation when it's black's turn.
    
    Args:
        board: chess.Board object representing current position
        move: string in UCI format (e.g. 'e2e4') or chess.Move object
    
    Returns:
        tuple: (from_index, to_index) representing indices in the flattened board
    """
    # Convert move to UCI format if it's a Move object
    if isinstance(move, chess.Move):
        move = move.uci()
    
    # Get from and to squares
    from_square = chess.parse_square(move[:2])
    to_square = chess.parse_square(move[2:4])
    
    # Get rank and file for both squares
    from_rank = chess.square_rank(from_square)
    from_file = chess.square_file(from_square)
    to_rank = chess.square_rank(to_square)
    to_file = chess.square_file(to_square)
    
    if not board.turn:  # If it's black's turn, we need to flip coordinates
        # Flip ranks and files (7 - coordinate to invert)
        from_rank = 7 - from_rank
        from_file = 7 - from_file
        to_rank = 7 - to_rank
        to_file = 7 - to_file
    
    # Calculate indices in flattened board
    from_index = from_rank * 8 + from_file
    to_index = to_rank * 8 + to_file
    
    return from_index, to_index

def generate_game_positions(engine, depth=1):
    """
    Generates a chess game using the SimpleChessEngine for both sides
    Returns a list of tuples: (flat_board, move_made, from_idx, to_idx)
    """
    board = chess.Board()
    positions = []
    moves_without_capture = 0
    
    while True:
        # Get engine's move
        move = engine.get_best_move(board, depth=depth)
        
        if move is None:  # No legal moves available
            break
            
        # Get move indices before making the move
        from_idx, to_idx = get_move_indices(board, move)

        legal_moves = []
        for legal_move in board.legal_moves:
            legal_move_from_idx, legal_move_to_idx = get_move_indices(board, legal_move)
            legal_moves.append((legal_move_from_idx, legal_move_to_idx))
        
        # Store position, move, and indices before making it
        flat_board = get_flat_board(board)
        positions.append((flat_board, move.uci(), from_idx, to_idx, legal_moves))
        
        # Make the move
        board.push(move)
        
        # Check for game end conditions
        if board.is_game_over():
            break
            
        # Track moves without captures for simple draw condition
        if board.is_capture(move):
            moves_without_capture = 0
        else:
            moves_without_capture += 1
            if moves_without_capture > 50:  # Simple draw condition
                break
    
    return positions
