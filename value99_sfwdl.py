"""Official Stockfish 19 WDL → Value99 side-to-move targets.

`avewright/local-wdl` stores White-absolute UCI_ShowWDL. Value99 trains
P(win)-P(loss) after canonicalizing the side to move as White. Mirroring
the board does not change that STM target.
"""
from pathlib import Path
import chess
import numpy as np
from chess_value99 import pack_board

REPO='avewright/local-wdl'
REVISION='c6626fee143c8679354c14e37e9a9e35a4bb8c91'
PIECE_SYMBOL={1:'P',2:'N',3:'B',4:'R',5:'Q',6:'K',7:'p',8:'n',9:'b',10:'r',11:'q',12:'k'}
CASTLE_BITS=((8,'K'),(4,'Q'),(2,'k'),(1,'q'))


def board_array_to_fen(board_array,turn,castling,ep_square):
    ranks=[]
    for rank in range(7,-1,-1):
        empty=0;cells=[]
        for file in range(8):
            pid=int(board_array[rank*8+file])
            if pid<=0:
                empty+=1;continue
            if empty:
                cells.append(str(empty));empty=0
            cells.append(PIECE_SYMBOL[pid])
        if empty:cells.append(str(empty))
        ranks.append(''.join(cells))
    castle=''.join(ch for bit,ch in CASTLE_BITS if int(castling)&bit) or '-'
    ep='-'
    if 0<=int(ep_square)<=63:ep=chess.square_name(int(ep_square))
    stm='b' if int(turn) else 'w'
    return f"{'/'.join(ranks)} {stm} {castle} {ep} 0 1"


def stm_target(wdl,turn):
    """White-absolute [Ww,D,Wl] → side-to-move P(win)-P(loss)."""
    w=np.asarray(wdl,dtype=np.float64)
    if w.shape!=(3,) or not np.isfinite(w).all() or (w<0).any():
        return None
    total=float(w.sum())
    if abs(total-1)>.02:return None
    w=w/total
    score=float(w[0]-w[2])
    return -score if int(turn) else score


def phase_from_array(board_array):
    arr=np.asarray(board_array).reshape(64)
    pieces=int(np.count_nonzero(arr))
    pawns=int(np.isin(arr,(1,7)).sum())
    return 2 if pieces-pawns<=4 else (0 if pieces>=28 else 1)


def pack_from_arrays(board_array,turn,castling,ep_square):
    """70-byte Value99 pack. Clocks unknown. Mirrors Black-to-move like pack_board."""
    arr=np.asarray(board_array,dtype=np.uint8).reshape(64).copy()
    c=int(castling);ep=int(ep_square)
    wk,wq,bk,bq=bool(c&8),bool(c&4),bool(c&2),bool(c&1)
    if int(turn):
        arr=arr.reshape(8,8)[::-1].reshape(64)
        swapped=np.zeros(64,dtype=np.uint8)
        white=(arr>=1)&(arr<=6);black=arr>=7
        swapped[white]=arr[white]+6;swapped[black]=arr[black]-6
        arr=swapped;wk,wq,bk,bq=bk,bq,wk,wq
        if 0<=ep<=63:ep=(7-ep//8)*8+(ep%8)
    out=np.zeros(70,dtype=np.uint8);out[:64]=arr
    out[64:68]=[wk,wq,bk,bq]
    out[68]=64 if not (0<=ep<=63) else ep
    return out


def phase_of(board):
    pieces=len(board.piece_map())
    nonpawn=sum(len(board.pieces(p,color)) for p in (2,3,4,5) for color in (False,True))
    return 2 if nonpawn<=4 else (0 if pieces>=28 else 1)


def convert_row(row):
    """Accept one local-wdl record. Honor split=1 as the only eval pool."""
    if int(row.get('wdl_source',0))!=1:return None,'not_official_wdl'
    target=stm_target(row.get('wdl'),row.get('turn'))
    if target is None:return None,'invalid_wdl'
    try:
        board=chess.Board(board_array_to_fen(row['board_array'],row['turn'],row['castling'],row['ep_square']))
    except ValueError:
        return None,'invalid_fen'
    if not board.is_valid():return None,'nonstandard_or_invalid'
    if board.is_game_over(claim_draw=False):return None,'terminal'
    if bool(board.turn)==bool(int(row['turn'])):
        return None,'turn_mismatch'
    return dict(packed=pack_from_arrays(row['board_array'],row['turn'],row['castling'],row['ep_square']),
                target=float(target),phase=phase_of(board),
                black=int(not board.turn),split=int(row.get('split',0)),
                nodes=int(row.get('nodes',0)),label_depth=int(row.get('label_depth',0)),
                n_pieces=int(row.get('n_pieces',len(board.piece_map()))),
                game_id=int(row.get('game_id',-1)),
                white_score=float(np.asarray(row['wdl'],dtype=np.float64)[0]
                                  -np.asarray(row['wdl'],dtype=np.float64)[2])),None


def convert_train_row(row):
    """Train-only: drop holdout and unofficial WDL. No chess.Board."""
    if int(row.get('split',0))==1:return None,'holdout'
    if int(row.get('wdl_source',1))!=1:return None,'not_official_wdl'
    target=stm_target(row.get('wdl'),row.get('turn'))
    if target is None:return None,'invalid_wdl'
    packed=pack_from_arrays(row['board_array'],row['turn'],row['castling'],row['ep_square'])
    return dict(packed=packed,target=float(target),phase=phase_from_array(row['board_array'])),None


def load_teacher(root):
    root=Path(root)
    packed=np.load(root/'packed.npy',mmap_mode='c')
    target=np.load(root/'target.npy',mmap_mode='c')
    phase=np.load(root/'phase.npy',mmap_mode='c')
    black=np.load(root/'black.npy',mmap_mode='c') if (root/'black.npy').exists() else None
    return packed,target,phase,black
