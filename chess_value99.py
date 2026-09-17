"""Fresh ~99M value-only transformer: 24 independent layers, 64 board squares.

V(s) is from the side-to-move perspective, [-1,1]. A move is scored as -V(child).
No policy head, recurrent weight sharing, pretrained weights, or move vocabulary.
"""
from dataclasses import asdict,dataclass
import chess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ValueConfig:
    width:int=576
    layers:int=24
    heads:int=8
    ffn_inner:int=1584
    value_square:int=128
    value_hidden:int=128
    dropout:float=.05
    gradient_checkpointing:bool=False


def canonical_board(board):
    # Mirror ranks AND colors so the side to move always appears as White.
    return board.copy(stack=False) if board.turn else board.mirror()


def pack_board(board):
    """70 bytes: 64 piece IDs, four rights, EP square/64, halfmove clock/100."""
    b=canonical_board(board);out=np.zeros(70,dtype=np.uint8)
    for sq,p in b.piece_map().items():out[sq]=p.piece_type+(0 if p.color else 6)
    out[64:68]=[b.has_kingside_castling_rights(True),b.has_queenside_castling_rights(True),
                b.has_kingside_castling_rights(False),b.has_queenside_castling_rights(False)]
    out[68]=64 if b.ep_square is None else b.ep_square
    out[69]=min(b.halfmove_clock,100)
    return out


def unpack_board(packed):
    b=chess.Board(None)
    for sq,idx in enumerate(packed[:64]):
        idx=int(idx)
        if idx:b.set_piece_at(sq,chess.Piece((idx-1)%6+1,idx<=6))
    b.turn=True
    for value,sq in zip(packed[64:68],[chess.H1,chess.A1,chess.H8,chess.A8]):
        if value:b.castling_rights|=chess.BB_SQUARES[sq]
    b.ep_square=None if int(packed[68])==64 else int(packed[68]);b.halfmove_clock=int(packed[69])
    return b


def planes(packed):
    ids=packed[:,:64].long()
    pieces=F.one_hot(ids,13)[:,:,1:].float()
    rights=packed[:,None,64:68].float().expand(-1,64,-1)
    ep=(torch.arange(64,device=packed.device)[None,:]==packed[:,68:69]).float().unsqueeze(-1)
    clock=(packed[:,69:70].float()/100)[:,None,:].expand(-1,64,-1)
    return torch.cat([pieces,rights,ep,clock],-1)


class Block(nn.Module):
    def __init__(self,c):
        super().__init__();d=c.width;self.heads=c.heads;self.dim=d//c.heads;self.dropout=c.dropout
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d,bias=False);self.out=nn.Linear(d,d)
        self.q_norm=nn.RMSNorm(self.dim);self.k_norm=nn.RMSNorm(self.dim)
        self.relative_bias=nn.Embedding(225,c.heads)
        self.gate=nn.Linear(d,c.ffn_inner);self.up=nn.Linear(d,c.ffn_inner);self.down=nn.Linear(c.ffn_inner,d)
        nn.init.zeros_(self.relative_bias.weight)

    def forward(self,x,relative_index):
        b,n,d=x.shape;y=self.norm1(x)
        q,k,v=self.qkv(y).reshape(b,n,3,self.heads,self.dim).permute(2,0,3,1,4).unbind(0)
        bias=self.relative_bias(relative_index).permute(2,0,1).unsqueeze(0).to(q.dtype)
        y=F.scaled_dot_product_attention(self.q_norm(q),self.k_norm(k),v,attn_mask=bias,
                 dropout_p=self.dropout if self.training else 0.)
        x=x+F.dropout(self.out(y.transpose(1,2).reshape(b,n,d)),self.dropout,self.training)
        y=self.norm2(x);y=self.down(F.silu(self.gate(y))*self.up(y))
        return x+F.dropout(y,self.dropout,self.training)


class ValueTransformer(nn.Module):
    def __init__(self,config=None):
        super().__init__();c=config or ValueConfig();self.config=c
        if c.width%c.heads:raise ValueError('Width must divide head count')
        self.input=nn.Linear(18,c.width);self.square=nn.Embedding(64,c.width);self.input_norm=nn.LayerNorm(c.width)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)]);self.norm=nn.LayerNorm(c.width)
        self.square_value=nn.Linear(c.width,c.value_square)
        self.decoder=nn.Sequential(nn.Linear(64*c.value_square,c.value_hidden),nn.GELU(),nn.Linear(c.value_hidden,1))
        sq=torch.arange(64);rank=sq//8;file=sq%8
        index=(rank[:,None]-rank[None,:]+7)*15+(file[:,None]-file[None,:]+7)
        self.register_buffer('relative_index',index,persistent=False)
        # Small residual projections, not zero: all layers can learn immediately.
        for block in self.blocks:
            with torch.no_grad():
                block.out.weight.div_((2*c.layers)**.5);block.down.weight.div_((2*c.layers)**.5)
        nn.init.normal_(self.decoder[-1].weight,std=.01);nn.init.zeros_(self.decoder[-1].bias)

    def forward(self,packed,return_logit=False):
        x=self.input_norm(self.input(planes(packed))+self.square.weight[None])
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x=checkpoint(block,x,self.relative_index,use_reentrant=False)
            else:x=block(x,self.relative_index)
        x=F.gelu(self.square_value(self.norm(x))).flatten(1)
        logit=self.decoder(x)[:,0].float()
        return logit if return_logit else 2*torch.sigmoid(logit)-1


@torch.no_grad()
def score_legal_moves(model,board,device='cpu',batch_size=64):
    """One-ply inference: score all legal options, resolve true terminals exactly."""
    model.eval();moves=list(board.legal_moves);scores={};pending=[]
    mover=board.turn
    for move in moves:
        board.push(move);out=board.outcome(claim_draw=True)
        if out:scores[move.uci()]=0. if out.winner is None else (1. if out.winner==mover else -1.)
        else:pending.append((move.uci(),pack_board(board)))
        board.pop()
    for start in range(0,len(pending),batch_size):
        chunk=pending[start:start+batch_size];x=torch.from_numpy(np.stack([p for _,p in chunk])).to(device)
        v=model(x).cpu().tolist()
        for (uci,_),value in zip(chunk,v):scores[uci]=-value
    return scores
