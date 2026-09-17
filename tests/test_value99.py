import json
import chess
import numpy as np
import torch
from chess_value99 import ValueConfig,ValueTransformer,pack_board,unpack_board,planes,score_legal_moves
from scripts.prepare_value99_data import convert
from experiments.value99_pretrain import load_split,digest


def test_parameter_budget():
    with torch.device('meta'):
        model=ValueTransformer()
    assert sum(p.numel() for p in model.parameters())==98920577
    assert len(model.blocks)==24
    assert model.blocks[0].qkv.weight is not model.blocks[1].qkv.weight


def test_encoding_color_symmetry_rule_state():
    board=chess.Board();board.push_uci('e2e4')
    packed=pack_board(board)
    np.testing.assert_array_equal(packed,pack_board(board.mirror()))
    np.testing.assert_array_equal(packed,pack_board(unpack_board(packed)))
    x=planes(torch.from_numpy(packed[None]))
    assert x.shape==(1,64,18)
    assert x[:,:,16].sum()==1
    assert x[:,:,:12].sum()==32


def test_targets_and_split_consistency():
    board=chess.Board()
    row,_=convert({'fen':board.fen(),'wdl':[.7,.2,.1]})
    mirrored,_=convert({'fen':board.mirror().fen(),'wdl':[.7,.2,.1]})
    board.halfmove_clock=20
    clock,_=convert({'fen':board.fen(),'wdl':[.7,.2,.1]})
    assert abs(row['target']-.6)<1e-6
    assert row['hash']==mirrored['hash']==clock['hash']
    assert row['split']==mirrored['split']==clock['split']
    assert convert({'fen':board.fen(),'wdl':[1,1,1]})[1]=='invalid_wdl'


def test_tiny_backward_checkpointing_and_reload():
    torch.set_num_threads(2)
    model=ValueTransformer(ValueConfig(width=32,layers=2,heads=4,ffn_inner=64,value_square=8,value_hidden=16,dropout=0,gradient_checkpointing=True))
    x=torch.from_numpy(np.stack([pack_board(chess.Board())]*2))
    loss=model(x).square().mean();loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval();before=model(x)
    assert before.shape==(2,) and (before.abs()<=1).all()
    restored=ValueTransformer(model.config);restored.load_state_dict(model.state_dict());restored.eval()
    torch.testing.assert_close(before,restored(x))


def test_oneply_negation_and_terminal_mate():
    class Constant(torch.nn.Module):
        def forward(self,x):return torch.full((len(x),),.4,device=x.device)
    board=chess.Board();fen=board.fen()
    scores=score_legal_moves(Constant(),board)
    assert len(scores)==20 and all(abs(v+.4)<1e-6 for v in scores.values())
    assert board.fen()==fen and not board.move_stack
    board=chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')
    assert score_legal_moves(Constant(),board)['f7g7']==1


def test_disk_backed_data(tmp_path):
    path=tmp_path/'train.npz'
    x=np.stack([pack_board(chess.Board())]*3)
    np.savez(path,packed=x,target=np.array([-.5,0,.5],dtype=np.float32),phase=np.zeros(3,dtype=np.uint8))
    manifest={'shards':[{'file':path.name,'rows':3,'split':'train','sha256':digest(path)}]}
    for _ in range(2):
        packed,target,phase=load_split(tmp_path,manifest,'train')
        np.testing.assert_array_equal(packed.numpy(),x)
        assert target.tolist()==[-.5,0,.5]
