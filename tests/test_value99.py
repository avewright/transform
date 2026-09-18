import json
import chess
import numpy as np
import torch
from chess_value99 import ValueConfig,ValueTransformer,pack_board,unpack_board,planes,score_legal_moves
from scripts.prepare_value99_data import convert
from experiments.value99_pretrain import digest,load_split,mix_n,resume_compatible,should_reload_sf,take_ids
from scripts.prepare_value99_sfwdl_train import normalize_done,source_key
from value99_valmix import build_valmix,cell_quotas,outcome_bucket,write_valmix,load_valmix
from value99_sfwdl import board_array_to_fen,convert_row,convert_train_row,pack_from_arrays,stm_target


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


def test_valmix_is_stratified_and_deterministic():
    phases=np.array([p for p in range(3) for _ in range(90)],dtype=np.uint8)
    targets=np.array([-0.8,-0.4,0.0,0.4,0.8,0.1]*45,dtype=np.float32)
    black=np.array([i%3==0 for i in range(len(targets))])
    assert len(targets)==len(phases)==270
    mix=build_valmix(targets,phases,90,seed=294,black=black)
    again=build_valmix(targets,phases,90,seed=294,black=black)
    np.testing.assert_array_equal(mix['indices'],again['indices'])
    assert sum(cell_quotas(90).values())==90
    assert mix['n']==90
    assert mix['shortfall']=={k:0 for k in mix['shortfall']}
    assert mix['filled']['middlegame_moderate']>=mix['filled']['opening_drawish']
    assert mix['black_source']>0
    # A missing opening-decisive cell is reported, not silently borrowed from train.
    thin_phases=np.array([1]*80+[2]*80,dtype=np.uint8)
    thin_targets=np.linspace(-1,1,160).astype(np.float32)
    thin=build_valmix(thin_targets,thin_phases,90,seed=1)
    assert thin['shortfall']['opening_drawish']>0
    assert thin['n']==90


def test_valmix_roundtrip(tmp_path):
    mix=build_valmix(np.array([0.0,0.7,-0.3],dtype=np.float32),np.array([0,1,2],dtype=np.uint8),3,seed=7)
    write_valmix(tmp_path,mix)
    loaded=load_valmix(tmp_path)
    np.testing.assert_array_equal(loaded['indices'],mix['indices'])
    assert loaded['n']==3


def test_outcome_buckets():
    assert outcome_bucket(0.0)==0
    assert outcome_bucket(0.19)==0
    assert outcome_bucket(0.2)==1
    assert outcome_bucket(0.59)==1
    assert outcome_bucket(-0.6)==2


def test_resume_allows_sfwdl_sprinkle():
    base=dict(parameters=1,data_manifest_sha256='abc',device='cuda',
              torch_version='1',initialization='random',optimizer={'name':'polar_normuon'},
              torch_compile=True,target='side-to-move W-L; scalar output',
              train_rows=10,validation_rows=2,benchmark=False,
              config={'batch':512,'steps':10,'data':'outputs/x'})
    old=dict(base,source_sha256={'a':'1'})
    new=dict(base,sfwdl_rows=100,source_sha256={'a':'2'},
             config={**base['config'],'sfwdl_data':'outputs/sf','sfwdl_mix':0.125})
    assert resume_compatible(old,new)
    bad=dict(new,config={**new['config'],'batch':256})
    assert not resume_compatible(old,bad)


def test_take_ids_wraps(tmp_path):
    rng=torch.Generator().manual_seed(0)
    order=torch.arange(5)
    ids,order,pos,wraps=take_ids(order,3,4,rng,5)
    assert ids[:2]==[3,4] and len(ids)==4 and wraps==1
    assert len(order)==5


def test_disk_backed_data(tmp_path):
    path=tmp_path/'train.npz'
    x=np.stack([pack_board(chess.Board())]*3)
    np.savez(path,packed=x,target=np.array([-.5,0,.5],dtype=np.float32),phase=np.zeros(3,dtype=np.uint8))
    manifest={'shards':[{'file':path.name,'rows':3,'split':'train','sha256':digest(path)}]}
    for _ in range(2):
        packed,target,phase=load_split(tmp_path,manifest,'train')
        np.testing.assert_array_equal(packed.numpy(),x)
        assert target.tolist()==[-.5,0,.5]


def _array_from_board(board):
    arr=np.zeros(64,dtype=np.int8)
    for sq,piece in board.piece_map().items():
        arr[sq]=piece.piece_type+(0 if piece.color else 6)
    castle=(8*board.has_kingside_castling_rights(True)+4*board.has_queenside_castling_rights(True)
            +2*board.has_kingside_castling_rights(False)+board.has_queenside_castling_rights(False))
    ep=-1 if board.ep_square is None else board.ep_square
    return arr,0 if board.turn else 1,castle,ep


def test_sfwdl_white_abs_becomes_stm():
    assert abs(stm_target([.8,.1,.1],0)-.7)<1e-6
    assert abs(stm_target([.8,.1,.1],1)+.7)<1e-6
    assert stm_target([.2,.2,.2],0) is None
    board=chess.Board()
    arr,turn,castle,ep=_array_from_board(board)
    assert chess.Board(board_array_to_fen(arr,turn,castle,ep)).fen().split()[:4]==board.fen().split()[:4]
    white,err=convert_row(dict(board_array=arr,turn=turn,castling=castle,ep_square=ep,
                               wdl=[.7,.2,.1],wdl_source=1,split=1))
    assert err is None and abs(white['target']-.6)<1e-6 and white['black']==0
    np.testing.assert_array_equal(white['packed'],pack_board(board))
    board.push_uci('e2e4')
    arr,turn,castle,ep=_array_from_board(board)
    black,err=convert_row(dict(board_array=arr,turn=turn,castling=castle,ep_square=ep,
                               wdl=[.7,.2,.1],wdl_source=1,split=1))
    assert err is None and abs(black['target']+.6)<1e-6 and black['black']==1
    np.testing.assert_array_equal(black['packed'],pack_board(board))
    assert convert_row(dict(board_array=arr,turn=turn,castling=castle,ep_square=ep,
                            wdl=[.7,.2,.1],wdl_source=2,split=1))[1]=='not_official_wdl'
    np.testing.assert_array_equal(pack_from_arrays(arr,turn,castle,ep),pack_board(board))
    assert convert_train_row(dict(board_array=arr,turn=turn,castling=castle,ep_square=ep,
                                 wdl=[.7,.2,.1],wdl_source=1,split=1))[1]=='holdout'
    train,err=convert_train_row(dict(board_array=arr,turn=turn,castling=castle,ep_square=ep,
                                    wdl=[.7,.2,.1],wdl_source=1,split=0))
    assert err is None and abs(train['target']+.6)<1e-6


def test_sfwdl_done_keys_survive_new_hf_snapshot():
    old='/cache/snapshots/aaa/data/shard_000541.parquet'
    new='/cache/snapshots/bbb/data/shard_000541.parquet'
    assert source_key('parquet',old)==source_key('parquet',new)=='parquet:shard_000541.parquet'
    assert source_key('pt','/inbox/shard_000542/soft_cache.pt')=='pt:shard_000542'
    done=normalize_done([old,'/inbox/shard_000542/soft_cache.pt'])
    assert 'parquet:shard_000541.parquet' in done
    assert source_key('parquet',new) in done
    assert 'pt:shard_000542' in done


def test_sfwdl_reload_gates():
    assert mix_n(512,0.125,None,1000)==64
    assert mix_n(512,0.125,None,0)==0
    stamp=(2,10)
    assert should_reload_sf(stamp,None,10,0,1800,False)
    assert not should_reload_sf(stamp,stamp,10,0,1800,True)
    assert not should_reload_sf(stamp,(1,10),10,9,1800,True)
    assert should_reload_sf(stamp,(1,10),2000,0,1800,True)
    assert should_reload_sf(stamp,(1,10),10,0,1800,False)
