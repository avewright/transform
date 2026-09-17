from pathlib import Path
import random
import chess
import pytest
import torch
from rl_selfplay.chessbot_ppo import (
    load_original,log_probs,actor_value,reward,finish,sample,collect,pair_stats,
)

SOURCE=Path(__file__).resolve().parents[1]/'outputs/chessbot_rl_source'


def test_pair_stats_counts_both_color_wins():
    opening=['e2e4']
    match=dict(games=[
        dict(opening=opening,color=True,reward=1),
        dict(opening=opening,color=False,reward=1),
        dict(opening=['d2d4'],color=True,reward=1),
        dict(opening=['d2d4'],color=False,reward=-1),
    ])
    s=pair_stats(match)
    assert s==dict(pairs=2,plus_pairs=1,minus_pairs=0,tied_pairs=1,pair_score=.5)


def test_terminal_and_truncation_credit():
    rows=[{'value':.2},{'value':.4}]
    finish(rows,-1,0,lam=1)
    assert [r['target'] for r in rows]==pytest.approx([-1,-1])
    rows=[{'value':.2},{'value':.4}]
    finish(rows,None,.7,lam=1)
    assert [r['target'] for r in rows]==pytest.approx([.7,.7])
    finish([],None,.2)


def test_color_and_outcomes():
    logits=torch.tensor([[0.,0.,10.],[0.,0.,10.]])
    v=actor_value(logits,[True,False])
    assert v[0]>.99 and v[1]<-.99
    b=chess.Board()
    for u in ['f2f3','e7e5','g2g4','d8h4']:b.push_uci(u)
    assert reward(b,chess.WHITE)==-1 and reward(b,chess.BLACK)==1
    b=chess.Board('8/8/8/8/8/2k5/8/K7 w - - 0 1')
    assert reward(b,True)==0
    b=chess.Board()
    for u in ['g1f3','g8f6','f3g1','f6g8']*2:b.push_uci(u)
    assert reward(b,True)==0
    assert reward(chess.Board(b.fen()),True) is None  # FEN loses repetition history.


def test_mask():
    logits=torch.tensor([[20.,1.,2.]],requires_grad=True)
    p=log_probs(logits,torch.tensor([[False,True,False]]))
    assert p.exp().tolist()==[[0.,1.,0.]]
    (-p[:,1].mean()).backward()
    assert torch.isfinite(logits.grad).all()


@pytest.fixture(scope='module')
def adapter():
    if not (SOURCE/'model.safetensors').exists():pytest.skip('Pinned source not downloaded')
    torch.set_num_threads(2)
    return load_original(SOURCE,'cpu')


def test_published_parity_and_promotions(adapter):
    fens=[chess.STARTING_FEN,
          'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
          '7k/P7/8/8/8/8/7p/K7 w - - 0 1',
          '7k/P7/8/8/8/8/7p/K7 b - - 0 1',
          '7k/8/8/3pP3/8/8/8/K7 w - d6 0 1']
    with torch.no_grad():
        p,q,mask,maps=adapter.forward(fens,'cpu')
        for i,fen in enumerate(fens):
            b=chess.Board(fen)
            assert set(maps[i].values())==set(b.legal_moves)
            assert len(maps[i])==len(list(b.legal_moves))
            native=adapter.model.get_position_value(fen,device='cpu')
            torch.testing.assert_close(q[i].softmax(-1),native)
            x,_,_=adapter.tensors([fen],'cpu')
            native_policy=adapter.model(x).last_hidden_state[0,0]
            torch.testing.assert_close(p[i],native_policy,rtol=2e-4,atol=2e-4)
    assert {chess.Move.from_uci('a7a8'+p) for p in 'qnrb'}<=set(maps[2].values())
    assert {chess.Move.from_uci('h2h1'+p) for p in 'qnrb'}<=set(maps[3].values())


def test_behavior_ratio_and_save_load(adapter,tmp_path):
    boards=[chess.Board(),chess.Board()]
    samples=sample(adapter,boards,'cpu',torch.Generator().manual_seed(291))
    with torch.no_grad():
        p,q,m,_=adapter.forward([r['fen'] for _,r in samples],'cpu')
        lp=log_probs(p,m)
    for i,(move,r) in enumerate(samples):
        assert move in boards[i].legal_moves
        assert float((lp[i,r['action']]-r['old_logp']).exp())==pytest.approx(1.)
    path=tmp_path/'state.pt';torch.save(adapter.model.state_dict(),path)
    clone=adapter.clone();clone.model.load_state_dict(torch.load(path,weights_only=True))
    with torch.no_grad():p2,q2,_,_=clone.forward([b.fen() for b in boards],'cpu')
    torch.testing.assert_close(p,p2);torch.testing.assert_close(q,q2)


def test_collector_caps_bootstrap_in_actor_frame(adapter):
    cfg=dict(environments=2,decisions=1,min_games=0,ply_cap=2)
    rows,games=collect(adapter,{'ref':(adapter,1.)},cfg,[[]],'cpu',
                       torch.Generator().manual_seed(8),random.Random(8),lambda _:None)
    assert len(games)==2 and rows
    assert all(g['reward'] is None and g['truncated'] for g in games)
    for g in games:
        b=chess.Board()
        for u in g['moves']:b.push_uci(u)
        assert b.turn==g['color']
        with torch.no_grad():_,q,_,_=adapter.forward([b.fen()],'cpu')
        assert g['bootstrap']==pytest.approx(float(actor_value(q,[g['color']])[0]))
