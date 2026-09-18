from pathlib import Path
import random
import chess
import pytest
import torch
from rl_selfplay.chessbot_eval import (
    game_score,paired_eval,paired_interval,previous_from_league,screen_roles,verdict,
)
from rl_selfplay.chessbot_ppo import (
    load_original,log_probs,actor_value,reward,finish,sample,collect,pair_stats,
    should_full_audit,update,
)

SOURCE=Path(__file__).resolve().parents[1]/'outputs/chessbot_rl_source'


def test_paired_ci_is_not_unfinished_bounds():
    opening=['e2e4']
    match=dict(games=[
        dict(opening=opening,color=True,reward=1),
        dict(opening=opening,color=False,reward=1),
    ], wins=2, draws=0, losses=0, unknown=0, n=2, score_bounds=[1.0, 1.0])
    ev=paired_eval(match)
    assert ev['score']==1.0
    assert ev['paired_ci_95']==[1.0, 1.0]
    assert ev['verdict']=='stronger'
    assert ev['score_bounds']==[1.0, 1.0]
    unfinished=dict(games=[
        dict(opening=opening,color=True,reward=1),
        dict(opening=opening,color=False,reward=None),
    ], wins=1, draws=0, losses=0, unknown=1, n=2, score_bounds=[0.5, 1.0])
    ev2=paired_eval(unfinished)
    assert ev2['verdict']=='not_evaluated'
    assert ev2['score_bounds']==[0.5, 1.0]
    assert ev2['unfinished_pairs']==1


def test_verdict_inconclusive_when_ci_covers_half():
    units=[0.5]*32
    mean,ci,_=paired_interval(units)
    assert mean==0.5 and ci==[0.5, 0.5]
    assert verdict(ci)=='inconclusive'
    assert verdict([0.51, 0.70])=='stronger'
    assert verdict([0.30, 0.49])=='weaker'
    assert verdict([None, None])=='not_evaluated'
    assert game_score(1)==1 and game_score(0)==.5 and game_score(-1)==0


def test_previous_excludes_just_saved_current(tmp_path):
    old=tmp_path/'actor_005.pt';old.write_bytes(b'x')
    cur=tmp_path/'actor_010.pt';cur.write_bytes(b'y')
    league=[str(old),str(cur)]
    assert previous_from_league(league,exclude=[cur])==str(old.resolve())
    assert previous_from_league([str(cur)],exclude=[cur]) is None
    assert screen_roles(incumbent=True,control=True,previous=str(old),recurrent=True)==[
        'original','incumbent','control','previous','self_n1']
    assert 'incumbent' in screen_roles(incumbent=True,control=False,previous=None,recurrent=False)


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


def test_full_audit_on_cadence_and_epoch_end():
    assert should_full_audit(8, False, 8)
    assert should_full_audit(7, True, 8)
    assert not should_full_audit(7, False, 8)
    assert should_full_audit(1, False, 1)


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


def test_published_adapter_reads_hf_output():
    if not (SOURCE/'model.safetensors').exists():
        pytest.skip('Pinned source not downloaded')
    torch.set_num_threads(2)
    published=load_original(SOURCE,'cpu',unrolls=1)
    x,_,_=published.tensors([chess.STARTING_FEN],'cpu')
    raw=published.model(x)
    assert hasattr(raw,'last_hidden_state')
    assert isinstance(raw,dict)
    assert 'policy_logits' not in raw
    with torch.no_grad():
        p,q,mask,maps=published.forward([chess.STARTING_FEN],'cpu')
    assert p.shape[-1]==1929 and q.shape[-1]==3
    assert int(mask[0].sum())==len(maps[0])==20


def test_gated_n2_matches_published_at_train_depth():
    if not (SOURCE/'model.safetensors').exists():
        pytest.skip('Pinned source not downloaded')
    from chess_chessbot import fens_to_planes
    from chess_chessbot_recurrent import depth_identity_errors, identity_errors
    from rl_selfplay.chessbot_ppo import compare_policies
    torch.set_num_threads(2)
    published=load_original(SOURCE,'cpu',unrolls=1)
    wrapped=load_original(SOURCE,'cpu',unrolls=2)
    assert wrapped.unrolls==2
    assert wrapped.model.effective_depth()==16
    assert float(wrapped.model.alpha)==0
    planes=fens_to_planes([chess.STARTING_FEN])
    assert max(identity_errors(wrapped.model,planes).values())<1e-5
    assert max(depth_identity_errors(wrapped.model,planes,2).values())<1e-5
    cmp=compare_policies(wrapped,published,[chess.STARTING_FEN],'cpu')
    assert cmp['legal_agreement']==1.0
    assert cmp['policy_kl']<1e-6
    wrapped.model.alpha.data.fill_(0.5)
    assert max(depth_identity_errors(wrapped.model,planes,2).values())>1e-4
    from rl_selfplay.chessbot_ppo import top_moves
    one=top_moves(wrapped,chess.STARTING_FEN,'cpu',unrolls=1)
    two=top_moves(wrapped,chess.STARTING_FEN,'cpu',unrolls=2)
    assert one['preferred']!=two['preferred'] or one['top'][0]['p']!=two['top'][0]['p']
    wrapped.model.alpha.data.zero_()
    assert top_moves(wrapped,chess.STARTING_FEN,'cpu',unrolls=2)['preferred']==top_moves(published,chess.STARTING_FEN,'cpu')['preferred']


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


def test_forward_tensors_matches_fen_forward(adapter):
    x,mask,_=adapter.tensors([chess.STARTING_FEN],'cpu')
    with torch.no_grad():
        p,q,_=adapter.forward_tensors(x,mask)
        p2,q2,_,_=adapter.forward([chess.STARTING_FEN],'cpu')
    torch.testing.assert_close(p,p2)
    torch.testing.assert_close(q,q2)


def test_cached_update_matches_uncached_first_step(adapter):
    torch.set_num_threads(2)
    boards=[chess.Board() for _ in range(8)]
    rng=torch.Generator().manual_seed(7)
    samples=sample(adapter,boards,'cpu',rng)
    rows=[]
    for i,(_,r) in enumerate(samples):
        r=dict(r);r['advantage']=1. if i%2==0 else -1.;r['target']=r['advantage']+r['value']
        rows.append(r)
    anchors=[chess.STARTING_FEN]*8
    cfg=dict(epochs=1,minibatch=8,microbatch=4,anchor_batch=4,audit_size=8,
             audit_cheap_size=8,audit_full_every=1,audit_microbatch=4,cache_rollout=False,lr=1e-6)
    def run(cache):
        actor=adapter.clone();control=adapter.clone();ref=adapter.clone()
        opt=torch.optim.AdamW([p for p in actor.model.parameters() if p.requires_grad],lr=1e-6,weight_decay=0.)
        copt=torch.optim.AdamW([p for p in control.model.parameters() if p.requires_grad],lr=1e-6,weight_decay=0.)
        rec=[]
        update(actor,control,ref,opt,copt,rows,anchors,{**cfg,'cache_rollout':cache},
               'cpu',torch.Generator().manual_seed(11),rec.append)
        return rec[0]
    a,b=run(False),run(True)
    assert a['policy']==pytest.approx(b['policy'],rel=1e-5,abs=1e-5)
    assert a['value']==pytest.approx(b['value'],rel=1e-5,abs=1e-5)
    assert a['reference_kl']==pytest.approx(b['reference_kl'],rel=1e-5,abs=1e-5)
    assert a['behavior_kl']==pytest.approx(b['behavior_kl'],rel=1e-4,abs=1e-5)


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
