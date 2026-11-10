"""Model-specific decision-boundary distances and VAL-only cross-check calibration."""
from __future__ import annotations

import numpy as np

QUANTILES=[.05,.10,.20,.30]
MIN_CONFLICT_BASES=10
MIN_WINNER_SHARE=.60


def distance(result):
    score=result['fraud_score']
    if score is None:raise ValueError('A continuous binary score is required.')
    if result['score_kind']=='decision_margin':return abs(float(score))
    if result['score_kind']=='model_probability_uncalibrated':return abs(float(score)-.5)
    raise ValueError('Unsupported boundary score kind.')


def resolution(y,p,s,ids,near,primary,secondary):
    """Direction-specific conflict policy, never based on TEST labels."""
    policies={};selection=np.array(p,dtype=int).copy();abstain=np.zeros(len(y),dtype=bool)
    for predicted in [0,1]:
        conflict=near & (p==predicted) & (p!=s)
        # Correlated variants from a base count as one vote for fitting a conflict policy.
        votes={}
        for i in np.flatnonzero(conflict):votes.setdefault(ids[i],[]).append(int(s[i]==y[i]))
        n=len(votes);secondary_share=float(np.mean([np.mean(v) for v in votes.values()])) if n else None
        if n>=MIN_CONFLICT_BASES and secondary_share>=MIN_WINNER_SHARE:winner=secondary
        elif n>=MIN_CONFLICT_BASES and secondary_share<=1-MIN_WINNER_SHARE:winner=primary
        else:winner='abstain'
        policies[str(predicted)]={'selected_model':winner,'unique_conflict_bases':n,'secondary_correct_share':secondary_share}
        if winner==secondary:selection[conflict]=s[conflict]
        elif winner=='abstain':abstain[conflict]=True
    return policies,selection,abstain


def calibrate(y,predictions,distances,ids,primary,candidate_order):
    choices=[]
    for q in QUANTILES:
        threshold=float(np.quantile(distances[primary],q))
        near=distances[primary]<=threshold
        for secondary in candidate_order:
            if secondary==primary:continue
            rules,selected,abstain=resolution(y,predictions[primary],predictions[secondary],ids,near,primary,secondary)
            correct=(selected==y)&~abstain;wrong=(selected!=y)&~abstain
            # Predeclared safety/call objective: wrong=-2, abstain=-.25, extra local call=-.02.
            utility=float((correct.sum()-2*wrong.sum()-.25*abstain.sum()-.02*near.sum())/len(y))
            choices.append({'primary_model':primary,'secondary_model':secondary,'quantile':q,'threshold':threshold,
                'distance_kind':'abs_native_margin' if primary.endswith('svm') else 'abs_uncalibrated_probability_minus_half',
                'conflict_by_primary_prediction':rules,'selection_objective':utility,
                'apparent_validation':{'n_correlated_variants':len(y),'correct':int(correct.sum()),'wrong':int(wrong.sum()),
                    'abstentions':int(abstain.sum()),'second_calls':int(near.sum()),'near_unique_bases':len({ids[i] for i in np.flatnonzero(near)})}})
    # Stable q order and candidate order break exact objective ties.
    return max(choices,key=lambda c:c['selection_objective'])
