"""Deterministic, letter-only stress variants and label-blind challenge sampling."""
from __future__ import annotations

import hashlib
import re

import numpy as np

from src.agent.candidates import CANDIDATES,features

FAMILIES=['invisible','homoglyph','fullwidth','spaced']
SEVERITIES=[0.25,0.5]
BASE_CHARS=4000
PROTECT=re.compile(r'(?:https?://|www\.)\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|'
    r'\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/\S*)?|'
    r'[$£€]\s*\d[\d,.]*(?:\s*(?:million|billion|thousand|USD|SGD|GBP|EUR))*',re.I)
HOMO={'a':'а','e':'е','o':'о','p':'р','c':'с','i':'і','s':'ѕ','x':'х','y':'у',
      'A':'А','E':'Е','O':'О','P':'Р','C':'С','I':'І','S':'Ѕ','X':'Х','Y':'Ү'}


def profile(observed):
    keys={'invisible_chars':'invisible','compat_chars':'compat','lookalike_letters':'lookalike','spaced_letters':'spaced'}
    active=[name for key,name in keys.items() if observed['normalisation_changes'][key]>0]
    if len(active)>1:return 'mixed'
    if active:return active[0]
    return 'digit_word' if observed['digit_inside_word'] else 'none'


def perturb(text,family,severity,seed):
    """Preserve digits, punctuation and protected contact/amount substrings.

    The inherited label is a simulation assumption; this is not a real attack.
    Return edits against the original base prefix, never a fabricated source view.
    """
    if family not in FAMILIES or severity not in SEVERITIES:raise ValueError('Unknown fixed stress condition.')
    spans=[m.span() for m in PROTECT.finditer(text)]
    eligible=[]
    for match in re.finditer(r'[A-Za-z]{4,}',text):
        a,b=match.span()
        if any(a<end and b>start for start,end in spans):continue
        eligible.append(match)
    parts=[];last=0;last_selected=-2;edits=[]
    for index,match in enumerate(eligible):
        digest=hashlib.sha256(f'{seed}:{match.start()}:{family}'.encode()).digest()
        if int.from_bytes(digest[:8],'big')/2**64>=severity:continue
        if family=='spaced' and index==last_selected+1:continue
        word=match.group()
        if family=='invisible':changed=word[:2]+'\u200b'+word[2:]
        elif family=='homoglyph':
            replacement=next((i for i,c in enumerate(word) if c in HOMO),None)
            if replacement is None:continue
            changed=word[:replacement]+HOMO[word[replacement]]+word[replacement+1:]
        elif family=='fullwidth':changed=''.join(chr(ord(c)+0xFEE0) for c in word)
        else:changed=' '.join(word)
        parts.extend([text[last:match.start()],changed]);last=match.end();last_selected=index
        edits.append({'start':match.start(),'end':match.end(),'original':word,'replacement':changed})
    parts.append(text[last:]);result=''.join(parts)
    assert ''.join(c for c in text if c.isdigit())==''.join(c for c in result if c.isdigit())
    assert len(result)<8000 or len(text)>BASE_CHARS
    return result,edits


def select_label_blind(pool,predictions,limit,seed,require_disagreement):
    """Pool must contain only id/text; truth labels must be joined after selection."""
    if any(set(item)-{'id','text'} for item in pool):raise ValueError('Selector must not receive labels or outcomes.')
    if len(pool)!=len(predictions):raise ValueError('Prediction alignment mismatch.')
    eligible=[]
    for item,pred in zip(pool,predictions):
        if require_disagreement and len(set(pred.values()))<2:continue
        key=hashlib.sha256(f'{seed}:{item["id"]}'.encode()).hexdigest()
        eligible.append((key,item))
    return [item for _,item in sorted(eligible,key=lambda x:x[0])[:limit]]


def batch_predictions(engine,channel,texts):
    return {name:engine.adapter(channel,name).model.predict(texts).astype(int) for name in CANDIDATES}


def disagreements(preds):
    stack=np.stack([preds[n] for n in CANDIDATES])
    return np.any(stack!=stack[0],axis=0)
