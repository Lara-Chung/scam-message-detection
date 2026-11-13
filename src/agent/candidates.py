"""Unified TRAIN-only candidates and VAL-only routing policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.agent.data import CHANNEL_DATASETS,EXPORT,LABEL_DEFINITION,eligible,file_hash,load_channel,membership_hash
from src.normalise import normalise_with_changes

ROOT=Path(__file__).resolve().parents[2]
CANDIDATES=['word_lr','word_nb','word_svm','char_svm']
MAX_CHARS=8000


def features(text):
    _,changes=normalise_with_changes(text)
    obfuscated=bool(sum(changes.values()) or re.search(r'[A-Za-z]\d+[A-Za-z]',text))
    bucket='obfuscated' if obfuscated else ('long' if len(text)>1200 else ('short' if len(text)<=160 else 'standard'))
    return {'characters':len(text),'bucket':bucket,'normalisation_changes':changes,
            'digit_inside_word':bool(re.search(r'[A-Za-z]\d+[A-Za-z]',text))}


def metrics(y,pred):
    cm=confusion_matrix(y,pred,labels=[0,1]).tolist()
    return {'n':len(y),'fraud_support':int(sum(y)),'legitimate_support':int(len(y)-sum(y)),
            'macro_f1':float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
            'accuracy':float(np.mean(np.asarray(y)==np.asarray(pred))),
            'fraud_precision':float(precision_score(y,pred,zero_division=0)),
            'fraud_recall':float(recall_score(y,pred,zero_division=0)),
            'false_positive_rate':cm[0][1]/sum(cm[0]) if sum(cm[0]) else None,'confusion_matrix_0_1':cm}


def choose_best(y,preds):
    # Stable declared candidate order breaks ties, never test outcomes or IDs.
    return max(CANDIDATES,key=lambda name:f1_score(y,preds[name],labels=[0,1],average='macro',zero_division=0))


def metadata(path,rows,channel,dataset,hyperparameters):
    return {'provenance':'canonical_train_verified','channel':channel,'dataset':dataset,
        'dataset_sha256':file_hash(EXPORT/f'{dataset}.csv.gz'),'artifact_sha256':file_hash(path),
        'split_key':'split_seed0','seed':0,'label_definition':LABEL_DEFINITION,'classes':[0,1],
        'train_ids':rows.id.tolist(),'train_ids_sha256':membership_hash(rows.id),
        'training_text_sha256':{r.id:hashlib.sha256(r.text.encode()).hexdigest() for r in rows.itertuples()},
        'fitted_prefix_sha256':{r.id:hashlib.sha256(r.text[:MAX_CHARS].encode()).hexdigest() for r in rows.itertuples()},
        'versions':{'python':platform.python_version(),'scikit-learn':sklearn.__version__,'pandas':pd.__version__,
                    'numpy':np.__version__,'scipy':scipy.__version__,'joblib':joblib.__version__},
        'hyperparameters':hyperparameters,'input_policy':{'name':'prefix_chars_v1','max_chars':MAX_CHARS},
        'selection_policy':'Candidate settings fixed before fitting; fixed/bucket routes selected using canonical VAL only.'}


def train(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Use a new candidate directory; existing artifacts are never overwritten.')
    output.mkdir(parents=True)
    catalog={'version':'1.0','candidate_order':CANDIDATES,'input_policy':{'name':'prefix_chars_v1','max_chars':MAX_CHARS},
        'label_definition':LABEL_DEFINITION,'partition_policy':'canonical seed0 TRAIN only; candidate/policy selection VAL only',
        'bucket_policy':{'priority':['obfuscated','long','short','standard'],'long_chars_gt':1200,'short_chars_le':160,
                         'minimum_bucket_n':30,'minimum_each_class':3,'fallback':'best global VAL macro-F1'},'channels':{}}
    validation=[]
    for channel,dataset in CHANNEL_DATASETS.items():
        df=eligible(load_channel(channel));rows=df.loc[df.split_seed0=='train'].sort_values('id')
        val=df.loc[df.split_seed0=='val'].sort_values('id').copy()
        texts=rows.text.str[:MAX_CHARS];y=rows.label_fraud.astype(int).to_numpy()
        val_texts=val.text.str[:MAX_CHARS];truth=val.label_fraud.astype(int).to_numpy()
        word=TfidfVectorizer(ngram_range=(1,2),max_features=50000,min_df=1,sublinear_tf=True,dtype=np.float32)
        char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=60000,min_df=2,sublinear_tf=True,dtype=np.float32)
        wx=word.fit_transform(texts);cx=char.fit_transform(texts)
        specs={'word_lr':(word,wx,LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=0)),
               'word_nb':(word,wx,MultinomialNB(alpha=1.,fit_prior=False)),
               'word_svm':(word,wx,LinearSVC(C=1.,class_weight='balanced',max_iter=3000,random_state=0)),
               'char_svm':(char,cx,LinearSVC(C=1.,class_weight='balanced',max_iter=3000,random_state=0))}
        info={'dataset':dataset,'train_n':len(rows),'val_n':len(val),'train_ids_sha256':membership_hash(rows.id),
              'val_ids_sha256':membership_hash(val.id),'dataset_sha256':file_hash(EXPORT/f'{dataset}.csv.gz'),'models':{}}
        preds={}
        for name,(vectorizer,matrix,clf) in specs.items():
            started=time.perf_counter();clf.fit(matrix,y)
            pipeline=Pipeline([('tfidf',vectorizer),('clf',clf)])
            path=output/f'{channel}_{name}.joblib';joblib.dump(pipeline,path)
            hyper={'vectorizer':{k:v for k,v in vectorizer.get_params().items() if k in ['analyzer','ngram_range','max_features','min_df','sublinear_tf']},
                   'classifier':{k:v for k,v in clf.get_params().items() if k in ['C','alpha','fit_prior','class_weight','max_iter','random_state']}}
            path.with_suffix('.metadata.json').write_text(json.dumps(metadata(path,rows,channel,dataset,hyper),indent=2))
            tick=time.perf_counter();pred=pipeline.predict(val_texts).astype(int)
            infer_ms=(time.perf_counter()-tick)*1000/len(val)
            preds[name]=pred
            artifact_path=path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            info['models'][name]={'path':artifact_path.as_posix(),'model_id':f'router_{channel}_{name}_seed0_v1','provenance':'canonical_train_verified',
                'label_map':{'0':'legitimate','1':'fraud'},'preprocessing':'Canonical TRAIN-fitted TF-IDF; common original 8000-character prefix.',
                'validation':metrics(truth,pred),'batch_inference_ms_per_case':infer_ms,'training_seconds':time.perf_counter()-started,
                'feature_family':'character 3-5 grams within word boundaries' if name=='char_svm' else 'word 1-2 grams',
                'score_kind':'model_probability_uncalibrated' if name in ['word_lr','word_nb'] else 'decision_margin'}
            print(channel,name,'VAL',round(info['models'][name]['validation']['macro_f1'],4),flush=True)
        best=choose_best(truth,preds);info['fixed_model']=best
        buckets=np.array([features(text)['bucket'] for text in val_texts])
        info['rules']={};info['bucket_validation']={}
        for bucket in ['obfuscated','long','short','standard']:
            mask=buckets==bucket;by=truth[mask]
            supported=len(by)>=30 and min(int(sum(by)),int(len(by)-sum(by)))>=3
            winner=choose_best(by,{n:p[mask] for n,p in preds.items()}) if supported else best
            info['rules'][bucket]=winner
            info['bucket_validation'][bucket]={'n':int(mask.sum()),'fraud_support':int(sum(by)),'rule_model':winner,
                'selection':'bucket VAL macro-F1' if supported else 'global fallback: insufficient bucket support'}
        stack=np.stack([preds[n] for n in CANDIDATES]);correct=stack==truth
        best_correct=preds[best]==truth
        info['complementarity']={'fixed_correct':int(best_correct.sum()),'oracle_correct':int(correct.any(axis=0).sum()),
            'oracle_rescues_fixed_errors':int((~best_correct & correct.any(axis=0)).sum()),
            'all_candidates_wrong':int((~correct.any(axis=0)).sum()),
            'disagreement_cases':int(np.any(stack!=stack[0],axis=0).sum()),
            'oracle_note':'Uses VAL truth for a diagnostic upper bound; impossible deployable routing policy, never sent as per-case clues.',
            'pairwise':{f'{a}__{b}':{'a_wrong_b_correct':int(((preds[a]!=truth)&(preds[b]==truth)).sum()),
                                      'b_wrong_a_correct':int(((preds[b]!=truth)&(preds[a]==truth)).sum())}
                        for i,a in enumerate(CANDIDATES) for b in CANDIDATES[i+1:]}}
        rule_pred=np.array([preds[info['rules'][bucket]][i] for i,bucket in enumerate(buckets)])
        info['rule_validation_apparent']=metrics(truth,rule_pred)
        info['rule_validation_note']='Apparent selection-set result, not independent held-out performance.'
        for i,row in enumerate(val.itertuples()):
            validation.append({'case_id':row.id,'channel':channel,'partition':'val','label':int(truth[i]),'bucket':buckets[i],
                'predictions':{n:int(preds[n][i]) for n in CANDIDATES},'fixed_model':best,'rule_model':info['rules'][buckets[i]]})
        catalog['channels'][channel]=info
        del wx,cx,specs
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2))
    with (output/'validation_predictions.jsonl').open('w') as stream:
        for row in validation:stream.write(json.dumps(row)+'\n')
    print('Saved catalog:',output/'catalog.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'models/router_candidates_seed0')
    args=parser.parse_args();train(args.output)

if __name__=='__main__':main()
