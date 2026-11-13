"""Build supplementary UCI demo models, not replacements for team benchmarks.

Run from repo root: python scripts/train_demo_models.py
Fixed settings, one seed, same held-out grouped split as LR run 1. SVM
probabilities are calibrated from grouped out-of-fold TRAIN predictions only.
No hyperparameter selection or calibration uses the held-out test set.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from lr_scam import load_data, grouped_split, evaluate, summarise


def main() -> None:
    """Fit three models and export actual held-out metrics and provenance."""
    data=load_data(str(ROOT/'data/demo/uci.csv.gz'),'text','label_orig')
    tr,te=grouped_split(data,42)
    train,test=data.iloc[tr],data.iloc[te]
    group_cv=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(train,train.label,groups=train.group))
    options=[
        ('nb', Pipeline([('tfidf',TfidfVectorizer(lowercase=True,sublinear_tf=True,ngram_range=(1,2))),('clf',MultinomialNB(alpha=1.0))])),
        ('svm_word', CalibratedClassifierCV(Pipeline([('tfidf',TfidfVectorizer(lowercase=True,sublinear_tf=True,ngram_range=(1,2),min_df=2)),('clf',LinearSVC(C=1,class_weight='balanced',random_state=42,max_iter=10000))]),method='sigmoid',cv=group_cv,ensemble=False)),
        ('svm_char', CalibratedClassifierCV(Pipeline([('tfidf',TfidfVectorizer(analyzer='char',lowercase=True,sublinear_tf=True,ngram_range=(3,5),min_df=2)),('clf',LinearSVC(C=1,class_weight='balanced',random_state=42,max_iter=10000))]),method='sigmoid',cv=group_cv,ensemble=False)),
    ]
    for name,model in options:
        output=ROOT/f'artifacts/demo_uci_{name}';output.mkdir(exist_ok=True)
        start=time.perf_counter()
        kwargs={'clf__sample_weight':compute_sample_weight('balanced',train.label)} if name=='nb' else {}
        model.fit(train.text,train.label,**kwargs)
        seconds=time.perf_counter()-start
        path=output/f'{name}_weighted_split0.joblib';joblib.dump(model,path)
        metrics=evaluate(model,test,42)
        metrics.update(split_seed=42,best_params={'settings':'Fixed demo parameters; not tuned. See scripts/train_demo_models.py', 'calibration':'3-fold template-grouped train-only sigmoid calibration' if name.startswith('svm') else 'Not probability-calibrated; balanced sample weights'},tune_seconds=0.0,train_seconds=seconds,model_size_mb=path.stat().st_size/1e6)
        report={'runs':{'weighted':[metrics]},'summary':{'weighted':summarise([metrics])},'provenance':{'kind':'supplementary_demo_training','not_team_original_outputs':True,'seed':42,'n_train':len(train),'n_test':len(test),'test_usage':'Final reporting only; fixed parameters; calibration fits train OOF predictions only','sklearn_version':sklearn.__version__}}
        (output/'results.json').write_text(json.dumps(report,indent=2))
        pd.DataFrame([report['summary']['weighted']]).to_csv(output/'results_table.csv',index=False)
        print(name,'seconds',round(seconds,2),'macro_f1',round(metrics['macro_f1'],5),'matrix',metrics['confusion_matrix'],flush=True)

if __name__=='__main__':main()
