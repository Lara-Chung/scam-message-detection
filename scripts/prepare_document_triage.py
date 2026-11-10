"""Create a document policy using canonical VAL only; never fit or call an LLM."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from src.agent.candidates import CANDIDATES,choose_best,metrics
from src.agent.data import eligible,file_hash,load_channel,membership_hash
from src.agent.documents import Document
from src.agent.routing import RouterEngine
from src.agent.service import ScamAgentService
from src.normalise import normalise_with_changes

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'models/triage_seed0'


def main():
    if OUTPUT.exists():
        raise FileExistsError('Document policy exists; no automatic overwrite or retuning.')
    # The public profile requires no credentials for preparation.
    service=ScamAgentService.from_config(ROOT/'configs/agent.yaml')
    engine=RouterEngine(service.config,service.limits)
    catalog=copy.deepcopy(engine.catalog)
    catalog['triage_policy_version']=1
    catalog['triage_window_chars']=2000
    calibration={}
    for channel,info in catalog['channels'].items():
        val=eligible(load_channel(channel));val=val.loc[val.split_seed0=='val'].sort_values('id')
        texts=[]
        for row in val.itertuples():
            # Oversized VAL sources do not silently disappear from calibration.
            doc=Document(row.text)
            texts.append(doc.read(doc.initial_id)['normalised'])
        y=val.label_fraud.astype(int).to_numpy()
        preds={};distances={}
        for name in CANDIDATES:
            model=engine.adapter(channel,name).model
            x=model.named_steps['tfidf'].transform(texts);clf=model.named_steps['clf']
            preds[name]=clf.predict(x).astype(int)
            if hasattr(clf,'predict_proba'):
                scores=clf.predict_proba(x)[:,list(clf.classes_).index(1)]
                distances[name]=np.abs(scores-.5)
            else:
                distances[name]=np.abs(clf.decision_function(x))
        primary=choose_best(y,preds)
        others=[n for n in CANDIDATES if n!=primary]
        # Prefer a candidate that rescues primary errors; use VAL F1 and order for ties.
        secondary=max(others,key=lambda n:(int(((preds[primary]!=y)&(preds[n]==y)).sum()),metrics(y,preds[n])['macro_f1']))
        policy={'primary_model':primary,'secondary_model':secondary,
                'boundary_quantile':.10,'boundary_threshold':float(np.quantile(distances[primary],.10)),
                'threshold_interpretation':'Native model distance, not calibrated error probability.',
                'val_ids_sha256':membership_hash(val.id),'val_n':len(val),
                'models':{n:metrics(y,preds[n]) for n in CANDIDATES}}
        info['triage_policy']=policy
        calibration[channel]=policy
        print(channel,primary,secondary,policy['boundary_threshold'],flush=True)
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'catalog.json').write_text(json.dumps(catalog,indent=2))
    (OUTPUT/'validation.json').write_text(json.dumps({'partition':'canonical_seed0_val','policy':calibration,
        'settings':'2000-char non-overlapping source windows; first substantive window; original model TRAIN untouched',
        'boundary_quantile_fixed_before_evaluation':.10,'source_code_sha256':{
            p:file_hash(ROOT/p) for p in ['src/agent/documents.py','scripts/prepare_document_triage.py']}},indent=2))

if __name__=='__main__':main()
