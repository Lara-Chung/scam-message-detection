"""Calibrate normalized routing on VAL, then freeze an independent label-blind TEST challenge."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.agent.boundary import calibrate,QUANTILES,MIN_CONFLICT_BASES,MIN_WINNER_SHARE
from src.agent.candidates import CANDIDATES,features,metrics,choose_best
from src.agent.data import eligible,load_channel,file_hash
from src.agent.routing import RouterEngine
from src.agent.service import ScamAgentService
from src.agent.stress import BASE_CHARS,FAMILIES,SEVERITIES,perturb,profile,select_label_blind
from src.normalise import normalise_with_changes
from scripts.prepare_router_stress import exclude_previous
from scripts.run_agent_comparison import PRICING

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/agent/normalised_boundary_v1'
CATALOG=ROOT/'models/router_normalised_seed0/catalog.json'


def batch(engine,ch,texts):
    preds={};distances={}
    for name in CANDIDATES:
        model=engine.adapter(ch,name).model
        x=model.named_steps['tfidf'].transform(texts);clf=model.named_steps['clf']
        pred=clf.predict(x).astype(int);preds[name]=pred
        score=clf.predict_proba(x)[:,list(clf.classes_).index(1)] if hasattr(clf,'predict_proba') else clf.decision_function(x)
        distances[name]=np.abs(score-.5) if hasattr(clf,'predict_proba') else np.abs(score)
    return preds,distances


def validate_design():
    if BASE.exists() or CATALOG.parent.exists():raise FileExistsError('Use a fresh experiment/catalog; no overwrite.')
    BASE.mkdir(parents=True);CATALOG.parent.mkdir(parents=True)
    service=ScamAgentService.from_config(ROOT/'configs/agent.local.yaml');engine=RouterEngine(service.config,service.limits)
    catalog=copy.deepcopy(engine.catalog);catalog['boundary_policy_version']=1;catalog['inference_normalisation']=True
    catalog['rule_feature']='normalisation_profile';diagnostics={};all_records=[]
    for ch,info in catalog['channels'].items():
        val=eligible(load_channel(ch));val=val.loc[val.split_seed0=='val'].sort_values('id')
        ys=[];ids=[];profiles=[];pp={n:[] for n in CANDIDATES};dd={n:[] for n in CANDIDATES};stats=[]
        for condition in ['clean']+[f'{f}:{s}' for f in FAMILIES for s in SEVERITIES]:
            texts=[];originals=[];restored=0;changed=0
            for row in val.itertuples():
                base=row.text[:BASE_CHARS];raw=base
                if condition!='clean':
                    f,s=condition.split(':');raw,edits=perturb(base,f,float(s),f'boundary-val:{row.id}');changed+=bool(edits)
                norm,_=normalise_with_changes(raw);texts.append(norm);originals.append(raw);restored+=norm==base
            preds,dists=batch(engine,ch,texts);truth=val.label_fraud.astype(int).to_numpy()
            stat={'condition':condition,'n':len(val),'changed_n':changed,'normalised_equals_clean_n':restored,
                'normalised_disagreement_n':int(np.any(np.stack(list(preds.values()))!=preds[CANDIDATES[0]],axis=0).sum()),
                'models':{n:metrics(truth,preds[n]) for n in CANDIDATES}}
            stats.append(stat);ys.extend(truth.tolist());ids.extend(val.id.tolist())
            current_profiles=[profile(features(t)) for t in originals];profiles.extend(current_profiles)
            for n in CANDIDATES:pp[n].extend(preds[n].tolist());dd[n].extend(dists[n].tolist())
            for i,row in enumerate(val.itertuples()):
                all_records.append({'source_id':row.id,'channel':ch,'partition':'val','condition':condition,'label':int(truth[i]),
                    'profile':current_profiles[i],'predictions':{n:int(preds[n][i]) for n in CANDIDATES},
                    'boundary_distances':{n:float(dists[n][i]) for n in CANDIDATES}})
            print(ch,condition,'normalised disagreements',stat['normalised_disagreement_n'],flush=True)
        y=np.array(ys);preds={n:np.array(pp[n]) for n in CANDIDATES};dists={n:np.array(dd[n]) for n in CANDIDATES}
        means={n:float(np.mean([s['models'][n]['macro_f1'] for s in stats])) for n in CANDIDATES}
        fixed=max(CANDIDATES,key=lambda n:means[n]);info['fixed_model']=fixed
        info['rules']={};notes={}
        for key in ['none','digit_word','invisible','compat','lookalike','spaced','mixed']:
            mask=np.array(profiles)==key;unique={ids[i]:int(y[i]) for i in np.flatnonzero(mask)}
            enough=len(unique)>=30 and min(sum(unique.values()),len(unique)-sum(unique.values()))>=3
            chosen=choose_best(y[mask],{n:p[mask] for n,p in preds.items()}) if enough else fixed
            info['rules'][key]=chosen;notes[key]={'unique_bases':len(unique),'model':chosen,'supported':enough}
        info['boundary_policies']={n:calibrate(y,preds,dists,ids,n,CANDIDATES) for n in CANDIDATES}
        for n,spec in info['models'].items():
            spec['original_validation']=spec['validation'];spec['validation']={'n_unique_bases':len(val),'conditions':9,
                'macro_f1':means[n],'input_view':'normalised','interpretation':'mean condition result; correlated variants'}
            spec['routing_profile_validation']={key:{'unique_base_n':note['unique_bases'],
                'macro_f1':metrics(y[np.array(profiles)==key],preds[n][np.array(profiles)==key])['macro_f1'] if note['unique_bases'] else None}
                for key,note in notes.items()}
        options=[s for s in stats if s['condition']!='clean' and s['changed_n']>=s['n']/2]
        hardest=max(options,key=lambda s:(s['normalised_disagreement_n']/s['n'],-float(s['condition'].split(':')[1]),
                                         -FAMILIES.index(s['condition'].split(':')[0])))
        diagnostics[ch]={'fixed_model':fixed,'mean_normalised_condition_f1':means,'profile_rules':notes,
                         'boundary_policies':info['boundary_policies'],'conditions':stats,'test_condition':hardest['condition']}
    CATALOG.write_text(json.dumps(catalog,indent=2))
    with (BASE/'validation_predictions.jsonl').open('x') as out:
        for r in all_records:out.write(json.dumps(r)+'\n')
    design={'stage':'VAL-only locked before independent TEST selection','normalise_function':'src.normalise.normalise_with_changes',
        'same_view_all_policies':'normalize original common base/perturbed prefix before every classifier',
        'threshold_quantiles':QUANTILES,'distance_semantics':'SVM abs oriented margin; LR/NB abs fraud probability minus .5, not calibrated risk',
        'calibration_objective':'(correct - 2*wrong - .25*abstain - .02*second_calls)/N; q and secondary by VAL only',
        'conflict_policy':{'minimum_unique_conflict_bases':MIN_CONFLICT_BASES,'minimum_winner_share':MIN_WINNER_SHARE,
                           'insufficient_support':'abstain and review','any_conflict':'review even with a selected winner'},
        'counting':'9 correlated conditions per VAL base; use unique base votes/support for conflict fitting',
        'sampling_seed':6407,'per_channel_strata':['clean_near_boundary_or_fallback','paired_clean','paired_obfuscated_near_boundary_or_fallback','random_control'],
        'catalog_sha256':file_hash(CATALOG),'diagnostics':diagnostics,'pricing':PRICING}
    (BASE/'validation_design.json').write_text(json.dumps(design,indent=2))
    print('Locked normalized/boundary design',flush=True)


def primary(info,text):return info['rules'].get(profile(features(text)),info['fixed_model'])


def select_test():
    if (BASE/'manifest.json').exists():raise FileExistsError('TEST manifest already exists.')
    design=json.loads((BASE/'validation_design.json').read_text());catalog=json.loads(CATALOG.read_text())
    if file_hash(CATALOG)!=design['catalog_sha256']:raise ValueError('VAL catalog changed.')
    service=ScamAgentService.from_config(ROOT/'configs/agent.local.yaml')
    config=service.config;config['routing_catalog_path']=str(CATALOG)
    engine=RouterEngine(config,config['limits']);excluded,sources=exclude_previous(BASE)
    # exclude_previous also knows independent challenge source IDs.
    cases=[];selection=[]
    for ch,info in catalog['channels'].items():
        df=eligible(load_channel(ch)).set_index('id')
        pool=[{'id':ident,'text':r.text[:BASE_CHARS]} for ident,r in df.iterrows() if r.split_seed0=='test' and (ch,ident) not in excluded]
        norm=[normalise_with_changes(r['text'])[0] for r in pool];pred,dists=batch(engine,ch,norm)
        near=[float(dists[primary(info,r['text'])][i])<=info['boundary_policies'][primary(info,r['text'])]['threshold'] for i,r in enumerate(pool)]
        candidates=[r for r,yes in zip(pool,near) if yes]
        first=select_label_blind(candidates,[{}]*len(candidates),1,f'6407:{ch}:near',False)
        fallback=not first
        if not first:first=select_label_blind(pool,[{}]*len(pool),1,f'6407:{ch}:near',False)
        rest=[r for r in pool if r['id']!=first[0]['id']]
        condition=design['diagnostics'][ch]['test_condition'];f,s=condition.split(':')
        transformed=[perturb(r['text'],f,float(s),f'boundary-test:{r["id"]}')[0] for r in rest]
        npred,ndist=batch(engine,ch,[normalise_with_changes(t)[0] for t in transformed])
        pair_candidates=[r for i,r in enumerate(rest) if ndist[primary(info,transformed[i])][i]<=info['boundary_policies'][primary(info,transformed[i])]['threshold']]
        pair=select_label_blind(pair_candidates,[{}]*len(pair_candidates),1,f'6407:{ch}:pair',False)
        pair_fallback=not pair
        if not pair:pair=select_label_blind(rest,[{}]*len(rest),1,f'6407:{ch}:pair',False)
        remaining=[r for r in rest if r['id']!=pair[0]['id']]
        control=select_label_blind(remaining,[{}]*len(remaining),1,f'6407:{ch}:control',False)[0]
        corrupted,edits=perturb(pair[0]['text'],f,float(s),f'boundary-test:{pair[0]["id"]}')
        entries=[(first[0],'clean_near_boundary' if not fallback else 'clean_control_fallback',first[0]['text'],[]),
                 (pair[0],'paired_clean',pair[0]['text'],[]),(pair[0],'paired_obfuscated',corrupted,edits),
                 (control,'random_control',control['text'],[])]
        for item,stratum,text,edit_list in entries:
            source=df.loc[item['id']]
            cases.append({'case_id':f'{item["id"]}::{stratum}','source_id':item['id'],'channel':ch,'stratum':stratum,
                'label':int(source.label_fraud),'text':text,'source_sha256':hashlib.sha256(source.text.encode()).hexdigest(),
                'input_sha256':hashlib.sha256(text.encode()).hexdigest(),'transform':condition if edit_list else 'none','edits':edit_list,
                'source_chars':len(source.text),'base_was_clipped':len(source.text)>BASE_CHARS})
        selection.append({'channel':ch,'unused_pool_n':len(pool),'near_boundary_n':sum(near),'near_fallback':fallback,
                          'pair_near_candidates':len(pair_candidates),'pair_fallback':pair_fallback,'stress_condition':condition})
    manifest={'partition':'independent_canonical_seed0_test_challenge','cases':cases,'selection':selection,
              'excluded_previous_base_ids':len(excluded),'excluded_sources':sources,'unique_bases':len({(c['channel'],c['source_id']) for c in cases})}
    (BASE/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    config['project_root']=str(ROOT);config['normalised_boundary_routing']=True
    config['limits'].update(llm_turns=3,tool_calls=2,max_output_tokens=350,max_message_tokens=18000)
    config['pricing']={'input_usd_per_million':.3,'output_usd_per_million':1.2}
    config['router_guidance']='Choose ONE primary classifier tool call. The backend may automatically return a required second classifier if near its calibrated boundary. Do not request another primary. Use normalised text for routing; no score is calibrated fraud risk. Conflicts follow the locked validation policy, with abstention when evidence is insufficient.'
    (BASE/'run_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False));(BASE/'run_config.yaml').chmod(0o600)
    files=['src/agent/boundary.py','src/agent/boundary_routing.py','src/agent/routing.py','src/agent/models.py','src/agent/contract.py',
        'src/agent/schema.json','src/agent/stress.py','src/agent/provider.py','src/agent/budget.py','src/agent/live_meter.py',
        'scripts/prepare_boundary_router.py','scripts/run_boundary_router.py']
    protocol={'design_sha256':file_hash(BASE/'validation_design.json'),'catalog_sha256':file_hash(CATALOG),'manifest_sha256':file_hash(BASE/'manifest.json'),'run_config_sha256':file_hash(BASE/'run_config.yaml'),
        'source_code_sha256':{p:file_hash(ROOT/p) for p in files},'global_token_budget':75000,'global_call_limit':36,'paid_messages':12,
        'same_normalisation_all_modes':True,'agent_role':'primary selection only; mandatory second and conflict policy enforced by backend',
        'provider':'deepseek_chat','model':'deepseek-flash','temperature':0,'thinking':'disabled','repeats':1,
        'stop_policy':'no retries or post-result policy changes; preserve every result and review'}
    (BASE/'protocol.json').write_text(json.dumps(protocol,indent=2))
    import shutil
    for p,h in protocol['source_code_sha256'].items():
        target=BASE/'frozen_source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,target)
    print(json.dumps({'cases':len(cases),'unique_bases':manifest['unique_bases'],'excluded':len(excluded),'selection':selection},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['validation','test']);a=p.parse_args()
    validate_design() if a.command=='validation' else select_test()

if __name__=='__main__':main()
