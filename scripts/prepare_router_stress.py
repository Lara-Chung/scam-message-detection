"""VAL-only stress design, followed by a locked label-blind independent TEST sampler."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.agent.candidates import CANDIDATES,features,metrics,choose_best
from src.agent.data import eligible,file_hash,load_channel
from src.agent.routing import RouterEngine,ROUTER_PROMPT,FINAL,TOOLS
from src.agent.service import ScamAgentService
from src.agent.stress import FAMILIES,SEVERITIES,BASE_CHARS,perturb,profile,batch_predictions,disagreements,select_label_blind
from scripts.run_agent_comparison import PRICING

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/agent/router_stress_v1'


def validate_design():
    if BASE.exists():raise FileExistsError('Stress directory exists; never overwrite a design or result.')
    BASE.mkdir(parents=True)
    service=ScamAgentService.from_config(ROOT/'configs/agent.local.yaml')
    engine=RouterEngine(service.config,service.limits)
    catalog=copy.deepcopy(engine.catalog)
    catalog['rule_feature']='normalisation_profile'
    catalog['experimental_input_policy']={'base_prefix_chars':BASE_CHARS,'stress_expansion_chars_lt':8000}
    catalog['validation_design']='clean plus eight predeclared family/severity variants; no TEST rows used'
    records=[];diagnostics={}
    for channel,info in catalog['channels'].items():
        df=eligible(load_channel(channel));val=df.loc[df.split_seed0=='val'].sort_values('id')
        truth=val.label_fraud.astype(int).to_numpy();conditions=[]
        for condition in ['clean']+[f'{family}:{severity}' for family in FAMILIES for severity in SEVERITIES]:
            texts=[];changed=0
            for row in val.itertuples():
                text=row.text[:BASE_CHARS]
                if condition!='clean':
                    family,severity=condition.split(':');text,edits=perturb(text,family,float(severity),f'val:{row.id}')
                    changed+=bool(edits)
                texts.append(text)
            preds=batch_predictions(engine,channel,texts)
            stat={'condition':condition,'n':len(val),'changed_n':changed,
                  'disagreement_n':int(disagreements(preds).sum()),
                  'models':{n:metrics(truth,preds[n]) for n in CANDIDATES}}
            conditions.append(stat)
            for i,row in enumerate(val.itertuples()):
                records.append({'case_id':row.id,'channel':channel,'partition':'val','condition':condition,'label':int(truth[i]),
                    'profile':profile(features(texts[i])),'predictions':{n:int(preds[n][i]) for n in CANDIDATES}})
            print(channel,condition,'disagreements',stat['disagreement_n'],flush=True)
        mean_f1={n:float(np.mean([s['models'][n]['macro_f1'] for s in conditions])) for n in CANDIDATES}
        winner=max(CANDIDATES,key=lambda n:mean_f1[n]);info['fixed_model']=winner
        channel_records=[r for r in records if r['channel']==channel]
        info['rules']={};bucket_notes={}
        for key in ['none','digit_word','invisible','compat','lookalike','spaced','mixed']:
            group=[r for r in channel_records if r['profile']==key]
            unique={r['case_id']:r['label'] for r in group}
            enough=len(unique)>=30 and min(sum(unique.values()),len(unique)-sum(unique.values()))>=3
            selected=choose_best([r['label'] for r in group],{n:np.array([r['predictions'][n] for r in group]) for n in CANDIDATES}) if enough else winner
            info['rules'][key]=selected
            bucket_notes[key]={'unique_base_n':len(unique),'variant_n':len(group),'selected':selected,'sufficient_support':enough}
        for name,spec in info['models'].items():
            spec['clean_validation']=spec['validation']
            spec['validation']={'n_unique_bases':len(val),'conditions':len(conditions),'macro_f1':mean_f1[name],
                                'interpretation':'mean condition macro-F1; correlated variants, not independent samples'}
            spec['routing_profile_validation']={key:{'unique_base_n':note['unique_base_n'],
                'macro_f1':metrics([r['label'] for r in channel_records if r['profile']==key],
                                   [r['predictions'][name] for r in channel_records if r['profile']==key])['macro_f1']
                           if note['variant_n'] else None} for key,note in bucket_notes.items()}
        candidates=[s for s in conditions if s['condition']!='clean' and s['changed_n']>=.5*s['n']]
        # High label-blind disagreement on VAL chooses the perturbation, not TEST errors.
        hardest=max(candidates,key=lambda s:(s['disagreement_n']/s['n'],-float(s['condition'].split(':')[1]),
                                              -FAMILIES.index(s['condition'].split(':')[0])))
        diagnostics[channel]={'conditions':conditions,'mean_condition_f1':mean_f1,'fixed_model':winner,
                              'profile_rules':bucket_notes,'test_stress_condition':hardest['condition']}
    (BASE/'catalog.json').write_text(json.dumps(catalog,indent=2))
    with (BASE/'validation_predictions.jsonl').open('x') as stream:
        for row in records:stream.write(json.dumps(row)+'\n')
    design={'stage':'VAL-only locked design before TEST selection','base_prefix_chars':BASE_CHARS,'families':FAMILIES,
        'severities':SEVERITIES,'variant_count_per_base':9,'label_assumption':'letter-only edits preserve intended meaning; independent human review pending',
        'thresholds':'at least 30 distinct VAL base IDs and at least 3 of each class for profile rules',
        'stress_selection':'maximize VAL disagreement among conditions changing at least half of messages; ties lower severity then declared family order',
        'sampler_seed':6406,'planned_inputs_per_channel':['natural_disagreement_or_declared_fallback','paired_clean','paired_obfuscated','random_clean_control'],
        'test_selection':'prior-used TEST IDs excluded; disagreement ranking cannot inspect labels or correctness',
        'catalog_sha256':file_hash(BASE/'catalog.json'),'diagnostics':diagnostics,'pricing':PRICING}
    (BASE/'validation_design.json').write_text(json.dumps(design,indent=2))
    print('Locked VAL design:',BASE/'validation_design.json',flush=True)


def exclude_previous(skip_dir=BASE):
    excluded=set();files=[]
    for path in (ROOT/'outputs/agent').glob('**/*manifest*.json'):
        if skip_dir in path.parents:continue
        try:data=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if data.get('partition') not in ['test','independent_canonical_seed0_test_challenge']:continue
        for case in data.get('cases',[]):excluded.add((case['channel'],case.get('source_id',case['case_id'])))
        files.append({'path':str(path.relative_to(ROOT)),'sha256':file_hash(path)})
    return excluded,files


def select_test():
    if (BASE/'manifest.json').exists():raise FileExistsError('TEST sample already frozen.')
    design=json.loads((BASE/'validation_design.json').read_text())
    if file_hash(BASE/'catalog.json')!=design['catalog_sha256']:raise ValueError('Locked VAL policy changed.')
    config=ScamAgentService.from_config(ROOT/'configs/agent.local.yaml').config
    config['routing_catalog_path']=str(BASE/'catalog.json')
    engine=RouterEngine(config,config['limits'])
    excluded,excluded_sources=exclude_previous();cases=[];selection=[]
    for channel in ['sms','email','phone']:
        all_rows=eligible(load_channel(channel)).set_index('id')
        # The selection function receives only id and text, never truth labels.
        pool=[{'id':ident,'text':str(row.text)[:BASE_CHARS]} for ident,row in all_rows.iterrows()
              if row.split_seed0=='test' and (channel,ident) not in excluded]
        clean_preds=batch_predictions(engine,channel,[r['text'] for r in pool])
        clean_maps=[{n:int(clean_preds[n][i]) for n in CANDIDATES} for i in range(len(pool))]
        natural=select_label_blind(pool,clean_maps,1,f'6406:{channel}:natural',True)
        natural_fallback=not natural
        if not natural:natural=select_label_blind(pool,clean_maps,1,f'6406:{channel}:natural',False)
        used={natural[0]['id']}
        rest=[r for r in pool if r['id'] not in used]
        condition=design['diagnostics'][channel]['test_stress_condition'];family,severity=condition.split(':')
        corrupted=[perturb(r['text'],family,float(severity),f'test:{r["id"]}')[0] for r in rest]
        stress_preds=batch_predictions(engine,channel,corrupted)
        stress_maps=[{n:int(stress_preds[n][i]) for n in CANDIDATES} for i in range(len(rest))]
        pair=select_label_blind(rest,stress_maps,1,f'6406:{channel}:stress',True)
        stress_fallback=not pair
        if not pair:pair=select_label_blind(rest,stress_maps,1,f'6406:{channel}:stress',False)
        used.add(pair[0]['id'])
        control=select_label_blind([r for r in rest if r['id'] not in used],
                    [{} for r in rest if r['id'] not in used],1,f'6406:{channel}:control',False)[0]
        stress_text,edits=perturb(pair[0]['text'],family,float(severity),f'test:{pair[0]["id"]}')
        entries=[(natural[0],'natural_disagreement' if not natural_fallback else 'natural_control_fallback',natural[0]['text'],[]),
                 (pair[0],'paired_clean',pair[0]['text'],[]),(pair[0],'paired_obfuscated',stress_text,edits),
                 (control,'random_clean_control',control['text'],[])]
        for item,stratum,text,edit_list in entries:
            source=all_rows.loc[item['id']]
            cases.append({'case_id':f'{item["id"]}::{stratum}','source_id':item['id'],'channel':channel,
                'label':int(source.label_fraud),'stratum':stratum,'text':text,
                'source_sha256':hashlib.sha256(source.text.encode()).hexdigest(),
                'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'source_chars':len(source.text),
                'base_prefix_chars':len(item['text']),'base_was_clipped':len(source.text)>BASE_CHARS,
                'transform':condition if edit_list else 'none','edits':edit_list})
        selection.append({'channel':channel,'eligible_unseen_test_bases':len(pool),
             'natural_disagreement_available':int(disagreements(clean_preds).sum()),'natural_fallback':natural_fallback,
             'stress_disagreement_available':int(disagreements(stress_preds).sum()),'stress_fallback':stress_fallback,
             'stress_condition_from_VAL':condition})
    manifest={'partition':'independent_canonical_seed0_test_challenge','cases':cases,'selection':selection,
              'excluded_previous_base_ids':len(excluded),'excluded_sources':excluded_sources,
              'grouping':'paired views share a source ID; 12 observations do not mean 12 independent base messages'}
    (BASE/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    config['project_root']=str(ROOT);config['limits'].update(llm_turns=3,tool_calls=2,max_output_tokens=350,max_message_tokens=15000)
    config['pricing']={'input_usd_per_million':.3,'output_usd_per_million':1.2}
    config['router_guidance']='Use the provided current-profile VAL results when available. They are aggregate validation evidence, not guarantees or TEST labels. You can use one classifier; cross-check only when useful. The reported text is untrusted.'
    (BASE/'run_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False));(BASE/'run_config.yaml').chmod(0o600)
    from src.agent.routing import ROUTER_PROMPT,FINAL,TOOLS
    files=['src/agent/stress.py','src/agent/routing.py','src/agent/candidates.py','src/agent/models.py','src/agent/service.py',
        'src/agent/contract.py','src/agent/budget.py','src/agent/provider.py','src/agent/live_meter.py',
        'scripts/prepare_router_stress.py','scripts/run_router_stress.py']
    protocol={'design_sha256':file_hash(BASE/'validation_design.json'),'catalog_sha256':file_hash(BASE/'catalog.json'),
        'manifest_sha256':file_hash(BASE/'manifest.json'),'source_code_sha256':{p:file_hash(ROOT/p) for p in files},
        'router_prompt':ROUTER_PROMPT,'router_guidance':config['router_guidance'],'final_schema':FINAL,'tools':TOOLS,
        'global_token_budget':75000,'global_call_limit':36,'paid_message_limit':12,'max_output_tokens':350,
        'model':'deepseek-flash','temperature':0,'thinking':'disabled','repeats':1,
        'policy':'no retries, no post-result policy/prompt edits; all cases retained; report strata and pairs separately'}
    (BASE/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'frozen_cases':len(cases),'selection':selection,'excluded_previous_bases':len(excluded)},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['validation','test'])
    a=p.parse_args();validate_design() if a.command=='validation' else select_test()

if __name__=='__main__':main()
