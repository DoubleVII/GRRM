"""Load flat candidate rows, with compatibility for legacy sampled Parquet data."""
from pathlib import Path
from utils.candidate_error_protocol import read_jsonl, identity


def load_items(input_file, data_id=None, candidate_model='provided', require_reference=False,
               src_lang=None, trg_lang=None, max_samples=0):
    if type(max_samples) is not int or max_samples<0:
        raise ValueError('max_samples must be a nonnegative number of source rows')
    path=Path(input_file)
    if path.suffix=='.jsonl':
        items=read_jsonl(path)
        if max_samples:
            keys=list(dict.fromkeys((x['data_id'],x['row_id']) for x in items))[:max_samples]
            items=[x for x in items if (x['data_id'],x['row_id']) in keys]
    elif path.suffix=='.parquet':
        import pandas as pd
        frame=pd.read_parquet(path)
        if max_samples:
            frame=frame.head(max_samples)
        items=[]
        for row_id,row in enumerate(frame.to_dict('records')):
            if 'candidate' in row:
                candidates=[row.get('candidate')]
            else:
                # Legacy datasets store multiple candidates in each source row.
                candidates=row.get('mt_sampling_text')
                if candidates is None:
                    candidates=[None]
            for ci,candidate in enumerate(candidates):
                flat = 'candidate' in row
                item=dict(data_id=data_id or (row.get('data_id', path.stem) if flat else path.stem),
                          row_id=row.get('row_id', row_id) if flat else row_id,
                          candidate_model=row.get('candidate_model', candidate_model) if flat else candidate_model,
                          candidate_index=row.get('candidate_index', ci) if flat else ci,
                          src_text=row.get('src_text'),candidate=candidate,
                          src_lang=row.get('src_lang') or src_lang,trg_lang=row.get('trg_lang') or trg_lang)
                if require_reference:
                    item['ref_text']=row.get('ref_text',row.get('trg_text'))
                    if isinstance(row.get('comment'),str):
                        item['comment']=row['comment']
                if row.get('ltb_id') is not None:
                    item['ltb_id']=row['ltb_id']
                for key in ('split','cohort','family','cluster'):
                    if isinstance(row.get(key),str):
                        item[key]=row[key]
                if flat:
                    for key in ('source_id', 'candidate_multiplicity', 'original_candidate_count',
                                'unique_candidate_count'):
                        if key in row:
                            item[key] = row[key]
                    if 'candidate_indices' in row:
                        item['candidate_indices'] = list(row['candidate_indices'])
                items.append(item)
    else:
        raise ValueError('Use .jsonl or .parquet input')
    for item in items:
        for key in ('src_text','candidate','src_lang','trg_lang')+(('ref_text',) if require_reference else ()):
            if key=='candidate' and not isinstance(item.get(key),str):
                # Preserve missing candidate identities; inference records a failure,
                # rather than treating absence as a correct or empty translation.
                item[key]=None
                continue
            if not isinstance(item.get(key),str) or (key!='candidate' and not item[key].strip()):
                raise ValueError(f'Invalid {key} for candidate {item.get("row_id")}; null candidates must be audited upstream')
    if len({identity(x) for x in items})!=len(items):
        raise ValueError('Duplicate input identities')
    return items
