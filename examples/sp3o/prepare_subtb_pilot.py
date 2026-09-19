"""Prepare the pilot split. Run in a compute allocation on scheduled clusters."""
import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--model', required=True, help='Local Qwen3-4B-Base tokenizer directory')
parser.add_argument('--output', type=Path, required=True, help='New output directory')
args = parser.parse_args()
dest = args.output.resolve()
dest.mkdir(parents=True, exist_ok=False)
source = args.source
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
groups = {}
exact_rows = Counter()
total = duplicates = too_long = 0
for i, line in enumerate(source.open()):
    row = json.loads(line)
    total += 1
    prompt, label = row['prompt'], row['label']
    key = json.dumps([(m['role'], ' '.join(m['content'].split())) for m in prompt], ensure_ascii=False)
    exact_rows[json.dumps(row, sort_keys=True, ensure_ascii=False)] += 1
    groups.setdefault(key, []).append((i, row))

def label_key(label):
    value = str(label).strip()
    try:
        number = Decimal(value)
        if number.is_finite():
            return ('number', number)
    except InvalidOperation:
        pass
    return ('text', value)

rows = []
conflicts = []
for key, members in groups.items():
    if len({label_key(row['label']) for _, row in members}) > 1:
        conflicts.append(dict(prompt_hash=hashlib.sha256(key.encode()).hexdigest(), source_rows=[i for i,_ in members], labels=[row['label'] for _,row in members]))
        continue  # Exclude the entire ambiguous group from BOTH splits.
    i, row = members[0]
    duplicates += len(members)-1
    tokens = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True)
    if len(tokens) > 1024:
        too_long += 1
        continue
    rows.append(dict(prompt=row['prompt'], label=row['label'], metadata=dict(source_row=i, prompt_tokens=len(tokens), prompt_hash=hashlib.sha256(key.encode()).hexdigest())))
(dest/'excluded-conflicts.json').write_text(json.dumps(conflicts, indent=2))
random.Random(1234).shuffle(rows)
evaluation, training = rows[:256], rows[256:]
assert len(training) > 6400 and len(evaluation) == 256
assert not ({r['metadata']['prompt_hash'] for r in training} & {r['metadata']['prompt_hash'] for r in evaluation})
for name, selected in [('train', training), ('eval', evaluation), ('capacity-eval', evaluation[:8])]:
    with (dest/f'{name}.jsonl').open('x') as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
for name, data in [('capacity', 'capacity-eval'), ('effect', 'eval')]:
    (dest/f'{name}.yaml').write_text(f'''eval:
  defaults:
    input_key: prompt
    label_key: label
    max_response_len: 8192
    temperature: 1.0
    top_p: 1.0
  datasets:
    - name: dapo_pilot
      path: {dest}/{data}.jsonl
      n_samples_per_eval_prompt: 4
''')
manifest = dict(seed=1234, source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_rows=total, source_revision=dict(repo='zhuzilin/dapo-math-17k', expected_revision='2e65612930298bde4c5d58fd97b3f23a483aaff9'), exact_unique_rows=len(exact_rows), exact_row_multiplicity_histogram=dict(Counter(exact_rows.values())), normalized_unique_prompts=len(groups), conflicting_prompt_groups=len(conflicts), conflicting_rows_removed=sum(len(c['source_rows']) for c in conflicts), duplicate_rows_removed=duplicates, over_1024_token_rows_removed=too_long, train_rows=len(training), eval_rows=len(evaluation), job_id=os.environ.get('SLURM_JOB_ID'))
(dest/'manifest.json').write_text(json.dumps(manifest, indent=2))
print('PILOT_DATA_READY', manifest, flush=True)
