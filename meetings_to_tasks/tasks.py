"""Export the task table of summary.md as tasks.json and tasks.md, for review and for creating issues.

The rows come from the LLM minutes: owners, due dates and timecodes must be checked against the transcript
before anyone acts on them (see docs/review.md).
"""
import argparse
import json
import re
from pathlib import Path

from .report import table_rows

FIELDS = ['task', 'owner', 'due', 'timecode']


def parse(summary):
    tasks = []
    for row in table_rows(summary):
        task = dict(zip(FIELDS, row))
        # Unconfirmed SPEAKER_XX owners are automatic labels, not people.
        task['owner_confirmed'] = not re.search(r'SPEAKER_\d+', task['owner'])
        tasks.append(task)
    return tasks


def export(job):
    tasks = parse((job/'summary.md').read_text(encoding='utf-8'))
    (job/'tasks.json').write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = [f"| {t['task']} | {t['owner']} | {t['due']} | {t['timecode']} |" for t in tasks]
    (job/'tasks.md').write_text('| Task | Owner | Due | Timecode |\n|---|---|---|---|\n' + '\n'.join(rows) + '\n', encoding='utf-8')
    return tasks


def main(argv=None):
    parser = argparse.ArgumentParser(prog='m2t tasks', description=__doc__)
    parser.add_argument('job', type=Path, help='result folder with summary.md')
    tasks = export(parser.parse_args(argv).job)
    print(f'{len(tasks)} tasks exported, {sum(not t["owner_confirmed"] for t in tasks)} with an unconfirmed owner')

