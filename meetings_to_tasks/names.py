"""Apply human-confirmed speaker names without changing the recognized words."""
import argparse
import json
from pathlib import Path

from . import report, tasks


def main(argv=None):
    parser = argparse.ArgumentParser(prog='m2t names', description=__doc__)
    parser.add_argument('results', type=Path, help='result folder of one meeting')
    parser.add_argument('names', type=Path, help='JSON mapping from SPEAKER_XX to a confirmed name')
    args = parser.parse_args(argv)
    data = json.loads((args.results/'transcript.json').read_text(encoding='utf-8'))
    names = json.loads(args.names.read_text(encoding='utf-8-sig'))
    old_names = data.get('speaker_names', {})
    for item in data['words']+data['turns']:
        item['speaker_name'] = names.get(item['speaker'], item['speaker'])
    data['speaker_names'] = names
    data['identity_confirmation'] = 'Names supplied by a person after listening to the voice samples; turn boundaries remain automatic'
    (args.results/'transcript.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    language = data.get('language')
    (args.results/'transcript.md').write_text(report.transcript_markdown(data['turns'], language, named=True), encoding='utf-8')
    summary_path = args.results/'summary.md'
    summary = summary_path.read_text(encoding='utf-8')
    for speaker, old in old_names.items():
        new = names.get(speaker, old)
        if old != new:
            summary = summary.replace(old, new)
    for speaker, name in names.items():
        summary = summary.replace(speaker, name)
        # The LLM sometimes guesses "Name (SPEAKER_XX)"; once confirmed, keep the name once.
        summary = summary.replace(f'{name} ({name})', name)
    summary_path.write_text(summary, encoding='utf-8')
    tasks.export(args.results)
    title = data.get('title') or (summary.splitlines()[0].lstrip('# ') if summary else 'Meeting')
    (args.results/'meeting.html').write_text(report.render_page(title, summary, data['turns'], language, named=True), encoding='utf-8')
    print('Speaker names applied; recognized words preserved')

