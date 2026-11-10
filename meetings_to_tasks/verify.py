"""Verify completeness and timestamp consistency of a real pipeline result."""
import argparse
import collections
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='m2t verify', description=__doc__)
    parser.add_argument('job', type=Path, help='result folder of one meeting')
    parser.add_argument('recordings', type=int, nargs='?', help='expected number of recordings')
    args = parser.parse_args(argv)
    root, expected_sources = args.job, args.recordings
    result = json.loads((root/'transcript.json').read_text(encoding='utf-8'))
    if expected_sources is not None:
        assert len(result['sources']) == expected_sources, f'Expected {expected_sources} recordings'
    assert result['words'] and result['turns'], 'Empty transcript'
    assert abs(sum(s['duration'] for s in result['sources'])-result['duration']) < 0.01
    for word in result['words']:
        assert 0 <= word['start'] <= word['end'] <= result['duration']+0.2
        assert word['word'].strip()
    assert all(a['start'] <= b['start']+0.05 for a,b in zip(result['words'], result['words'][1:])), 'Non-monotonic words'
    for source in result['sources']:
        assert any(w['source'] == source['file'] for w in result['words']), 'Recording has no text'
    for file in ['transcript.md', 'summary.md', 'meeting.html']:
        assert (root/file).stat().st_size > 100
    counts = collections.Counter(w['speaker'] for w in result['words'])
    assert len(counts) == result.get('speaker_count',5), 'Unexpected speaker count'
    report = {'duration_minutes': round(result['duration']/60, 2), 'words': len(result['words']), 'turns': len(result['turns']), 'words_by_speaker': dict(counts), 'uncertain_turns': sum(t['uncertain'] for t in result['turns']), 'tasks': len(json.loads((root/'tasks.json').read_text(encoding='utf-8'))), 'checks': 'passed; semantic accuracy and speaker identity are not certified'}
    (root/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

