"""Library of confirmed voices: enroll named speakers from a finished job, suggest names for a new one.

The library holds one mean ECAPA embedding per person, built from the 3 s windows the pipeline already saved
(speaker-embeddings.npz), so no audio is re-processed. It lives in voices/ of the working directory (or $M2T_VOICES) and stays out of Git: voice
embeddings identify people.
"""
import argparse
import json
import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

LIBRARY = Path(os.environ.get('M2T_VOICES', 'voices'))/'library.json'
# Optional {"label used in one meeting": "canonical name"}, for the same person named differently across meetings.
ALIASES = LIBRARY.parent/'aliases.json'


def window_speakers(job, min_share=0.8, min_margin=0.08):
    """Label each embedding window with the speaker of the confident words inside it, if one speaker dominates."""
    data = np.load(job/'speaker-embeddings.npz')
    vectors, windows = data['vectors'], data['windows']
    words = json.loads((job/'transcript.json').read_text(encoding='utf-8'))['words']
    mids = np.array([(w['start']+w['end'])/2 for w in words])
    labels = []
    for start, end in windows:
        inside = [words[i] for i in np.nonzero((mids >= start) & (mids <= end))[0] if words[i]['speaker_margin'] >= min_margin]
        speakers = [w['speaker'] for w in inside]
        top = max(set(speakers), key=speakers.count) if speakers else None
        labels.append(top if speakers and speakers.count(top)/len(speakers) >= min_share else None)
    return vectors, labels


def centroids(job, min_windows=5):
    vectors, labels = window_speakers(job)
    result = {}
    for speaker in sorted(set(l for l in labels if l)):
        rows = vectors[[i for i, l in enumerate(labels) if l == speaker]]
        if len(rows) >= min_windows:
            mean = rows.mean(0)
            result[speaker] = (mean/np.linalg.norm(mean), len(rows))
    return result


def load():
    return json.loads(LIBRARY.read_text(encoding='utf-8')) if LIBRARY.exists() else {}


def enroll(job, names_path):
    names = json.loads((names_path or job/'speaker-names.json').read_text(encoding='utf-8-sig'))
    aliases = json.loads(ALIASES.read_text(encoding='utf-8-sig')) if ALIASES.exists() else {}
    library = load()
    for speaker, (vector, count) in centroids(job).items():
        if speaker not in names:
            continue
        name = aliases.get(names[speaker], names[speaker])
        entry = library.get(name)
        if entry and any(s['job'] == job.name and s['speaker'] == speaker for s in entry['sources']):
            print(f'{name}: {job.name} already enrolled')
            continue
        if entry:
            # Window-weighted running mean, so long meetings count more than a few phrases.
            mean = np.array(entry['vector'])*entry['windows'] + vector*count
            entry['vector'] = (mean/np.linalg.norm(mean)).round(6).tolist()
            entry['windows'] += count
        else:
            entry = library[name] = {'vector': vector.round(6).tolist(), 'windows': count, 'sources': []}
        entry['sources'].append({'job': job.name, 'speaker': speaker, 'windows': count, 'enrolled': date.today().isoformat()})
        clips = LIBRARY.parent/name
        clips.mkdir(parents=True, exist_ok=True)
        for clip in sorted((job/'speaker-samples').glob(f'{speaker}-*.wav')):
            shutil.copy2(clip, clips/f'{job.name}-{clip.name}')
        print(f'{name}: +{count} windows from {job.name} ({speaker}), {entry["windows"]} total')
    LIBRARY.parent.mkdir(exist_ok=True)
    LIBRARY.write_text(json.dumps(library, ensure_ascii=False, indent=1), encoding='utf-8')


def match(job, threshold):
    library = load()
    if not library:
        raise SystemExit('Voice library is empty: run "m2t voices enroll" on a job with confirmed names first')
    people = sorted(library)
    known = np.array([library[p]['vector'] for p in people])
    found = centroids(job)
    speakers = sorted(found)
    similarity = np.stack([found[s][0] for s in speakers]) @ known.T
    rows, cols = linear_sum_assignment(-similarity)
    suggested = {}
    print(f'{"voice":<11} {"best match":<22} {"cosine":>6} {"next":>6}  windows')
    for row, col in zip(rows, cols):
        others = np.delete(similarity[row], col)
        runner_up = others.max() if len(others) else float('nan')
        accepted = similarity[row, col] >= threshold
        if accepted:
            suggested[speakers[row]] = people[col]
        print(f'{speakers[row]:<11} {people[col] if accepted else "(unknown)":<22} {similarity[row, col]:6.2f} {runner_up:6.2f}  {found[speakers[row]][1]}')
    for row in set(range(len(speakers))) - set(rows):
        print(f'{speakers[row]:<11} {"(unknown)":<22} {similarity[row].max():6.2f}')
    (job/'speaker-names.suggested.json').write_text(json.dumps(suggested, ensure_ascii=False, indent=4), encoding='utf-8')
    print(f'Suggested names written to {job/"speaker-names.suggested.json"}; listen to the clips before applying them')
    return suggested


def main(argv=None):
    parser = argparse.ArgumentParser(prog='m2t voices', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('enroll', help='add the confirmed speakers of a job to the library')
    e.add_argument('job', type=Path)
    e.add_argument('--names', type=Path, help='defaults to <job>/speaker-names.json')
    m = sub.add_parser('match', help='suggest names for a new job from the library')
    m.add_argument('job', type=Path)
    m.add_argument('--threshold', type=float, default=0.5, help='minimum cosine to accept a match')
    args = parser.parse_args(argv)
    if args.command == 'enroll':
        enroll(args.job, args.names)
    else:
        match(args.job, args.threshold)

