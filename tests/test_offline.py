"""Tests that need no GPU, audio, model weights or LLM: python -m unittest discover tests"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meetings_to_tasks import cli, gaps, report, speakers, tasks, voices  # noqa: E402

SUMMARY = """# Weekly sync

**Summary**
1. The team reviewed the release plan (00:01:10).

**Tasks**
| Task | Owner | Deadline | Timecode |
|---|---|---|---|
| Update the onboarding guide | SPEAKER_01 | Tue | 00:09:57 |
| Prepare the demo | SPEAKER_10 | not set | 00:42:39 |

**Open questions**
* Which release date to choose?
"""


def voice_vectors(speakers, per_speaker, dim=192, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(speakers, dim))
    x = np.concatenate([c + noise*np.linalg.norm(c)/np.sqrt(dim)*rng.normal(size=(per_speaker, dim)) for c in centers])
    return x/np.linalg.norm(x, axis=1, keepdims=True), np.repeat(np.arange(speakers), per_speaker)


def make_job(folder, speakers=3, per_speaker=40):
    """A finished job: one embedding window per 1.5 s, each covered by two confident words of one speaker."""
    x, truth = voice_vectors(speakers, per_speaker)
    order = np.random.default_rng(1).permutation(len(x))
    x, truth = x[order], truth[order]
    windows = [(i*1.5, i*1.5+1.5) for i in range(len(x))]
    words = []
    for (start, end), who in zip(windows, truth):
        for k in range(2):
            words.append({'start': start+0.2+k*0.6, 'end': start+0.6+k*0.6, 'word': ' word', 'probability': 0.9,
                          'speaker': f'SPEAKER_{who+1:02}', 'speaker_name': f'SPEAKER_{who+1:02}', 'speaker_margin': 0.3,
                          'source': 'part1.flac', 'source_start': start})
    turns = [{'start': w['start'], 'end': w['end'], 'source': w['source'], 'source_start': w['source_start'], 'speaker': w['speaker'],
              'speaker_name': w['speaker_name'], 'text': 'word', 'uncertain': False} for w in words[::2]]
    duration = windows[-1][1]
    np.savez_compressed(folder/'speaker-embeddings.npz', vectors=x, windows=np.array(windows))
    data = {'title': 'Weekly sync', 'language': 'en', 'sources': [{'file': 'part1.flac', 'duration': duration, 'offset': 0}],
            'duration': duration, 'speaker_count': speakers, 'speaker_names': {}, 'words': words, 'turns': turns}
    (folder/'transcript.json').write_text(json.dumps(data), encoding='utf-8')
    (folder/'summary.md').write_text(SUMMARY, encoding='utf-8')
    (folder/'transcript.md').write_text(report.transcript_markdown(turns, 'en'), encoding='utf-8')
    (folder/'meeting.html').write_text(report.render_page('Weekly sync', SUMMARY, turns, 'en'), encoding='utf-8')
    tasks.export(folder)
    (folder/'speaker-samples').mkdir()
    return data


class SpeakerCount(unittest.TestCase):
    def test_estimates_the_number_of_voices(self):
        for count in (3, 5):
            x, _ = voice_vectors(count, 60, seed=count)
            labels, info = speakers.diarize_auto(x, max_speakers=10)
            self.assertEqual(info['chosen'], count)
            self.assertEqual(len(set(labels)), count)


class GapRecovery(unittest.TestCase):
    def audio(self):
        rng = np.random.default_rng(0)
        audio = 0.1*rng.normal(size=60*16000).astype(np.float32)
        audio[40*16000:52*16000] *= 0.001  # a real pause
        return audio

    def test_finds_only_loud_gaps(self):
        segments = [{'start': 0, 'end': 10}, {'start': 30, 'end': 40}, {'start': 52, 'end': 60}]
        self.assertEqual(gaps.speech_level_gaps(self.audio(), segments), [(10, 30)])

    def test_recovered_segments_are_shifted_and_marked(self):
        word = types.SimpleNamespace(start=0.5, end=1.0, word=' found', probability=0.9)
        good = types.SimpleNamespace(start=0.5, end=3.0, text=' found', words=[word], avg_logprob=-0.3, no_speech_prob=0.1)
        junk = types.SimpleNamespace(start=4.0, end=5.0, text=' ...', words=[], avg_logprob=-1.5, no_speech_prob=0.9)
        asr = mock.Mock()
        asr.transcribe.return_value = ([good, junk], None)
        segments = [{'start': 0, 'end': 10}, {'start': 30, 'end': 40}, {'start': 52, 'end': 60}]
        merged, tried, recovered = gaps.recover_gaps(asr, self.audio(), segments, 'en')
        self.assertEqual((tried, recovered), (1, 1))
        found = [s for s in merged if s.get('recovered')][0]
        self.assertEqual(found['start'], 11.5)  # decoding starts 1 s inside the gap
        self.assertEqual(found['words'][0]['start'], 11.5)


class Tasks(unittest.TestCase):
    def test_parses_the_task_table_in_any_language(self):
        parsed = tasks.parse(SUMMARY)
        self.assertEqual([t['task'] for t in parsed], ['Update the onboarding guide', 'Prepare the demo'])
        self.assertFalse(parsed[0]['owner_confirmed'])
        russian = SUMMARY.replace('| Task | Owner | Deadline | Timecode |', '| Задача | Ответственный | Срок | Таймкод |')
        self.assertEqual(tasks.parse(russian), parsed)

    def test_minutes_render_tasks_as_a_list(self):
        page = report.render_summary(SUMMARY, 'en')
        self.assertIn('<strong>Update the onboarding guide</strong>', page)
        self.assertNotIn('Deadline', page)


class Naming(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.job = Path(self.tmp.name)/'job1'
        self.job.mkdir()
        make_job(self.job)

    def tearDown(self):
        self.tmp.cleanup()

    def test_apply_names_keeps_words_and_renames_owners(self):
        names = self.job/'speaker-names.json'
        names.write_text(json.dumps({'SPEAKER_01': 'Alex', 'SPEAKER_10': 'Sam'}), encoding='utf-8')
        before = json.loads((self.job/'transcript.json').read_text(encoding='utf-8'))['words']
        subprocess.run([sys.executable, '-m', 'meetings_to_tasks', 'names', str(self.job), str(names)], cwd=ROOT, check=True, capture_output=True)
        after = json.loads((self.job/'transcript.json').read_text(encoding='utf-8'))['words']
        self.assertEqual([w['word'] for w in before], [w['word'] for w in after])
        exported = json.loads((self.job/'tasks.json').read_text(encoding='utf-8'))
        self.assertEqual([t['owner'] for t in exported], ['Alex', 'Sam'])
        self.assertTrue(all(t['owner_confirmed'] for t in exported))

    def test_voice_library_recognizes_enrolled_voices(self):
        library = Path(self.tmp.name)/'voices'/'library.json'
        with mock.patch.object(voices, 'LIBRARY', library), mock.patch.object(voices, 'ALIASES', library.parent/'aliases.json'):
            names = {'SPEAKER_01': 'Alex', 'SPEAKER_02': 'Sam', 'SPEAKER_03': 'Kim'}
            (self.job/'speaker-names.json').write_text(json.dumps(names), encoding='utf-8')
            voices.enroll(self.job, None)
            self.assertEqual(voices.match(self.job, 0.5), names)

    def test_verify_passes_on_a_complete_job(self):
        run = subprocess.run([sys.executable, '-m', 'meetings_to_tasks', 'verify', str(self.job), '1'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


class Cli(unittest.TestCase):
    def test_lists_commands_and_rejects_unknown_ones(self):
        with mock.patch('sys.stdout'), self.assertRaises(SystemExit) as stop:
            cli.main(['transcribe'])
        self.assertEqual(stop.exception.code, 2)

    def test_tasks_command_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job/'summary.md').write_text(SUMMARY, encoding='utf-8')
            with mock.patch('sys.stdout'):
                cli.main(['tasks', str(job)])
            self.assertEqual(len(json.loads((job/'tasks.json').read_text(encoding='utf-8'))), 2)


if __name__ == '__main__':
    unittest.main()
