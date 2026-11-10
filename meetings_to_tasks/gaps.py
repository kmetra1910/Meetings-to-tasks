"""Recover speech that sequential decoding dropped.

Decoding without VAD skips silences through hallucination_silence_threshold, and on far-field recordings it
sometimes skips whole real passages with them. A long gap between recognized segments whose loudness is at the
level of this file's speech is decoded again on its own, without that threshold.
"""
import numpy as np

# Bump when gap recovery changes so cached ASR is recomputed.
VERSION = 1


def segment_record(seg, shift=0.0):
    return {'start': seg.start+shift, 'end': seg.end+shift, 'text': seg.text,
            'words': [{'start': w.start+shift, 'end': w.end+shift, 'word': w.word, 'probability': w.probability} for w in (seg.words or [])]}


def speech_level_gaps(audio, segments, min_gap=8.0, frame=0.5, rate=16000):
    """(start, end) of gaps longer than min_gap where at least half the frames are louder than quiet speech."""
    size = int(frame*rate)
    frames = audio[:len(audio)//size*size].reshape(-1, size)
    level = 10*np.log10((frames**2).mean(1)+1e-10)
    inside = np.zeros(len(level), bool)
    for seg in segments:
        inside[int(seg['start']/frame):int(seg['end']/frame)+1] = True
    if not inside.any():
        return []
    # Quiet end of recognized speech; real silence stays below it.
    floor = np.percentile(level[inside], 10)
    bounds = [0.0] + [t for seg in segments for t in (seg['start'], seg['end'])] + [len(audio)/rate]
    gaps = []
    for start, end in zip(bounds[::2], bounds[1::2]):
        part = level[int(start/frame):int(end/frame)]
        if end-start > min_gap and len(part) and (part > floor).mean() >= 0.5:
            gaps.append((start, end))
    return gaps


def recover_gaps(asr, audio, segments, language, rate=16000):
    """Return the segments with recovered ones added, the number of gaps tried and of segments recovered."""
    gaps, found = speech_level_gaps(audio, segments, rate=rate), []
    for start, end in gaps:
        # Starting exactly at a neighbouring segment's edge can derail decoding into one junk token; step 1 s inside.
        start, end = start+1, end-1
        stream, _ = asr.transcribe(audio[int(start*rate):int(end*rate)], language=language, beam_size=5, word_timestamps=True,
                                   vad_filter=False, condition_on_previous_text=False)
        for seg in stream:
            if seg.avg_logprob > -0.6 and seg.no_speech_prob < 0.6 and seg.text.strip():
                found.append({**segment_record(seg, start), 'recovered': True})
    return sorted(segments+found, key=lambda seg: seg['start']), len(gaps), len(found)
