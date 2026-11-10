# Meetings-to-tasks

Turn meeting recordings into a timecoded transcript with separated voices, minutes, and a task list that a person
or a coding agent can check and act on. Everything runs on your own hardware: Whisper for speech, an ECAPA voice
model for speakers, and any OpenAI-compatible local LLM for the minutes. No paid API is needed.

The repository is built to be operated together with a coding agent (Claude Code, Codex, Cursor and the like).
[AGENTS.md](AGENTS.md) is the agent's runbook: how to run a job, name the speakers with a person, check the
minutes against the transcript and hand over the tasks.

## Repository layout

```
meetings_to_tasks/      the package; one module per step, all behind the m2t command
  cli.py                m2t run | names | tasks | voices | verify
  pipeline.py           speech recognition, voice embeddings, diarization, minutes
  gaps.py               re-decodes speech that sequential decoding dropped
  speakers.py           speaker-count estimate (NME-SC) and merging of incoherent groups
  report.py             transcript, minutes and HTML page; interface strings per language
  tasks.py              task table of the minutes -> tasks.json, tasks.md
  names.py              applies confirmed speaker names everywhere
  voices.py             local library of confirmed voices
  verify.py             completeness and timestamp checks of a job
scripts/run.ps1         runs a job in Docker on a local or remote GPU host
docs/outputs.md         every output file and its fields
docs/review.md          how to check the minutes against the transcript
tests/                  offline tests: no GPU, audio, weights or LLM
examples/               sample input files
AGENTS.md, CLAUDE.md    instructions for coding agents
.claude/skills/         the same procedure as a Claude Code skill
```

## How it works

1. **Speech.** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) large-v3 with word timestamps. By default
   the whole recording is decoded without VAD: on a phone recording of a room meeting, Silero VAD at 0.5 kept 37 s of
   330 s and dropped 14 minutes of talk. `--vad-threshold 0.5` restores fast batched decoding for clean call recordings.
2. **Dropped passages.** Decoding without VAD can skip a real passage together with a silence. Long gaps between
   segments that are as loud as speech are decoded again on their own; what comes back is marked `recovered` and its
   turns `uncertain`.
3. **Voices.** [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) embeddings of
   overlapping 3 s windows, then spectral clustering. The number of speakers is estimated with the NME-SC eigengap
   (Park et al., 2019), then groups with silhouette below 0.15, usually overlapping speech and noise, are merged away.
   `--speakers N` fixes the count.
4. **Minutes.** The transcript goes to the LLM in chunks; notes per chunk, then one set of minutes. The prompts forbid
   inventing owners and deadlines and require a timecode per item. Russian and English prompts are built in; other
   languages use the English prompts and ask for minutes in the meeting's language.
5. **Names.** Voices stay `SPEAKER_XX` until a person confirms who is who from the samples. A local library of
   confirmed voices suggests names for the next meeting.

Each job writes `results/<job>/`: `meeting.html` (minutes and searchable transcript), `summary.md`, `tasks.json` and
`tasks.md`, `transcript.md`, `transcript.json` and `speaker-samples/`. [docs/outputs.md](docs/outputs.md) describes
every file. ASR and per-chunk notes are cached, so an interrupted job resumes where it stopped.

## Install

Python 3.10 or newer and FFmpeg. For speech recognition also PyTorch and torchaudio (2.4 with CUDA 12.1 and cuDNN 9
is tested; install them from [pytorch.org](https://pytorch.org) for your platform).

```bash
pip install -e ".[asr]"
```

Without `[asr]` you get everything except `m2t run`: naming speakers, exporting tasks, the voice library, checks and
the tests. The [Dockerfile](Dockerfile) is the reference environment for `m2t run`.

```bash
python -m unittest discover tests
```

## Run

You need an OpenAI-compatible LLM endpoint for the minutes, for example a
[llama.cpp server](https://github.com/ggml-org/llama.cpp) with a Qwen model.

On a machine with the `[asr]` dependencies (`--device cpu` works, slowly):

```bash
m2t run part1.flac part2.flac --out results/weekly-sync --title "Weekly sync" --llm-url http://localhost:8080/v1
```

On a GPU host through Docker, from Windows PowerShell. `scripts/run.ps1` converts the recordings to lossless 16 kHz
mono FLAC, picks an idle GPU, uploads with a SHA256 check, runs the job in its own container and copies the results
back to `results\<job>`:

```powershell
./scripts/run.ps1 -Files 'C:\rec\part1.webm','C:\rec\part2.webm' -Title 'Weekly sync' -DockerContext gpu-host -LlmModel /models/qwen.gguf
```

| Parameter | Meaning |
|---|---|
| `-DockerContext` | Docker context of the GPU host; a remote TCP context must use verified TLS |
| `-AllowedGpus` | GPU UUIDs the job may use; empty allows any idle GPU |
| `-LlmModel` | GGUF path on the host; starts a llama.cpp server on the same GPU, reachable only from the job |
| `-LlmUrl` | existing endpoint, used when `-LlmModel` is empty |
| `-Speakers`, `-MaxSpeakers` | fixed or upper-bounded number of voices |
| `-Language` | `en`, `ru`, …; detected from the first recording when empty |

Pass the files of one meeting in chronological order. The meeting timecode is the sum of file durations, without
the breaks between recordings; `sources` in `transcript.json` holds the file boundaries.

### After the run

```bash
m2t verify results/<job> 2                                   # 2 = number of recordings
m2t run --recluster-only --speakers 6 --out results/<job>    # wrong voice count: recluster without ASR
m2t names results/<job> results/<job>/speaker-names.json     # confirmed names, see examples/
m2t voices enroll results/<job>                              # remember the confirmed voices
m2t voices match results/<new-job>                           # suggest names for a new meeting
m2t run --summary-only --out results/<job>                   # rewrite the minutes only
m2t tasks results/<job>                                      # re-export tasks after editing summary.md
```

`scripts/run.ps1` runs `voices match` itself when the library exists and writes `speaker-names.suggested.json`.
A suggestion needs cosine similarity of at least 0.5 (`--threshold`).

## Privacy

Recordings, results and `voices/` identify people and are ignored by Git. Do not commit them, and do not publish
minutes without the participants' consent. The voice library lives in `voices/` of the working directory, or in
`$M2T_VOICES`. `voices/aliases.json` (optional) maps different labels of one person to one name.

## Limitations

- `SPEAKER_XX` are automatic voice groups, not identities. The speaker-count estimate is a heuristic: someone who
  said two phrases can merge into a similar voice, and one person on two microphones can split in two. The silhouette
  threshold was tuned on a handful of meetings; diagnostics are in `transcript.json` → `speaker_estimation`.
- Overlapping speech is not separated. Short remarks and similar voices can be attributed wrongly. `speaker_margin`
  is a heuristic, not a probability.
- The minutes are LLM output. Owners, deadlines and terms must be checked against the transcript at the cited
  timecodes before anyone acts on them; [docs/review.md](docs/review.md) describes that review.

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free to use, change and share for any noncommercial purpose, including
personal use, research, education and work in public-interest, charitable or government organizations. Commercial
use is not permitted; for it, contact the author. This is a source-available license, not an OSI open-source one.

The components keep their own licenses: faster-whisper (MIT), SpeechBrain (Apache-2.0), scikit-learn (BSD-3-Clause). Model weights you download (Whisper, ECAPA, the LLM) come under their own terms.
