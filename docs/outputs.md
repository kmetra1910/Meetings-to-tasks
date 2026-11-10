# Output files

A job writes everything to its result folder, `results/<job>/`. All text files are UTF-8.

| File | Written by | Contents |
|---|---|---|
| `meeting.html` | `run`, `names` | minutes and a searchable transcript; opens without a server |
| `summary.md` | `run`, `names` | minutes: summary, decisions, tasks table, open questions |
| `tasks.json`, `tasks.md` | `run`, `names`, `tasks` | the task table of `summary.md` |
| `transcript.md` | `run`, `names` | one line per turn |
| `transcript.json` | `run`, `names` | words, turns, sources, speaker diagnostics |
| `speaker-samples/` | `run` | `SPEAKER_XX-1.wav`, `-2.wav` per voice and `samples.json` |
| `speaker-names.suggested.json` | `voices match` | names suggested from the voice library |
| `speaker-names.json` | you | confirmed names, the input of `names` and `voices enroll` |
| `validation.json` | `verify` | counts and the result of the checks |
| `speaker-embeddings.npz` | `run` | `vectors` (normalized ECAPA embeddings), `windows` (start, end in seconds) |
| `asr-N.json`, `notes-N.json` | `run` | caches of recording N and of minutes chunk N; safe to delete |

## transcript.md

```
[00:09:57–00:10:21] Alex (SPEAKER_01): I'll finish the adapters by Tuesday.
```

Start and end are meeting time (`HH:MM:SS`), then the name (or the label while no name is confirmed) and the label.

## transcript.json

| Field | Meaning |
|---|---|
| `title`, `language` | as given or detected |
| `duration` | seconds, the sum of all recordings |
| `sources[]` | `file`, `sha256`, `duration`, `offset` (where the recording starts in meeting time) |
| `speaker_count` | number of voice groups |
| `speaker_estimation` | `mode` (`automatic` or `fixed`), `estimated`, `chosen`, eigengap candidates, silhouette per group |
| `speaker_names` | `SPEAKER_XX` → name, after `names` |
| `reference_matches` | cosine per reference speaker when `--speaker-references` was used |
| `words[]` | `word`, `start`, `end`, `probability`, `speaker`, `speaker_name`, `speaker_margin`, `source`, `source_start`, `recovered` |
| `turns[]` | `start`, `end`, `speaker`, `speaker_name`, `text`, `source`, `source_start`, `uncertain` |

`start` and `end` are meeting time; `source_start` is the time inside the source recording, for finding the place
in the original file. A turn is `uncertain` when a word in it has `speaker_margin` below 0.08, recognition
probability below 0.5, or came from gap recovery.

## tasks.json

```json
[
  {"task": "Finish the adapters", "owner": "Alex", "due": "Tue 03.11", "timecode": "00:09:57–00:13:26", "owner_confirmed": true}
]
```

`owner_confirmed` is false while the owner is still a `SPEAKER_XX` label. The rows are LLM output until reviewed
([review.md](review.md)).

## speaker-names.json

```json
{"SPEAKER_01": "Alex", "SPEAKER_02": "Sam", "SPEAKER_03": "Alex"}
```

Two labels may map to one person, for example the same voice in a noisy stretch. Labels left out keep `SPEAKER_XX`.

## speaker-samples/samples.json

One entry per clip: `speaker`, `clip`, `source`, `source_start_seconds`, `meeting_start_seconds`,
`duration_seconds`, `text`. The file is also valid input for `--speaker-references`, which pins those labels to
the same voices when a job is reclustered.
