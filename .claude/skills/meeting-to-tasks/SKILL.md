---
name: meeting-to-tasks
description: Turn meeting recordings into a verbatim transcript, checked minutes and a task list with this repository. Use when the user gives recordings of a meeting, asks to process or transcribe a call, to name speakers, to check minutes against a transcript, or to hand meeting tasks over to a notes repository or issue tracker.
---

# Meeting to tasks

Follow AGENTS.md; this is the short form.

1. **Collect inputs**: recordings in chronological order, meeting date, title, number of participants if known,
   and where the result should go. Ask for what is missing; the date is needed to resolve deadlines.
2. **Run** `m2t run` (or `scripts/run.ps1` on a Docker GPU host) in the background, then `m2t verify results/<job> <n>`.
3. **Voice count**: compare `speaker_count` in `transcript.json` with the participants; recluster with
   `m2t run --recluster-only --speakers N` if needed.
4. **Names**: show `speaker-names.suggested.json` if present and the clips in `speaker-samples/`. Ask the user who
   is who. Never name a voice yourself. Write `speaker-names.json`, run `m2t names`, then `m2t voices enroll`.
5. **Dropped speech**: check long gaps between turns and `recovered` words; ask the user about anything that looks
   lost.
6. **Review** `summary.md` against `transcript.md` (docs/review.md): owners, deadlines (resolved to weekday and date), terms,
   decisions vs proposals, missing tasks. Fix `summary.md`, run `m2t tasks results/<job>`, and list the corrections
   with timecodes.
7. **Hand over** only where the user said, after they approve: notes repository (follow its conventions), issue
   tracker (one issue per task, confirmed handles only), or chat. Report links and what is still unverified.

Recordings, results and voices never go into Git or external services without explicit approval.
