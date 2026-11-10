# Reviewing the minutes

`summary.md` and `tasks.json` are drafts written by an LLM from an automatic transcript. Check them before anyone
acts on them. Read `transcript.md` in full first, then go through the minutes item by item at the cited timecodes.

## What to check

- **Owner.** Who actually took the task at that timecode? The LLM often assigns the person who raised the topic, or
  someone merely mentioned. `owner_confirmed: false` means the owner is still a `SPEAKER_XX` label.
- **Deadline.** Was it stated, or inferred? Resolve relative dates against the meeting date ("tomorrow" at a Monday
  meeting is Tuesday) and write the weekday and the date. An unstated deadline stays "not set".
- **Terms and names.** Product, tool and dataset names are where speech recognition fails most. Check the spelling
  against the transcript and what the team actually uses; ask when unsure.
- **Decisions and proposals.** Only what was agreed is a decision.
- **Coverage.** Tasks spoken in the meeting but missing from the table; items that belong to another meeting or
  that the team says are out of scope.
- **Dropped speech.** A long gap between turns can be a pause or lost speech. Words marked `recovered` were decoded
  again from such a gap and need a closer look; their speaker is less certain.

## How to fix

1. Edit `summary.md`; keep the timecodes.
2. Run `m2t tasks results/<job>`, so `tasks.json` and `tasks.md` follow the edit.
3. Do not edit `transcript.md` or `transcript.json` to make the minutes fit: the transcript stays verbatim. A passage
   that had to be inserted by hand is marked as inserted and uncertain.
4. Write down what was corrected and why, with timecodes, for whoever receives the minutes.

## Handing the tasks over

- **Notes repository.** One folder per meeting with `summary.md`, `tasks.md`, `transcript.md`, following that
  repository's conventions: language, naming, index pages, checks.
- **Issue tracker.** One issue per row of `tasks.json`: the task as the title, the timecode and a short quote in the
  body, the owner as assignee only when their account is confirmed by a person.
- **Team chat.** The summary and the task table.

Publishing minutes or creating issues is visible to others: agree on it with the participants first.
