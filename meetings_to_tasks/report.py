"""Render the transcript, the minutes and the searchable HTML page; shared by the pipeline and the names command."""
import html
import re

import markdown

# Interface strings by meeting language; other languages fall back to English.
STRINGS = {
    'en': {
        'transcript': 'Meeting transcript',
        'auto_note': 'Speaker labels are automatic estimates. Overlapping speech and short remarks may be attributed wrongly.',
        'named_note': 'Names were confirmed from voice samples. Turn boundaries and short remarks are labelled automatically and may be wrong.',
        'minutes': 'Minutes and tasks',
        'search': 'Search the transcript',
        'owner': 'Owner', 'due': 'Due', 'timecode': 'Timecode',
    },
    'ru': {
        'transcript': 'Транскрипт встречи',
        'auto_note': 'Номера говорящих — автоматические оценки. Одновременная речь и короткие реплики могут быть отнесены неверно.',
        'named_note': 'Имена подтверждены по голосовым примерам. Границы реплик и короткие высказывания размечены автоматически и могут содержать ошибки.',
        'minutes': 'Содержание и задачи',
        'search': 'Поиск по транскрипту',
        'owner': 'Ответственный', 'due': 'Срок', 'timecode': 'Таймкод',
    },
}

STYLE = ('body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;line-height:1.6}pre{white-space:pre-wrap}'
         'input{padding:10px;width:90%}b{color:#4557aa}table{border-collapse:collapse;width:100%;font-size:14px}'
         'th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background:#eef1fa}')


def strings(language):
    return STRINGS.get(language or 'en', STRINGS['en'])


def stamp(seconds):
    seconds = int(seconds)
    return f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}'


def speaker_label(turn):
    return turn.get('speaker_name') or turn['speaker']


def transcript_lines(turns):
    return [f"[{stamp(t['start'])}–{stamp(t['end'])}] {speaker_label(t)} ({t['speaker']}): {t['text']}" for t in turns]


def transcript_markdown(turns, language, named=False):
    s = strings(language)
    return f"# {s['transcript']}\n\n{s['named_note' if named else 'auto_note']}\n\n" + '\n\n'.join(transcript_lines(turns))


def table_rows(text, columns=4):
    """Rows of the first Markdown table with this many columns, without its header and separator."""
    rows, seen_header = [], False
    for line in text.splitlines():
        if not line.lstrip().startswith('|'):
            if rows or seen_header:
                break
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) != columns:
            continue
        if all(re.fullmatch(r':?-+:?', c) for c in cells):
            continue
        if not seen_header:
            seen_header = True
            continue
        rows.append(cells)
    return rows


def render_summary(text, language):
    """Minutes as HTML; the task table becomes a list, which reads better than a wide table."""
    s = strings(language)
    tasks = {tuple(r) for r in table_rows(text)}
    prepared = []
    for line in text.splitlines():
        heading = re.fullmatch(r'\s*\*\*(.+?)\*\*\s*', line)
        if heading:
            prepared.extend(['', '## ' + heading.group(1).rstrip(':'), ''])
            continue
        if line.lstrip().startswith('|'):
            cells = tuple(c.strip() for c in line.strip().strip('|').split('|'))
            if len(cells) == 4:
                if cells in tasks:
                    task, owner, due, timecode = cells
                    prepared.extend([f'- **{task}**  ', f"  **{s['owner']}:** {owner}  ", f"  **{s['due']}:** {due}  ",
                                     f"  **{s['timecode']}:** {timecode}", ''])
                continue
        prepared.append(line)
    return markdown.markdown(html.escape('\n'.join(prepared)), extensions=['fenced_code'])


def render_page(title, summary, turns, language, named=False):
    s = strings(language)
    cards = ''.join(f'<p><b>{stamp(t["start"])} {html.escape(speaker_label(t))}</b> {html.escape(t["text"])}</p>' for t in turns)
    search = "document.querySelectorAll('#transcript p').forEach(p=>p.hidden=!p.textContent.toLowerCase().includes(this.value.toLowerCase()))"
    return (f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title><style>{STYLE}</style>'
            f'<h1>{html.escape(title)}</h1><h2>{s["minutes"]}</h2>{render_summary(summary, language)}'
            f'<h2>{s["transcript"]}</h2><p>{s["named_note" if named else "auto_note"]}</p>'
            f'<input placeholder="{s["search"]}" oninput="{search}"><div id="transcript">{cards}</div>')
