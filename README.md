# Free Manual Chinese–Dutch Lyrics Converter

A completely free local Streamlit dashboard that converts a basic-format bilingual lyric document into the required UTF-8 TXT structure.

## No paid services

This version does not use:

- an OpenAI API key;
- ChatGPT API usage;
- cloud AI;
- subscriptions or usage-based services.

Everything runs locally on the computer. The user manually confirms which Chinese and Dutch sections and line ranges are translations of each other.

Chinese word segmentation uses the free MIT-licensed `jieba` package in its standard offline mode. No lyric text is sent to an API or external service.

## Dashboard workflow

1. Upload a supported source file.
2. Review the detected Chinese and Dutch sections.
3. Select the true Dutch translation section or sections for each Chinese section.
4. Review the side-by-side numbered Chinese and Dutch reference panels, edit the Chinese range rows when needed, and drag the Dutch cards into the matching row boxes.
5. Validate that every Chinese and Dutch line is covered reciprocally.
6. Choose where the output changes from `Chinese|Dutch` to `Dutch|Chinese`.
7. Remove punctuation from title/lyric content and normalize whitespace.
8. Apply the language-order switch to determine which language is before and after `|`.
9. Calculate cleaned lengths with the stricter first-side limit and add at most one word-safe `//` per language when needed.
10. Add the generated `[Section]`, `|`, and `//` formatting.
11. Preview and edit the final result.
12. Download a UTF-8 `.txt` file.

## Unequal song structures

The application does not require equal numbers of sections or lyric lines. It supports:

- different numbers of Chinese and Dutch verses;
- one Chinese section matching multiple Dutch sections;
- multiple Chinese lines matching one Dutch line;
- one Chinese line matching multiple Dutch lines;
- repeated translated sections;
- choruses, bridges, intros, outros, and other detected headings.

The dashboard never assumes that matching verse numbers are translations. Its initial section and line suggestions are only a convenient starting layout. The user must verify and edit them.

## Drag-and-drop line matching

Dutch sections are assigned codes such as `D1`, `D2`, and `D3`. Every Dutch lyric line appears as a uniquely numbered draggable card.

- Each section keeps the original clear reference layout: numbered Chinese lines on the left and numbered Dutch lines on the right.
- The editable **Chinese line(s)** column defines the vertical mapping rows. Use `1`, `2-3`, and additional rows when the structures differ.
- Each mapping row shows its Chinese range in the left box and a draggable **Dutch reference(s)** box on the right.
- Drag one or more Dutch cards into each row; cards from multiple selected Dutch sections can share a row.
- Use **Move all cards to pool** to start from an unassigned Dutch card list, or **Use suggested placement** to restore the non-semantic starting arrangement.
- Cards may remain in the pool on one board when that Dutch section is shared with another Chinese section. Final validation still requires every Dutch line to be used somewhere.

No Dutch reference syntax needs to be typed. The user must still review every card placement and click the validation button before conversion.

## Supported uploads

- DOCX
- PDF containing selectable text
- PPTX
- XLSX
- TXT, MD, CSV, JSON, XML and HTML

The document must use recognizable headings such as `Verse 1`, `Chorus 1`, `Bridge`, or `Refrein 1`.

## Output rules

- Section names are surrounded by square brackets.
- The two languages are separated by `|`.
- Before the selected switch point, output is `Chinese|Dutch`.
- From the selected switch point onward, output is `Dutch|Chinese`.
- The configured Chinese and Dutch limits are the normal limits used after `|`. Their defaults are 10 Chinese characters and 40 Dutch characters.
- The language before `|` uses `max(4, floor(normal limit × 0.80))`. With the defaults, that is 8 Chinese characters or 32 Dutch characters; the language after `|` remains at 10 or 40 respectively.
- All Unicode punctuation is removed from title and lyric content before length checks. This includes Western and Chinese commas, stops, quotes, brackets, dashes, ellipses, and other Unicode punctuation categories.
- Whitespace left after cleaning is trimmed and collapsed without joining Dutch words.
- Separator spaces created only to keep punctuation-separated words apart are not split candidates; former punctuation locations never determine a `//` position.
- Generated structural markers such as `[Title]`, `[Verse 1]`, `|`, and `//` are added after cleaning and remain intact.
- A cleaned lyric receives no `//` when it is within the applicable first- or second-side limit.
- First-side safe boundaries are accepted only when both fragments contain at least two characters and at least 25% of the cleaned text. This prevents the lower limit from creating tiny fragments. Dutch remains unsplit when no acceptable word-safe boundary exists; Chinese retains its existing warned character fallback.
- A long Chinese lyric is divided at the closest balanced `jieba` word boundary.
- If the local segmenter finds no Chinese word boundary, existing whitespace is tried before a last-resort character boundary. The dashboard shows a non-blocking warning when that fallback is used.
- A long Dutch lyric is divided only at the closest balanced whitespace boundary, never inside a word.
- Each language receives at most one `//` per output row, even when the text is extremely long.
- Apart from the required punctuation removal and whitespace normalization, lyric characters and words are never translated, rewritten, deleted, duplicated, or reordered.
- Export is blocked until all detected sections and lines have valid reciprocal mappings.
- The final preview remains editable before download.

## Run on Windows

Prerequisite: install Python 3.10 or newer and enable **Add Python to PATH** during installation.

Double-click:

```text
start_dashboard.bat
```

The launcher creates a private local Python environment inside the project folder, installs the pinned free open-source dependencies, and starts the dashboard in the browser. The first run requires an internet connection for the dependency download; later launches can run offline after setup succeeds. Keep the command window open while using the dashboard; closing it stops the local site.

Manual commands:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Run on macOS or Linux

Install Python 3.10 or newer, open a terminal in the extracted project folder, and run:

```sh
sh start_dashboard.sh
```

Keep that terminal open while using the dashboard. Press `Ctrl+C` to stop it.

## Run tests

```powershell
python -m pytest -q
```

## Deploy on Streamlit Community Cloud

- Put `app.py`, `requirements.txt`, `lyrics_dashboard/`, and the other non-ignored project files in the repository root.
- Select `app.py` as the Community Cloud entrypoint.
- Select Python 3.12 in Advanced settings. Every pinned runtime dependency supports Python 3.12 and Debian Linux.
- No `packages.txt`, API key, secret, paid service, or fixed local data file is required.
- Do not commit `.venv/`, caches, local Streamlit logs/PIDs, `.streamlit/secrets.toml`, temporary uploads, generated downloads, or `shareable-release/`.

The same clean-install startup commands used by Community Cloud are:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Project structure

- `app.py` — free manual Streamlit workflow
- `lyrics_dashboard/extractors.py` — file text extraction
- `lyrics_dashboard/parser.py` — title, section, and language parsing
- `lyrics_dashboard/alignment.py` — exact manual mapping, reciprocal validation, and editable suggestions
- `lyrics_dashboard/drag_mapping.py` — drag-board grouping, card decoding, and structured line mappings
- `lyrics_dashboard/text_processing.py` — Unicode punctuation removal and whitespace normalization
- `lyrics_dashboard/splitter.py` — local word-safe and fallback `//` placement
- `lyrics_dashboard/converter.py` — final TXT generation
- `tests/` — regression tests
