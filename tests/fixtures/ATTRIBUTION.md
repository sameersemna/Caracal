# Fixture attribution

- `audio/jfk.wav` — excerpt of J. F. Kennedy inaugural address (1961), US Government work, public domain. Copied from the whisper.cpp repository samples.
- `whisper/jfk.json` — real response of the local whisper.cpp server (ggml-large-v3) for `jfk.wav`, `verbose_json`, `language=auto`, recorded 2026-11-14.
- `whisper/talk_c0.json`, `talk_c1.json` — real whisper.cpp (large-v3) output for 0–61 s and 60–120 s (1 s overlap) of the TED talk
  "How to stay calm when you know you'll be stressed" by Daniel Levitin (YouTube 8jPQjjsBbIc, CC BY-NC-ND 4.0), recorded 2026-11-14.
  `talk_c0_forced_de.json` — same chunk with `language=de` forced (shows `language` echoes the request while `detected_language*` keep the detection).
- `whisper/silence20.json` — 20 s of digital silence (0 segments; detected "en" at only 0.32).
- `captions/talk_en_manual.vtt`, `captions/talk_en_auto.vtt` — the same talk's YouTube manual and auto (`en-orig`, word-timed, rolling) captions,
  trimmed to cues starting before 125 s. Same licence.
- `ollama/*.json` — real responses of Ollama cloud `gemma4:31b` (2026-11-14) for the German test transcript in `ollama/README.json`:
  tool call with pydantic `$defs` schema (invented field names, upper-case enums), tool call with inlined schema (correct fields, enum
  synonyms such as `location`), `format:"json"` without tools (fenced JSON), `format:<schema>` without tools (ignored → markdown).
- `search/*.json` — real Brave Web Search and Exa `/search` responses recorded 2026-11-14 (keys removed): "All Ears English podcast co-host Lindsay"
  (both name Lindsay McMahon), the JFK quote query, and both providers' invalid-key error responses (Brave 422 `SUBSCRIPTION_TOKEN_INVALID`, Exa 401).
- `whisper/tagesschau_seg0.json` — first merged segment (14 whisper tokens, ~2.6 s) of a Caracal transcript of a "Tagesschau in 100 Sekunden"
  re-upload (YouTube OgDZqde_My4), recorded 2026-11-15. Shows sub-word tokens without a leading space (`Tag`+`ess`+`chau`, `Sek`+`unden`).
- `whisper/zh_news40.json` (intro: hallucinated credits only) and `whisper/zh_speech40.json` (60–100 s: real speech) — real whisper.cpp (large-v3) `verbose_json` for the first 40 s of a Chinese news clip (YouTube HPVvfcTOSJg),
  recorded 2026-11-15. Shows (a) CJK tokens without spaces, (b) multi-byte characters split into `U+FFFD` byte-fragment tokens
  (`志` `�` `�` `者` for 志愿者) while the segment `text` is correct, (c) the classic hallucinated credit "中文字幕志愿者 李宗盛" at 0 s.
- `whisper/zh_full_loop.json` — real whisper.cpp output for the **whole** 93 s of HPVvfcTOSJg with default context: the credit hallucination repeated at 0/30/60/90 s (context loop). Recorded 2026-11-15.
