# Generated media fixtures

No audio or video binaries are committed. `tests/unit/test_ffmpeg.py` builds every fixture in
`tmp_path` with the local ffmpeg (6.1.1 on this host) using lavfi sources, so the suite stays
small and reproducible:

| Fixture | ffmpeg source |
|---|---|
| tone | `-f lavfi -i sine=frequency=440:duration=<s>` |
| tone + silence + tone | three lavfi inputs (`sine`, `anullsrc`, `sine`) joined by `concat=n=3:v=0:a=1` |
| pure silence | `-f lavfi -i anullsrc=r=16000:cl=mono:d=<s>` |
| stereo 44.1 kHz mp3/m4a/ogg/flac | `sine` input with `-ac 2 -ar 44100`; the muxer is chosen from the suffix |
| video only (no audio) | `-f lavfi -i testsrc=duration=1:size=160x120:rate=10 -an` |

Hostile inputs are plain text or random bytes written by the tests:

- `evil.m3u8` — an HLS playlist pointing at `http://127.0.0.1:9/x.ts` (must be rejected by
  the demuxer whitelist before any connection is attempted)
- `evil.wav` — an ffconcat list containing `file '/etc/hostname'`
- `noise.mp3` — random bytes
- `empty.mp3` — zero bytes

`tests/fixtures/audio/jfk.wav` remains the one committed real-audio fixture (docs/07 §6).
