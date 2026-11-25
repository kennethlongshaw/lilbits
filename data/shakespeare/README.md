# Toy Shakespeare data

- `tiny_shakespeare.txt`: the tiny-shakespeare corpus (1,115,394 bytes, about 40k lines of speaker turns from
  the plays; public domain text, as collected by Andrej Karpathy for char-rnn:
  https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare). sha256
  `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- `qa_train.jsonl`, `qa_test.jsonl`: question / answer pairs generated from it by `lilbits.shakespeare.make_qa`
  (regenerate with `uv run python -m lilbits.shakespeare`). Every answer is read off the text, so it is exactly
  scorable. The test pairs come only from the last 10% of speeches, which training text excludes.

The corpus has speaker turns but no play / act / scene markers, so the questions are about speakers and lines:

| kind | question | answer |
|---|---|---|
| `speaker` | Who says "<line>"? | the speaker, e.g. `MENENIUS` |
| `next_line` | What line comes after "<line>"? | the next line of the same speech |
| `next_speaker` | Who speaks after <SPEAKER> says "<last line>"? | the next speaker |
| `line_count` | How many lines are in <SPEAKER>'s speech that begins "<line>"? | a number |
