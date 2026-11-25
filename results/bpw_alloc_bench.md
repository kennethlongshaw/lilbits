Synthetic layers (output-weighted error, same total bits):

| bpw budget | uniform bpw | allocated bpw | uniform error | allocated error | ratio | alloc time |
|---|---|---|---|---|---|---|
| 1.0 | 0.995 | 0.994 | 3.33e+03 | 2.83e+03 | 0.85 | 2.7 s |
| 0.55 | 0.544 | 0.527 | 3.7e+03 | 3.31e+03 | 0.89 | 2.5 s |
| 0.3 | 0.296 | 0.295 | 4.13e+03 | 3.77e+03 | 0.91 | 2.6 s |

Ranks (uniform -> allocated):

- bpw 1.0: L0 256x256 a=0.2 46->170, L1 256x1024 a=0.4 85->155, L2 1024x256 a=0.6 85->155, L3 512x512 a=0.8 110->73, L4 256x256 a=0.9 46->12, L5 768x256 a=1.1 78->20, L6 256x768 a=1.3 78->20, L7 512x512 a=1.5 110->42
- bpw 0.55: L0 256x256 a=0.2 18->86, L1 256x1024 a=0.4 39->39, L2 1024x256 a=0.6 39->96, L3 512x512 a=0.8 53->28, L4 256x256 a=0.9 18->18, L5 768x256 a=1.1 36->9, L6 256x768 a=1.3 36->9, L7 512x512 a=1.5 53->13
- bpw 0.3: L0 256x256 a=0.2 3->40, L1 256x1024 a=0.4 14->14, L2 1024x256 a=0.6 14->34, L3 512x512 a=0.8 22->12, L4 256x256 a=0.9 3->1, L5 768x256 a=1.1 12->3, L6 256x768 a=1.3 12->3, L7 512x512 a=1.5 22->6

Looped index LM core, d=256, untrained FP -> LittleBit (hidden-state error vs FP on held-out tokens):

| bpw budget | ranks | hidden rel. error | core size | decode 16 tok | CPU | peak RSS growth | alloc time |
|---|---|---|---|---|---|---|---|
| 1.0 | uniform | 1.095 | 180 KiB | 711 ms | 733 ms | 15.7 MiB | 0.0 s |
| 1.0 | allocated | 1.208 | 180 KiB | 715 ms | 738 ms | 18.8 MiB | 2.8 s |
| 0.55 | uniform | 1.219 | 117 KiB | 592 ms | 618 ms | 18.6 MiB | 0.0 s |
| 0.55 | allocated | 1.348 | 117 KiB | 489 ms | 504 ms | 17.4 MiB | 2.6 s |
| 0.3 | uniform | 1.336 | 82 KiB | 411 ms | 429 ms | 18.9 MiB | 0.0 s |
| 0.3 | allocated | 1.382 | 82 KiB | 390 ms | 404 ms | 21.5 MiB | 2.5 s |
