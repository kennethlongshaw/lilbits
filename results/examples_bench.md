| example | phase | ok | process wall / CPU (util) | process peak RSS | phase wall / CPU | phase peak RSS growth | key metrics | deployed | checkpoint file |
|---|---|---|---|---|---|---|---|---|---|
| 01 train+pack, LittleBit | train | ✅ | 9.8 s / 10.0 s (1.03) | 954 MiB | 6.22 s / 6.54 s | 101 MiB | steps = 60; final_train_eval_loss = 3.19 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 3.4 s / 3.3 s (0.99) | 883 MiB | 0.11 s / 0.11 s | 28 MiB | held_out_loss = 3.18; entropy_floor_approx = 0.89 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 4.4 s / 4.2 s (0.97) | 881 MiB | 0.64 s / 0.65 s | 26 MiB | kernel = unpack; packed_held_out_loss = 3.18; ms_per_32x63_batch = 33.8 | 103 KiB | 1,648 KiB |
| 01 train+pack, LittleBit, LUT kernel | train | ✅ | 10.3 s / 10.5 s (1.02) | 930 MiB | 6.88 s / 7.13 s | 100 MiB | steps = 60; final_train_eval_loss = 3.19 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 4.4 s / 4.3 s (0.99) | 883 MiB | 0.14 s / 0.14 s | 28 MiB | held_out_loss = 3.18; entropy_floor_approx = 0.89 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 4.2 s / 4.1 s (0.99) | 882 MiB | 0.55 s / 0.57 s | 30 MiB | kernel = lut; packed_held_out_loss = 3.18; ms_per_32x63_batch = 33.3 | 108 KiB | 1,648 KiB |
| 01 train+pack, BitNet | train | ✅ | 11.0 s / 11.2 s (1.02) | 968 MiB | 7.45 s / 7.78 s | 138 MiB | steps = 60; final_train_eval_loss = 3.26 | 110 KiB | 1,618 KiB |
|  | test | ✅ | 4.4 s / 4.3 s (0.99) | 881 MiB | 0.24 s / 0.25 s | 35 MiB | held_out_loss = 3.26; entropy_floor_approx = 0.89 | 110 KiB | 1,618 KiB |
|  | infer | ✅ | 4.2 s / 4.1 s (0.99) | 876 MiB | 0.70 s / 0.72 s | 29 MiB | kernel = unpack; packed_held_out_loss = 3.26; ms_per_32x63_batch = 39.6 | 130 KiB | 1,618 KiB |
| 01 train+pack, BitNet, LUT kernel | train | ✅ | 10.5 s / 10.6 s (1.01) | 990 MiB | 7.27 s / 7.49 s | 135 MiB | steps = 60; final_train_eval_loss = 3.26 | 110 KiB | 1,618 KiB |
|  | test | ✅ | 3.4 s / 3.3 s (0.99) | 879 MiB | 0.19 s / 0.19 s | 34 MiB | held_out_loss = 3.26; entropy_floor_approx = 0.89 | 110 KiB | 1,618 KiB |
|  | infer | ✅ | 3.7 s / 3.6 s (0.99) | 877 MiB | 0.55 s / 0.56 s | 33 MiB | kernel = lut; packed_held_out_loss = 3.26; ms_per_32x63_batch = 31 | 131 KiB | 1,618 KiB |
| 01 train+pack, FP | train | ✅ | 8.9 s / 9.0 s (1.01) | 908 MiB | 5.44 s / 5.57 s | 99 MiB | steps = 60; final_train_eval_loss = 3.14 | 805 KiB | 1,618 KiB |
|  | test | ✅ | 3.7 s / 3.6 s (0.99) | 879 MiB | 0.11 s / 0.12 s | 26 MiB | held_out_loss = 3.13; entropy_floor_approx = 0.89 | 805 KiB | 1,618 KiB |
|  | infer | ✅ | 3.9 s / 3.8 s (0.99) | 879 MiB | 0.38 s / 0.41 s | 26 MiB | kernel = dense; packed_held_out_loss = 3.13; ms_per_32x63_batch = 23 | 805 KiB | 1,618 KiB |
| 02 recurrent mixers (stateful LMs) | train | ✅ | 8.4 s / 8.5 s (1.02) | 952 MiB | 4.87 s / 5.04 s | 85 MiB | last_loss = {gdn: 3.28, mamba2: 3.29, gla: 3.29} | 1,068 KiB | 2,189 KiB |
|  | test | ✅ | 3.9 s / 3.9 s (1.00) | 888 MiB | 0.13 s / 0.13 s | 27 MiB | held_out_loss = {gdn: 3.26, mamba2: 3.25, gla: 3.29} | 1,068 KiB | 2,189 KiB |
|  | infer | ✅ | 4.1 s / 4.1 s (0.98) | 877 MiB | 0.14 s / 0.14 s | 10 MiB | ms_per_decoded_token_incl_prefill = {gdn: 0.778, mamba2: 0.621, gla: 0.622}; state_bytes = {gdn: 8192, mamba2: 16384, gla: 8192}; attention_kv_bytes_same_length = 65536 | 1,068 KiB | 2,189 KiB |
| 02 recurrent mixers, LittleBit, LUT decode | train | ✅ | 8.4 s / 8.5 s (1.00) | 950 MiB | 4.90 s / 5.06 s | 83 MiB | last_loss = {gdn: 3.45, mamba2: 3.44, gla: 3.44} | 163 KiB | 920 KiB |
|  | test | ✅ | 3.8 s / 3.8 s (0.99) | 879 MiB | 0.13 s / 0.13 s | 27 MiB | held_out_loss = {gdn: 3.44, mamba2: 3.43, gla: 3.45} | 163 KiB | 920 KiB |
|  | infer | ✅ | 3.8 s / 3.8 s (0.99) | 879 MiB | 0.24 s / 0.25 s | 11 MiB | ms_per_decoded_token_incl_prefill = {gdn: 1.08, mamba2: 1.17, gla: 0.9}; state_bytes = {gdn: 8192, mamba2: 16384, gla: 8192}; attention_kv_bytes_same_length = 65536 | 207 KiB | 920 KiB |
| 03 state rollback | train | ✅ | 4.8 s / 4.8 s (0.99) | 925 MiB | 0.90 s / 0.92 s | 76 MiB | steps = 20; last_loss = 3.18 | 213 KiB | 440 KiB |
|  | test | ✅ | 3.9 s / 3.8 s (0.99) | 875 MiB | 0.04 s / 0.04 s | 27 MiB | held_out_loss = 3.17 | 213 KiB | 440 KiB |
|  | infer | ✅ | 3.9 s / 3.8 s (0.99) | 877 MiB | 0.03 s / 0.03 s | 12 MiB | drafts = 5; accepted = 2; rolled_back_state_max_diff_vs_clean = 2.35e-07 | 213 KiB | 440 KiB |
| 04 decision, LittleBit GDN | train | ✅ | 13.3 s / 13.4 s (1.01) | 1136 MiB | 9.67 s / 9.94 s | 267 MiB | steps = 30; last_loss = 3.82 | 282 KiB | 1,003 KiB |
|  | test | ✅ | 3.7 s / 3.6 s (0.98) | 898 MiB | 0.45 s / 0.46 s | 79 MiB | choice_acc = 0.254; choice_ece = 0.0283; choice_brier = 0.753 | 282 KiB | 1,003 KiB |
|  | infer | ✅ | 3.2 s / 3.2 s (1.00) | 883 MiB | 0.14 s / 0.14 s | 16 MiB | prefill_ms_per_state = 3.23; ms_per_question_batched = 0.368 | 282 KiB | 1,003 KiB |
| 04 decision, BitNet GDN | train | ✅ | 12.7 s / 13.0 s (1.02) | 1152 MiB | 9.22 s / 9.61 s | 293 MiB | steps = 30; last_loss = 3.85 | 299 KiB | 1,461 KiB |
|  | test | ✅ | 4.6 s / 4.6 s (1.00) | 914 MiB | 0.81 s / 0.85 s | 98 MiB | choice_acc = 0.266; choice_ece = 0.0214; choice_brier = 0.753 | 299 KiB | 1,461 KiB |
|  | infer | ✅ | 4.1 s / 4.0 s (0.99) | 890 MiB | 0.22 s / 0.23 s | 23 MiB | prefill_ms_per_state = 5.48; ms_per_question_batched = 0.596 | 299 KiB | 1,461 KiB |
| 04 decision, FP GDN | train | ✅ | 10.6 s / 10.7 s (1.01) | 1108 MiB | 6.90 s / 7.08 s | 256 MiB | steps = 30; last_loss = 3.85 | 721 KiB | 1,461 KiB |
|  | test | ✅ | 4.3 s / 4.3 s (1.00) | 918 MiB | 0.63 s / 0.66 s | 99 MiB | choice_acc = 0.266; choice_ece = 0.0371; choice_brier = 0.754 | 721 KiB | 1,461 KiB |
|  | infer | ✅ | 3.8 s / 3.8 s (0.99) | 879 MiB | 0.18 s / 0.19 s | 22 MiB | prefill_ms_per_state = 3.09; ms_per_question_batched = 0.499 | 721 KiB | 1,461 KiB |
| 04 decision, LittleBit Mamba2 | train | ✅ | 13.4 s / 13.7 s (1.03) | 1142 MiB | 9.83 s / 10.24 s | 299 MiB | steps = 30; last_loss = 3.82 | 382 KiB | 1,150 KiB |
|  | test | ✅ | 4.1 s / 4.1 s (0.99) | 907 MiB | 0.57 s / 0.59 s | 86 MiB | choice_acc = 0.277; choice_ece = 0.015; choice_brier = 0.753 | 382 KiB | 1,150 KiB |
|  | infer | ✅ | 3.6 s / 3.6 s (0.99) | 883 MiB | 0.15 s / 0.16 s | 18 MiB | prefill_ms_per_state = 5.97; ms_per_question_batched = 0.376 | 382 KiB | 1,150 KiB |
| 05 pack + LUT + Helion (interpret) | train | ✅ | 8.2 s / 8.3 s (1.02) | 934 MiB | 5.03 s / 5.22 s | 98 MiB | steps = 60; final_train_eval_loss = 3.29 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 3.3 s / 3.3 s (1.00) | 880 MiB | 0.04 s / 0.04 s | 21 MiB | held_out_loss = 3.28; entropy_floor = 0.891 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 12.9 s / 11.7 s (0.91) | 996 MiB | 8.27 s / 7.16 s | 142 MiB | unpacked_weight_ram_bytes = 1641024; packed_checkpoint_bytes = 163769; torch = {held_out_loss: 3.3, max_logit_diff: 0.00177, same_greedy: True} | 103 KiB | 1,648 KiB |
| 06 looped LM, FP | train | ✅ | 14.2 s / 14.5 s (1.02) | 1078 MiB | 10.31 s / 10.66 s | 268 MiB | steps = 40; last_loss = 3.09; mean_sampled_depth = 7.3 | 921 KiB | 1,860 KiB |
|  | test | ✅ | 4.7 s / 4.7 s (1.00) | 882 MiB | 1.11 s / 1.15 s | 28 MiB | held_out_loss_by_depth = {1: 3.23, 2: 3.13, 4: 3.12} | 921 KiB | 1,860 KiB |
|  | infer | ✅ | 3.4 s / 3.3 s (0.98) | 883 MiB | 0.19 s / 0.20 s | 29 MiB | adaptive_exit_held_out_loss = 3.12; adaptive_mean_loops = 6.05; r4_latent_loop_mflop = 2.61 | 921 KiB | 1,860 KiB |
| 06 looped LM, LittleBit core | train | ✅ | 13.8 s / 14.1 s (1.03) | 1132 MiB | 10.11 s / 10.47 s | 271 MiB | steps = 40; last_loss = 3.11; mean_sampled_depth = 7.3 | 516 KiB | 1,444 KiB |
|  | test | ✅ | 4.3 s / 4.3 s (1.00) | 880 MiB | 1.01 s / 1.04 s | 22 MiB | held_out_loss_by_depth = {1: 3.27, 2: 3.16, 4: 3.14} | 516 KiB | 1,444 KiB |
|  | infer | ✅ | 3.7 s / 3.7 s (0.99) | 888 MiB | 0.19 s / 0.20 s | 22 MiB | adaptive_exit_held_out_loss = 3.14; adaptive_mean_loops = 6.55; r4_latent_loop_mflop = 2.61 | 516 KiB | 1,444 KiB |
| 06 looped LM, Ouro exit gate | train | ✅ | 14.2 s / 14.5 s (1.02) | 1154 MiB | 10.97 s / 11.32 s | 320 MiB | steps = 40; last_loss = 3.05; mean_sampled_depth = 7.3 | 922 KiB | 1,861 KiB |
|  | test | ✅ | 4.7 s / 4.7 s (1.00) | 881 MiB | 1.44 s / 1.48 s | 28 MiB | held_out_loss_by_depth = {1: 3.2, 2: 3.13, 4: 3.12} | 922 KiB | 1,861 KiB |
|  | infer | ✅ | 3.2 s / 3.1 s (0.98) | 876 MiB | 0.09 s / 0.09 s | 28 MiB | adaptive_exit_held_out_loss = 3.13; adaptive_mean_loops = 2.15; r4_latent_loop_mflop = 2.61 | 922 KiB | 1,861 KiB |
| 07 byte-level: baseline / BLT / H-Net | train | ✅ | 54.2 s / 57.0 s (1.05) | 1205 MiB | 50.58 s / 53.49 s | 366 MiB | steps_per_model = 30 | 5,944 KiB | 11,939 KiB |
|  | test | ✅ | 3.3 s / 3.3 s (0.99) | 904 MiB | 0.24 s / 0.25 s | 39 MiB | byte transformer = {bits_per_byte: 4.43, bytes_per_patch: 1, inner_len: 256}; BLT = {bits_per_byte: 5.09, bytes_per_patch: 3.59, inner_len: 71.3}; H-Net = {bits_per_byte: 4.26, bytes_per_patch: 2.94, inner_len: 105} | 5,944 KiB | 11,939 KiB |
|  | infer | ✅ | 3.9 s / 3.9 s (0.99) | 914 MiB | 0.88 s / 0.90 s | 39 MiB | ms_per_16xseq = {byte transformer: 74.4, BLT (incl. entropy patching): 68.8, H-Net: 71.2}; embedding_params_bytes_vocab = 65536; embedding_params_200k_vocab_at_d128 = 51200000 | 5,944 KiB | 11,939 KiB |
| 08 LLM-JEPA (LM / online / EMA / EMA+dropout) | train | ✅ | 5.9 s / 6.0 s (1.01) | 894 MiB | 2.15 s / 2.28 s | 29 MiB |  | 788 KiB | 2,412 KiB |
|  | test | ✅ | 3.9 s / 3.8 s (0.98) | 880 MiB | 0.05 s / 0.05 s | 14 MiB | LM loss only = {code_loss: 0.0795, code_token_acc: 1, retrieval_top1: 0.0156}; LLM-JEPA, online target = {code_loss: 0.481, code_token_acc: 0.936, retrieval_top1: 0.188}; LLM-JEPA, EMA target = {code_loss: 0.108, code_token_acc: 1, retrieval_top1: 0.0469} | 788 KiB | 2,412 KiB |
|  | infer | ✅ | 3.5 s / 3.5 s (1.00) | 877 MiB | 0.08 s / 0.09 s | 11 MiB | LM loss only = {ms_per_64_pairs: 3.59}; LLM-JEPA, online target = {ms_per_64_pairs: 3.37}; LLM-JEPA, EMA target = {ms_per_64_pairs: 3.2} | 788 KiB | 2,412 KiB |
| 09 looped index LM (5 stages + decode modes) | train | ✅ | 14.3 s / 14.9 s (1.04) | 1091 MiB | 10.43 s / 11.24 s | 318 MiB |  | 908 KiB | 1,990 KiB |
|  | test | ✅ | 5.3 s / 5.2 s (0.98) | 903 MiB | 1.24 s / 1.26 s | 128 MiB | lean = {loss: 2.29, recall_acc: 0, two_hop_acc: 0}; fp_dense_ref = {loss: 2.18, recall_acc: 0, two_hop_acc: 0} | 908 KiB | 1,990 KiB |
|  | infer | ✅ | 4.3 s / 4.3 s (0.99) | 886 MiB | 0.59 s / 0.61 s | 10 MiB | kernel = native; greedy_distinct_tokens = 1; memory_tiers = {within_token_loop_memory_bytes_per_position: 768, gdn_state_bytes: 40960, summary_bytes: 256} | 908 KiB | 1,990 KiB |
| 09 looped index LM, LUT decode | train | ✅ | 14.7 s / 15.3 s (1.04) | 1099 MiB | 10.73 s / 11.41 s | 315 MiB |  | 908 KiB | 1,990 KiB |
|  | test | ✅ | 4.3 s / 4.3 s (1.00) | 901 MiB | 0.98 s / 1.01 s | 126 MiB | lean = {loss: 2.29, recall_acc: 0, two_hop_acc: 0}; fp_dense_ref = {loss: 2.18, recall_acc: 0, two_hop_acc: 0} | 908 KiB | 1,990 KiB |
|  | infer | ✅ | 3.8 s / 3.7 s (0.99) | 887 MiB | 0.66 s / 0.68 s | 11 MiB | kernel = lut; lut_greedy_same_as_native = True; greedy_distinct_tokens = 1 | 935 KiB | 1,990 KiB |
| 09 looped index LM, greedy bpw allocation | train | ✅ | 13.2 s / 13.7 s (1.03) | 1098 MiB | 9.48 s / 10.01 s | 321 MiB |  | 908 KiB | 1,990 KiB |
|  | test | ✅ | 4.2 s / 4.2 s (1.00) | 900 MiB | 1.06 s / 1.08 s | 126 MiB | lean = {loss: 2.29, recall_acc: 0, two_hop_acc: 0}; fp_dense_ref = {loss: 2.18, recall_acc: 0, two_hop_acc: 0} | 908 KiB | 1,990 KiB |
|  | infer | ✅ | 3.9 s / 3.9 s (0.99) | 886 MiB | 0.57 s / 0.58 s | 11 MiB | kernel = native; greedy_distinct_tokens = 1; memory_tiers = {within_token_loop_memory_bytes_per_position: 768, gdn_state_bytes: 40960, summary_bytes: 256} | 908 KiB | 1,990 KiB |
| 10 pocket LM vs FP16 same size | train | ✅ | 27.2 s / 28.6 s (1.05) | 1952 MiB | 21.85 s / 23.40 s | 1123 MiB | pocket: 1 FP core = {steps: 4, last_loss: 10.5, lm: 5.17}; pocket: 2 LittleBit core = {steps: 4, last_loss: 9.22, lm: 4.57}; pocket: 3 QA instruct = {steps: 4, last_loss: 4.68, lm: 4.64} | 7,817 KiB | 47,397 KiB |
|  | test | ✅ | 8.6 s / 8.6 s (1.00) | 996 MiB | 3.64 s / 3.74 s | 105 MiB | pocket = {bits_per_byte: 7.63, bytes_per_patch: 4.72}; fp16_same_size = {bits_per_byte: 4.42, bytes_per_patch: 1}; note = smoke-trained: bits per byte and exact match do not rank the models | 7,817 KiB | 47,397 KiB |
|  | infer | ✅ | 8.0 s / 8.0 s (1.01) | 973 MiB | 3.82 s / 3.94 s | 55 MiB | kernel = native; pocket_decode_split_ms = {byte_layers: 47.4, router: 7.63, body: 167}; state_bytes = {pocket: 1326152, fp16_same_size: 131072} | 7,817 KiB | 47,397 KiB |
| 10 pocket LM, LUT decode | train | ✅ | 24.8 s / 25.9 s (1.05) | 1959 MiB | 19.67 s / 20.94 s | 1130 MiB | pocket: 1 FP core = {steps: 4, last_loss: 10.5, lm: 5.17}; pocket: 2 LittleBit core = {steps: 4, last_loss: 9.22, lm: 4.57}; pocket: 3 QA instruct = {steps: 4, last_loss: 4.68, lm: 4.64} | 7,817 KiB | 47,397 KiB |
|  | test | ✅ | 8.6 s / 8.6 s (1.00) | 990 MiB | 3.82 s / 3.87 s | 106 MiB | pocket = {bits_per_byte: 7.63, bytes_per_patch: 4.72}; fp16_same_size = {bits_per_byte: 4.42, bytes_per_patch: 1}; note = smoke-trained: bits per byte and exact match do not rank the models | 7,817 KiB | 47,397 KiB |
|  | infer | ✅ | 9.6 s / 9.7 s (1.01) | 941 MiB | 4.96 s / 5.13 s | 53 MiB | kernel = lut; lut_same_as_native = True; pocket_decode_split_ms = {byte_layers: 52.1, router: 8.74, body: 170} | 8,177 KiB | 47,397 KiB |
| 11 energy-based thinking vs plain LM | train | ✅ | 7.3 s / 7.5 s (1.02) | 1053 MiB | 3.52 s / 3.76 s | 258 MiB | ebt: bytes = {steps: 4, last_loss: 6.09, ce: 6.09}; ebt: QA instruct = {steps: 2, last_loss: 6, ce: 6}; lm: bytes = {steps: 6, last_loss: 3.33, wall_s: 0.601} | 7,791 KiB | 15,611 KiB |
|  | test | ✅ | 6.3 s / 6.3 s (1.00) | 897 MiB | 2.21 s / 2.27 s | 36 MiB | lm_bits_per_byte = 4.65; qa_exact_match = {lm: 0, ebt (4 steps): 0}; note = smoke-trained: quality numbers show the mechanics only | 7,791 KiB | 15,611 KiB |
|  | infer | ✅ | 4.3 s / 4.3 s (1.00) | 885 MiB | 0.85 s / 0.87 s | 19 MiB | energy_head_params = 444161; backbone_params = 1772432 | 7,791 KiB | 15,611 KiB |
| 12 Shakespeare task: needle | train | ✅ | 5.1 s / 5.2 s (1.01) | 1074 MiB | 1.66 s / 1.77 s | 228 MiB | sft = {steps: 4, last_loss: 5.15, wall_s: 1.51} | 893 KiB | 1,800 KiB |
|  | test | ✅ | 4.2 s / 4.2 s (0.99) | 883 MiB | 0.45 s / 0.46 s | 20 MiB | task = needle; exact_match = 0 | 893 KiB | 1,800 KiB |
|  | infer | ✅ | 3.9 s / 3.9 s (0.99) | 876 MiB | 0.19 s / 0.19 s | 9 MiB | answer_sample = LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL; ms = 41.6 | 893 KiB | 1,800 KiB |
| 12 Shakespeare task: copy | train | ✅ | 4.2 s / 4.1 s (0.99) | 932 MiB | 0.40 s / 0.42 s | 76 MiB | sft = {steps: 4, last_loss: 4.74, wall_s: 0.253} | 893 KiB | 1,800 KiB |
|  | test | ✅ | 3.9 s / 3.9 s (1.00) | 880 MiB | 0.39 s / 0.40 s | 17 MiB | task = copy; exact_match = 0 | 893 KiB | 1,800 KiB |
|  | infer | ✅ | 4.0 s / 3.9 s (0.98) | 876 MiB | 0.20 s / 0.20 s | 9 MiB | answer_sample = COLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL; ms = 45.7 | 893 KiB | 1,800 KiB |
| 12 Shakespeare task: sort | train | ✅ | 4.1 s / 4.1 s (0.99) | 929 MiB | 0.43 s / 0.46 s | 75 MiB | sft = {steps: 4, last_loss: 4.82, wall_s: 0.271} | 893 KiB | 1,800 KiB |
|  | test | ✅ | 4.1 s / 4.0 s (0.99) | 879 MiB | 0.42 s / 0.44 s | 15 MiB | task = sort; exact_match = 0 | 893 KiB | 1,800 KiB |
|  | infer | ✅ | 4.2 s / 4.2 s (1.00) | 876 MiB | 0.20 s / 0.20 s | 9 MiB | answer_sample = OLUCIOLIOLIOLIOLIOLIOLIOLIOLIOLI; ms = 42.4 | 893 KiB | 1,800 KiB |
| 12 Shakespeare task: tools | train | ✅ | 11.0 s / 10.8 s (0.99) | 931 MiB | 6.84 s / 6.91 s | 78 MiB | sft = {steps: 4, last_loss: 4.95, wall_s: 6.67} | 893 KiB | 1,800 KiB |
|  | test | ✅ | 5.2 s / 5.1 s (0.99) | 882 MiB | 1.01 s / 1.08 s | 18 MiB | task = tools; valid_call_rate = 0; exact_match = 0 | 893 KiB | 1,800 KiB |
|  | infer | ✅ | 4.1 s / 4.1 s (1.00) | 876 MiB | 0.22 s / 0.23 s | 9 MiB | answer_sample = the the the the the the the the ; ms = 57.5 | 893 KiB | 1,800 KiB |
| 12 Shakespeare task: dpo | train | ✅ | 4.8 s / 4.8 s (1.00) | 966 MiB | 1.03 s / 1.09 s | 131 MiB | sft = {steps: 3, last_loss: 5.3, wall_s: 0.177}; dpo = {steps: 4, last_loss: 0.866, dpo: 0.866} | 893 KiB | 1,800 KiB |
|  | test | ✅ | 4.4 s / 4.4 s (0.99) | 889 MiB | 0.52 s / 0.53 s | 32 MiB | task = dpo; exact_match = 0; speaker = 0 | 893 KiB | 1,800 KiB |
|  | infer | ✅ | 4.1 s / 4.1 s (0.99) | 876 MiB | 0.20 s / 0.21 s | 9 MiB | answer_sample = KI; ms = 52.2 | 893 KiB | 1,800 KiB |
| 12 Shakespeare task: grpo | train | ✅ | 6.5 s / 6.6 s (1.02) | 991 MiB | 2.59 s / 2.78 s | 230 MiB | sft = {steps: 3, last_loss: 5.3, wall_s: 0.21}; grpo = {steps: 4, last_loss: 0.000355, reward: 0} | 893 KiB | 1,800 KiB |
|  | test | ✅ | 4.1 s / 4.0 s (0.99) | 878 MiB | 0.44 s / 0.45 s | 10 MiB | task = grpo; exact_match = 0; speaker = 0 | 893 KiB | 1,800 KiB |
|  | infer | ✅ | 4.0 s / 3.9 s (0.99) | 876 MiB | 0.19 s / 0.20 s | 9 MiB | answer_sample = 1ve�;j�RIcL����!�|RIcM.S4\�I ; ms = 51.2 | 893 KiB | 1,800 KiB |
| 12 Shakespeare task: needle, looped index LM | train | ✅ | 7.3 s / 7.4 s (1.02) | 1769 MiB | 3.67 s / 3.89 s | 920 MiB | sft = {steps: 4, last_loss: 4.94, wall_s: 3.53} | 1,263 KiB | 3,015 KiB |
|  | test | ✅ | 7.7 s / 7.8 s (1.01) | 901 MiB | 4.09 s / 4.22 s | 34 MiB | task = needle; exact_match = 0 | 1,263 KiB | 3,015 KiB |
|  | infer | ✅ | 4.4 s / 4.3 s (1.00) | 891 MiB | 0.43 s / 0.45 s | 22 MiB | answer_sample = IL; ms = 254 | 1,263 KiB | 3,015 KiB |
