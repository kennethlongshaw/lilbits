| example | phase | ok | process wall / CPU (util) | process peak RSS | phase wall / CPU | phase peak RSS growth | key metrics | deployed | checkpoint file |
|---|---|---|---|---|---|---|---|---|---|
| 01 train+pack, LittleBit | train | ✅ | 12.0 s / 12.2 s (1.02) | 941 MiB | 7.90 s / 8.19 s | 101 MiB | steps = 60; final_train_eval_loss = 3.19 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 3.7 s / 3.7 s (0.99) | 883 MiB | 0.11 s / 0.12 s | 29 MiB | held_out_loss = 3.18; entropy_floor_approx = 0.89 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 5.0 s / 4.9 s (0.99) | 882 MiB | 0.71 s / 0.74 s | 27 MiB | kernel = unpack; packed_held_out_loss = 3.18; ms_per_32x63_batch = 36.8 | 103 KiB | 1,648 KiB |
| 01 train+pack, LittleBit, LUT kernel | train | ✅ | 11.8 s / 11.8 s (1.01) | 933 MiB | 7.90 s / 8.12 s | 101 MiB | steps = 60; final_train_eval_loss = 3.19 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 4.3 s / 4.3 s (0.99) | 884 MiB | 0.13 s / 0.14 s | 29 MiB | held_out_loss = 3.18; entropy_floor_approx = 0.89 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 4.2 s / 4.1 s (0.99) | 882 MiB | 0.43 s / 0.44 s | 27 MiB | kernel = lut; packed_held_out_loss = 3.18; ms_per_32x63_batch = 23.9 | 108 KiB | 1,648 KiB |
| 01 train+pack, BitNet | train | ✅ | 13.5 s / 13.7 s (1.01) | 966 MiB | 9.54 s / 9.84 s | 137 MiB | steps = 60; final_train_eval_loss = 3.26 | 110 KiB | 1,618 KiB |
|  | test | ✅ | 4.2 s / 4.1 s (0.99) | 880 MiB | 0.23 s / 0.23 s | 35 MiB | held_out_loss = 3.26; entropy_floor_approx = 0.89 | 110 KiB | 1,618 KiB |
|  | infer | ✅ | 5.3 s / 5.3 s (1.00) | 876 MiB | 0.99 s / 1.02 s | 30 MiB | kernel = unpack; packed_held_out_loss = 3.26; ms_per_32x63_batch = 57 | 130 KiB | 1,618 KiB |
| 01 train+pack, BitNet, LUT kernel | train | ✅ | 14.7 s / 14.9 s (1.02) | 990 MiB | 10.15 s / 10.49 s | 139 MiB | steps = 60; final_train_eval_loss = 3.26 | 110 KiB | 1,618 KiB |
|  | test | ✅ | 4.2 s / 4.2 s (0.99) | 882 MiB | 0.24 s / 0.24 s | 35 MiB | held_out_loss = 3.26; entropy_floor_approx = 0.89 | 110 KiB | 1,618 KiB |
|  | infer | ✅ | 5.0 s / 5.0 s (0.99) | 878 MiB | 0.76 s / 0.78 s | 31 MiB | kernel = lut; packed_held_out_loss = 3.26; ms_per_32x63_batch = 44.9 | 131 KiB | 1,618 KiB |
| 01 train+pack, FP | train | ✅ | 9.7 s / 9.7 s (1.01) | 936 MiB | 5.67 s / 5.86 s | 96 MiB | steps = 60; final_train_eval_loss = 3.14 | 805 KiB | 1,618 KiB |
|  | test | ✅ | 3.7 s / 3.7 s (0.99) | 880 MiB | 0.11 s / 0.11 s | 27 MiB | held_out_loss = 3.13; entropy_floor_approx = 0.89 | 805 KiB | 1,618 KiB |
|  | infer | ✅ | 3.9 s / 3.8 s (0.99) | 880 MiB | 0.34 s / 0.35 s | 27 MiB | kernel = dense; packed_held_out_loss = 3.13; ms_per_32x63_batch = 19.9 | 805 KiB | 1,618 KiB |
| 02 recurrent mixers (stateful LMs) | train | ✅ | 9.9 s / 10.0 s (1.01) | 956 MiB | 5.92 s / 6.10 s | 96 MiB | last_loss = {gdn: 3.28, mamba2: 3.29, gla: 3.29} | 1,068 KiB | 2,189 KiB |
|  | test | ✅ | 4.2 s / 4.2 s (0.99) | 896 MiB | 0.14 s / 0.14 s | 34 MiB | held_out_loss = {gdn: 3.25, mamba2: 3.25, gla: 3.29} | 1,068 KiB | 2,189 KiB |
|  | infer | ✅ | 4.5 s / 4.5 s (0.98) | 879 MiB | 0.21 s / 0.22 s | 11 MiB | ms_per_decoded_token_incl_prefill = {gdn: 1.83, mamba2: 0.954, gla: 0.565}; state_bytes = {gdn: 8192, mamba2: 16384, gla: 8192}; attention_kv_bytes_same_length = 65536 | 1,068 KiB | 2,189 KiB |
| 02 recurrent mixers, LittleBit, LUT decode | train | ✅ | 9.6 s / 9.6 s (1.00) | 953 MiB | 5.51 s / 5.65 s | 95 MiB | last_loss = {gdn: 3.45, mamba2: 3.44, gla: 3.44} | 163 KiB | 920 KiB |
|  | test | ✅ | 3.9 s / 3.8 s (0.99) | 896 MiB | 0.17 s / 0.18 s | 34 MiB | held_out_loss = {gdn: 3.46, mamba2: 3.43, gla: 3.45} | 163 KiB | 920 KiB |
|  | infer | ✅ | 4.3 s / 4.3 s (0.99) | 880 MiB | 0.30 s / 0.31 s | 12 MiB | ms_per_decoded_token_incl_prefill = {gdn: 2.08, mamba2: 1.24, gla: 0.86}; state_bytes = {gdn: 8192, mamba2: 16384, gla: 8192}; attention_kv_bytes_same_length = 65536 | 207 KiB | 920 KiB |
| 03 state rollback | train | ✅ | 5.0 s / 5.0 s (0.99) | 930 MiB | 0.97 s / 1.00 s | 78 MiB | steps = 20; last_loss = 3.16 | 213 KiB | 440 KiB |
|  | test | ✅ | 4.1 s / 4.1 s (0.99) | 893 MiB | 0.05 s / 0.05 s | 35 MiB | held_out_loss = 3.17 | 213 KiB | 440 KiB |
|  | infer | ✅ | 4.2 s / 4.1 s (0.98) | 878 MiB | 0.04 s / 0.05 s | 13 MiB | drafts = 5; accepted = 2; rolled_back_state_max_diff_vs_clean = 1.79e-07 | 213 KiB | 440 KiB |
| 04 decision, LittleBit GDN | train | ✅ | 13.2 s / 13.4 s (1.01) | 1129 MiB | 9.56 s / 9.85 s | 265 MiB | steps = 30; last_loss = 3.82 | 282 KiB | 1,003 KiB |
|  | test | ✅ | 3.9 s / 3.9 s (1.00) | 900 MiB | 0.55 s / 0.57 s | 91 MiB | choice_acc = 0.246; choice_ece = 0.036; choice_brier = 0.754 | 282 KiB | 1,003 KiB |
|  | infer | ✅ | 3.6 s / 3.6 s (0.99) | 889 MiB | 0.20 s / 0.21 s | 22 MiB | prefill_ms_per_state = 6.75; ms_per_question_batched = 0.509 | 282 KiB | 1,003 KiB |
| 04 decision, BitNet GDN | train | ✅ | 12.6 s / 12.8 s (1.02) | 1141 MiB | 9.27 s / 9.55 s | 286 MiB | steps = 30; last_loss = 3.87 | 299 KiB | 1,461 KiB |
|  | test | ✅ | 4.3 s / 4.3 s (1.00) | 927 MiB | 0.88 s / 0.91 s | 111 MiB | choice_acc = 0.258; choice_ece = 0.0696; choice_brier = 0.756 | 299 KiB | 1,461 KiB |
|  | infer | ✅ | 3.8 s / 3.7 s (0.98) | 883 MiB | 0.25 s / 0.26 s | 31 MiB | prefill_ms_per_state = 7.48; ms_per_question_batched = 0.661 | 299 KiB | 1,461 KiB |
| 04 decision, FP GDN | train | ✅ | 13.7 s / 13.9 s (1.01) | 1104 MiB | 9.93 s / 10.25 s | 260 MiB | steps = 30; last_loss = 3.85 | 721 KiB | 1,461 KiB |
|  | test | ✅ | 3.9 s / 3.9 s (1.00) | 927 MiB | 0.60 s / 0.62 s | 108 MiB | choice_acc = 0.266; choice_ece = 0.0336; choice_brier = 0.754 | 721 KiB | 1,461 KiB |
|  | infer | ✅ | 4.1 s / 4.1 s (0.99) | 886 MiB | 0.27 s / 0.28 s | 28 MiB | prefill_ms_per_state = 7.05; ms_per_question_batched = 0.728 | 721 KiB | 1,461 KiB |
| 04 decision, LittleBit Mamba2 | train | ✅ | 14.4 s / 14.7 s (1.02) | 1167 MiB | 10.77 s / 11.14 s | 301 MiB | steps = 30; last_loss = 3.82 | 382 KiB | 1,150 KiB |
|  | test | ✅ | 4.3 s / 4.3 s (1.00) | 927 MiB | 0.71 s / 0.75 s | 71 MiB | choice_acc = 0.281; choice_ece = 0.00828; choice_brier = 0.753 | 382 KiB | 1,150 KiB |
|  | infer | ✅ | 4.7 s / 4.6 s (0.98) | 886 MiB | 0.28 s / 0.29 s | 24 MiB | prefill_ms_per_state = 9.41; ms_per_question_batched = 0.719 | 382 KiB | 1,150 KiB |
| 05 pack + LUT + Helion (interpret) | train | ✅ | 9.9 s / 10.0 s (1.01) | 931 MiB | 6.12 s / 6.32 s | 100 MiB | steps = 60; final_train_eval_loss = 3.29 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 3.8 s / 3.8 s (0.99) | 886 MiB | 0.04 s / 0.04 s | 20 MiB | held_out_loss = 3.28; entropy_floor = 0.891 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 11.0 s / 11.1 s (1.01) | 997 MiB | 6.51 s / 6.70 s | 142 MiB | unpacked_weight_ram_bytes = 1641024; packed_checkpoint_bytes = 163769; torch = {held_out_loss: 3.3, max_logit_diff: 0.00177, same_greedy: True} | 103 KiB | 1,648 KiB |
| 06 looped LM, FP | train | ✅ | 14.7 s / 14.9 s (1.02) | 1082 MiB | 10.48 s / 10.81 s | 272 MiB | steps = 40; last_loss = 3.09; mean_sampled_depth = 7.3 | 921 KiB | 1,860 KiB |
|  | test | ✅ | 4.9 s / 4.8 s (0.99) | 882 MiB | 1.13 s / 1.17 s | 27 MiB | held_out_loss_by_depth = {1: 3.23, 2: 3.13, 4: 3.12} | 921 KiB | 1,860 KiB |
|  | infer | ✅ | 4.4 s / 4.4 s (1.00) | 876 MiB | 0.28 s / 0.30 s | 29 MiB | adaptive_exit_held_out_loss = 3.12; adaptive_mean_loops = 6.05; r4_latent_loop_mflop = 2.61 | 921 KiB | 1,860 KiB |
| 06 looped LM, LittleBit core | train | ✅ | 16.1 s / 16.4 s (1.02) | 1129 MiB | 12.18 s / 12.63 s | 275 MiB | steps = 40; last_loss = 3.11; mean_sampled_depth = 7.3 | 516 KiB | 1,444 KiB |
|  | test | ✅ | 4.8 s / 4.8 s (1.00) | 881 MiB | 1.11 s / 1.14 s | 21 MiB | held_out_loss_by_depth = {1: 3.27, 2: 3.16, 4: 3.14} | 516 KiB | 1,444 KiB |
|  | infer | ✅ | 4.1 s / 4.0 s (0.98) | 881 MiB | 0.22 s / 0.23 s | 21 MiB | adaptive_exit_held_out_loss = 3.14; adaptive_mean_loops = 6.55; r4_latent_loop_mflop = 2.61 | 516 KiB | 1,444 KiB |
| 06 looped LM, Ouro exit gate | train | ✅ | 17.6 s / 18.0 s (1.02) | 1156 MiB | 13.60 s / 14.07 s | 323 MiB | steps = 40; last_loss = 3.05; mean_sampled_depth = 7.3 | 922 KiB | 1,861 KiB |
|  | test | ✅ | 5.3 s / 5.3 s (0.99) | 875 MiB | 1.39 s / 1.43 s | 29 MiB | held_out_loss_by_depth = {1: 3.2, 2: 3.13, 4: 3.12} | 922 KiB | 1,861 KiB |
|  | infer | ✅ | 4.3 s / 4.2 s (0.99) | 875 MiB | 0.12 s / 0.13 s | 29 MiB | adaptive_exit_held_out_loss = 3.13; adaptive_mean_loops = 2.15; r4_latent_loop_mflop = 2.61 | 922 KiB | 1,861 KiB |
| 07 byte-level: baseline / BLT / H-Net | train | ✅ | 45.1 s / 48.6 s (1.08) | 1192 MiB | 41.58 s / 45.19 s | 359 MiB | steps_per_model = 30 | 5,944 KiB | 11,939 KiB |
|  | test | ✅ | 4.1 s / 4.0 s (0.99) | 904 MiB | 0.28 s / 0.29 s | 42 MiB | byte transformer = {bits_per_byte: 4.41, bytes_per_patch: 1, inner_len: 256}; BLT = {bits_per_byte: 5.02, bytes_per_patch: 4.77, inner_len: 53.6}; H-Net = {bits_per_byte: 4.23, bytes_per_patch: 2.83, inner_len: 114} | 5,944 KiB | 11,939 KiB |
|  | infer | ✅ | 5.2 s / 5.2 s (0.99) | 909 MiB | 1.12 s / 1.15 s | 39 MiB | ms_per_16xseq = {byte transformer: 90.2, BLT (incl. entropy patching): 83.7, H-Net: 102}; embedding_params_bytes_vocab = 65536; embedding_params_200k_vocab_at_d128 = 51200000 | 5,944 KiB | 11,939 KiB |
| 08 LLM-JEPA (LM / online / EMA / EMA+dropout) | train | ✅ | 6.1 s / 6.1 s (1.01) | 893 MiB | 2.09 s / 2.21 s | 30 MiB |  | 788 KiB | 2,412 KiB |
|  | test | ✅ | 4.2 s / 4.1 s (0.98) | 880 MiB | 0.04 s / 0.04 s | 14 MiB | LM loss only = {code_loss: 0.0795, code_token_acc: 1, retrieval_top1: 0.0156}; LLM-JEPA, online target = {code_loss: 0.481, code_token_acc: 0.936, retrieval_top1: 0.188}; LLM-JEPA, EMA target = {code_loss: 0.108, code_token_acc: 1, retrieval_top1: 0.0469} | 788 KiB | 2,412 KiB |
|  | infer | ✅ | 4.0 s / 4.0 s (0.99) | 877 MiB | 0.09 s / 0.09 s | 11 MiB | LM loss only = {ms_per_64_pairs: 3.6}; LLM-JEPA, online target = {ms_per_64_pairs: 3.51}; LLM-JEPA, EMA target = {ms_per_64_pairs: 3.47} | 788 KiB | 2,412 KiB |
| 09 looped index LM (5 stages + decode modes) | train | ✅ | 18.6 s / 19.5 s (1.05) | 1087 MiB | 14.26 s / 15.34 s | 306 MiB |  | 908 KiB | 1,990 KiB |
|  | test | ✅ | 4.9 s / 4.9 s (1.00) | 906 MiB | 1.15 s / 1.23 s | 133 MiB | lean = {loss: 2.27, recall_acc: 0, two_hop_acc: 0}; fp_dense_ref = {loss: 2.19, recall_acc: 0, two_hop_acc: 0} | 908 KiB | 1,990 KiB |
|  | infer | ✅ | 5.4 s / 5.3 s (0.99) | 887 MiB | 1.44 s / 1.47 s | 11 MiB | kernel = native; greedy_distinct_tokens = 2; memory_tiers = {within_token_loop_memory_bytes_per_position: 768, gdn_state_bytes: 40960, summary_bytes: 256} | 908 KiB | 1,990 KiB |
| 09 looped index LM, LUT decode | train | ✅ | 16.4 s / 17.2 s (1.05) | 1085 MiB | 13.07 s / 13.95 s | 306 MiB |  | 908 KiB | 1,990 KiB |
|  | test | ✅ | 4.8 s / 4.8 s (1.00) | 907 MiB | 1.16 s / 1.21 s | 132 MiB | lean = {loss: 2.27, recall_acc: 0, two_hop_acc: 0}; fp_dense_ref = {loss: 2.19, recall_acc: 0, two_hop_acc: 0} | 908 KiB | 1,990 KiB |
|  | infer | ✅ | 4.7 s / 4.7 s (0.99) | 889 MiB | 1.42 s / 1.45 s | 12 MiB | kernel = lut; lut_greedy_same_as_native = True; greedy_distinct_tokens = 2 | 935 KiB | 1,990 KiB |
