| example | phase | ok | process wall / CPU (util) | process peak RSS | phase wall / CPU | phase peak RSS growth | key metrics | deployed | checkpoint file |
|---|---|---|---|---|---|---|---|---|---|
| 01 train+pack, LittleBit | train | ✅ | 11.6 s / 11.8 s (1.02) | 927 MiB | 7.95 s / 8.19 s | 100 MiB | steps = 60; final_train_eval_loss = 3.19 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 4.0 s / 3.9 s (0.98) | 883 MiB | 0.13 s / 0.13 s | 29 MiB | held_out_loss = 3.18; entropy_floor_approx = 0.89 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 4.1 s / 4.1 s (0.99) | 882 MiB | 0.53 s / 0.55 s | 27 MiB | kernel = unpack; packed_held_out_loss = 3.18; ms_per_32x63_batch = 25.9 | 103 KiB | 1,648 KiB |
| 01 train+pack, LittleBit, LUT kernel | train | ✅ | 10.4 s / 10.4 s (1.00) | 930 MiB | 6.78 s / 6.83 s | 102 MiB | steps = 60; final_train_eval_loss = 3.19 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 3.6 s / 3.5 s (0.98) | 884 MiB | 0.12 s / 0.12 s | 29 MiB | held_out_loss = 3.18; entropy_floor_approx = 0.89 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 4.0 s / 3.9 s (0.98) | 883 MiB | 0.40 s / 0.41 s | 27 MiB | kernel = lut; packed_held_out_loss = 3.18; ms_per_32x63_batch = 22 | 108 KiB | 1,648 KiB |
| 01 train+pack, BitNet | train | ✅ | 13.6 s / 13.7 s (1.01) | 979 MiB | 9.79 s / 10.00 s | 141 MiB | steps = 60; final_train_eval_loss = 3.26 | 110 KiB | 1,618 KiB |
|  | test | ✅ | 3.4 s / 3.4 s (0.98) | 881 MiB | 0.18 s / 0.18 s | 35 MiB | held_out_loss = 3.26; entropy_floor_approx = 0.89 | 110 KiB | 1,618 KiB |
|  | infer | ✅ | 4.2 s / 4.1 s (0.98) | 875 MiB | 0.73 s / 0.73 s | 29 MiB | kernel = unpack; packed_held_out_loss = 3.26; ms_per_32x63_batch = 41.6 | 130 KiB | 1,618 KiB |
| 01 train+pack, BitNet, LUT kernel | train | ✅ | 12.5 s / 12.7 s (1.01) | 971 MiB | 9.22 s / 9.44 s | 138 MiB | steps = 60; final_train_eval_loss = 3.26 | 110 KiB | 1,618 KiB |
|  | test | ✅ | 3.5 s / 3.5 s (0.99) | 881 MiB | 0.20 s / 0.20 s | 36 MiB | held_out_loss = 3.26; entropy_floor_approx = 0.89 | 110 KiB | 1,618 KiB |
|  | infer | ✅ | 3.9 s / 3.8 s (0.98) | 879 MiB | 0.67 s / 0.66 s | 33 MiB | kernel = lut; packed_held_out_loss = 3.26; ms_per_32x63_batch = 38.6 | 131 KiB | 1,618 KiB |
| 01 train+pack, FP | train | ✅ | 8.4 s / 8.4 s (1.01) | 919 MiB | 5.10 s / 5.26 s | 95 MiB | steps = 60; final_train_eval_loss = 3.14 | 805 KiB | 1,618 KiB |
|  | test | ✅ | 3.3 s / 3.3 s (0.99) | 880 MiB | 0.10 s / 0.10 s | 26 MiB | held_out_loss = 3.13; entropy_floor_approx = 0.89 | 805 KiB | 1,618 KiB |
|  | infer | ✅ | 4.6 s / 4.5 s (0.99) | 880 MiB | 0.42 s / 0.43 s | 27 MiB | kernel = dense; packed_held_out_loss = 3.13; ms_per_32x63_batch = 24.7 | 805 KiB | 1,618 KiB |
| 02 recurrent mixers (stateful LMs) | train | ✅ | 10.5 s / 10.6 s (1.00) | 949 MiB | 6.29 s / 6.43 s | 96 MiB | last_loss = {gdn: 3.28, mamba2: 3.29, gla: 3.29} | 1,068 KiB | 2,189 KiB |
|  | test | ✅ | 4.2 s / 4.1 s (0.98) | 900 MiB | 0.15 s / 0.14 s | 34 MiB | held_out_loss = {gdn: 3.25, mamba2: 3.25, gla: 3.29} | 1,068 KiB | 2,189 KiB |
|  | infer | ✅ | 3.8 s / 3.7 s (0.98) | 879 MiB | 0.24 s / 0.24 s | 11 MiB | ms_per_decoded_token_incl_prefill = {gdn: 2.24, mamba2: 1.15, gla: 0.622}; state_bytes = {gdn: 8192, mamba2: 16384, gla: 8192}; attention_kv_bytes_same_length = 65536 | 1,068 KiB | 2,189 KiB |
| 02 recurrent mixers, LittleBit, LUT decode | train | ✅ | 9.3 s / 9.3 s (0.99) | 952 MiB | 5.81 s / 5.92 s | 96 MiB | last_loss = {gdn: 3.45, mamba2: 3.44, gla: 3.44} | 163 KiB | 920 KiB |
|  | test | ✅ | 4.0 s / 3.9 s (0.98) | 885 MiB | 0.15 s / 0.15 s | 33 MiB | held_out_loss = {gdn: 3.46, mamba2: 3.43, gla: 3.45} | 163 KiB | 920 KiB |
|  | infer | ✅ | 4.5 s / 4.5 s (0.99) | 881 MiB | 0.32 s / 0.34 s | 12 MiB | ms_per_decoded_token_incl_prefill = {gdn: 2.43, mamba2: 1.29, gla: 0.882}; state_bytes = {gdn: 8192, mamba2: 16384, gla: 8192}; attention_kv_bytes_same_length = 65536 | 207 KiB | 920 KiB |
| 03 state rollback | train | ✅ | 4.8 s / 4.7 s (0.98) | 926 MiB | 1.00 s / 1.00 s | 77 MiB | steps = 20; last_loss = 3.16 | 213 KiB | 440 KiB |
|  | test | ✅ | 4.2 s / 4.2 s (0.98) | 892 MiB | 0.05 s / 0.05 s | 34 MiB | held_out_loss = 3.17 | 213 KiB | 440 KiB |
|  | infer | ✅ | 3.5 s / 3.5 s (0.99) | 878 MiB | 0.04 s / 0.04 s | 13 MiB | drafts = 5; accepted = 2; rolled_back_state_max_diff_vs_clean = 1.79e-07 | 213 KiB | 440 KiB |
| 04 decision, LittleBit GDN | train | ✅ | 15.5 s / 15.4 s (0.99) | 1124 MiB | 11.18 s / 11.24 s | 266 MiB | steps = 30; last_loss = 3.82 | 282 KiB | 1,003 KiB |
|  | test | ✅ | 4.2 s / 4.2 s (0.99) | 943 MiB | 0.61 s / 0.61 s | 89 MiB | choice_acc = 0.246; choice_ece = 0.036; choice_brier = 0.754 | 282 KiB | 1,003 KiB |
|  | infer | ✅ | 4.2 s / 4.2 s (0.98) | 889 MiB | 0.23 s / 0.23 s | 23 MiB | prefill_ms_per_state = 7.46; ms_per_question_batched = 0.594 | 282 KiB | 1,003 KiB |
| 04 decision, BitNet GDN | train | ✅ | 14.3 s / 14.4 s (1.01) | 1150 MiB | 10.24 s / 10.48 s | 290 MiB | steps = 30; last_loss = 3.87 | 299 KiB | 1,461 KiB |
|  | test | ✅ | 4.4 s / 4.4 s (0.99) | 927 MiB | 0.95 s / 0.96 s | 111 MiB | choice_acc = 0.258; choice_ece = 0.0696; choice_brier = 0.756 | 299 KiB | 1,461 KiB |
|  | infer | ✅ | 4.2 s / 4.2 s (0.98) | 887 MiB | 0.29 s / 0.29 s | 31 MiB | prefill_ms_per_state = 8.93; ms_per_question_batched = 0.74 | 299 KiB | 1,461 KiB |
| 04 decision, FP GDN | train | ✅ | 12.5 s / 12.5 s (1.00) | 1124 MiB | 8.51 s / 8.71 s | 258 MiB | steps = 30; last_loss = 3.85 | 721 KiB | 1,461 KiB |
|  | test | ✅ | 4.8 s / 4.8 s (0.99) | 926 MiB | 0.86 s / 0.88 s | 118 MiB | choice_acc = 0.266; choice_ece = 0.0336; choice_brier = 0.754 | 721 KiB | 1,461 KiB |
|  | infer | ✅ | 3.6 s / 3.5 s (0.98) | 895 MiB | 0.20 s / 0.20 s | 28 MiB | prefill_ms_per_state = 6.82; ms_per_question_batched = 0.495 | 721 KiB | 1,461 KiB |
| 04 decision, LittleBit Mamba2 | train | ✅ | 17.4 s / 17.6 s (1.01) | 1177 MiB | 13.63 s / 13.97 s | 301 MiB | steps = 30; last_loss = 3.82 | 382 KiB | 1,150 KiB |
|  | test | ✅ | 4.1 s / 4.1 s (0.98) | 944 MiB | 0.70 s / 0.69 s | 80 MiB | choice_acc = 0.281; choice_ece = 0.00828; choice_brier = 0.753 | 382 KiB | 1,150 KiB |
|  | infer | ✅ | 3.8 s / 3.8 s (0.99) | 889 MiB | 0.21 s / 0.22 s | 24 MiB | prefill_ms_per_state = 8.03; ms_per_question_batched = 0.509 | 382 KiB | 1,150 KiB |
| 05 pack + LUT + Helion (interpret) | train | ✅ | 8.8 s / 8.9 s (1.00) | 944 MiB | 5.55 s / 5.67 s | 102 MiB | steps = 60; final_train_eval_loss = 3.29 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 3.6 s / 3.6 s (0.99) | 880 MiB | 0.04 s / 0.05 s | 21 MiB | held_out_loss = 3.28; entropy_floor = 0.891 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 10.2 s / 10.4 s (1.02) | 996 MiB | 6.02 s / 6.39 s | 142 MiB | unpacked_weight_ram_bytes = 1641024; packed_checkpoint_bytes = 163769; torch = {held_out_loss: 3.3, max_logit_diff: 0.00177, same_greedy: True} | 103 KiB | 1,648 KiB |
| 06 looped LM, FP | train | ✅ | 16.0 s / 16.2 s (1.01) | 1099 MiB | 11.85 s / 12.15 s | 271 MiB | steps = 40; last_loss = 3.09; mean_sampled_depth = 7.3 | 921 KiB | 1,860 KiB |
|  | test | ✅ | 5.5 s / 5.5 s (0.99) | 881 MiB | 1.26 s / 1.28 s | 28 MiB | held_out_loss_by_depth = {1: 3.23, 2: 3.13, 4: 3.12} | 921 KiB | 1,860 KiB |
|  | infer | ✅ | 3.4 s / 3.3 s (0.99) | 877 MiB | 0.21 s / 0.21 s | 29 MiB | adaptive_exit_held_out_loss = 3.12; adaptive_mean_loops = 6.05; r4_latent_loop_mflop = 2.61 | 921 KiB | 1,860 KiB |
| 06 looped LM, LittleBit core | train | ✅ | 13.5 s / 13.7 s (1.01) | 1071 MiB | 9.97 s / 10.25 s | 274 MiB | steps = 40; last_loss = 3.11; mean_sampled_depth = 7.3 | 516 KiB | 1,444 KiB |
|  | test | ✅ | 4.4 s / 4.4 s (1.00) | 887 MiB | 1.01 s / 1.02 s | 21 MiB | held_out_loss_by_depth = {1: 3.27, 2: 3.16, 4: 3.14} | 516 KiB | 1,444 KiB |
|  | infer | ✅ | 3.7 s / 3.6 s (0.98) | 881 MiB | 0.19 s / 0.20 s | 22 MiB | adaptive_exit_held_out_loss = 3.14; adaptive_mean_loops = 6.55; r4_latent_loop_mflop = 2.61 | 516 KiB | 1,444 KiB |
| 06 looped LM, Ouro exit gate | train | ✅ | 15.5 s / 15.6 s (1.01) | 1155 MiB | 11.91 s / 12.15 s | 325 MiB | steps = 40; last_loss = 3.05; mean_sampled_depth = 7.3 | 922 KiB | 1,861 KiB |
|  | test | ✅ | 5.0 s / 4.9 s (0.99) | 882 MiB | 1.28 s / 1.28 s | 27 MiB | held_out_loss_by_depth = {1: 3.2, 2: 3.13, 4: 3.12} | 922 KiB | 1,861 KiB |
|  | infer | ✅ | 3.9 s / 3.8 s (0.98) | 876 MiB | 0.11 s / 0.11 s | 28 MiB | adaptive_exit_held_out_loss = 3.13; adaptive_mean_loops = 2.15; r4_latent_loop_mflop = 2.61 | 922 KiB | 1,861 KiB |
| 07 byte-level: baseline / BLT / H-Net | train | ✅ | 44.9 s / 46.4 s (1.03) | 1144 MiB | 40.98 s / 42.61 s | 372 MiB | steps_per_model = 30 | 5,944 KiB | 11,939 KiB |
|  | test | ✅ | 4.1 s / 4.1 s (0.99) | 904 MiB | 0.31 s / 0.31 s | 39 MiB | byte transformer = {bits_per_byte: 4.43, bytes_per_patch: 1, inner_len: 256}; BLT = {bits_per_byte: 5.1, bytes_per_patch: 4.35, inner_len: 58.9}; H-Net = {bits_per_byte: 4.25, bytes_per_patch: 2.83, inner_len: 116} | 5,944 KiB | 11,939 KiB |
|  | infer | ✅ | 5.0 s / 4.9 s (0.99) | 915 MiB | 1.09 s / 1.09 s | 37 MiB | ms_per_16xseq = {byte transformer: 88.4, BLT (incl. entropy patching): 90.9, H-Net: 83.2}; embedding_params_bytes_vocab = 65536; embedding_params_200k_vocab_at_d128 = 51200000 | 5,944 KiB | 11,939 KiB |
| 08 LLM-JEPA (LM / online / EMA / EMA+dropout) | train | ✅ | 6.7 s / 6.6 s (0.98) | 895 MiB | 2.33 s / 2.35 s | 29 MiB |  | 788 KiB | 2,412 KiB |
|  | test | ✅ | 4.4 s / 4.3 s (0.98) | 879 MiB | 0.05 s / 0.05 s | 13 MiB | LM loss only = {code_loss: 0.0795, code_token_acc: 1, retrieval_top1: 0.0156}; LLM-JEPA, online target = {code_loss: 0.481, code_token_acc: 0.936, retrieval_top1: 0.188}; LLM-JEPA, EMA target = {code_loss: 0.108, code_token_acc: 1, retrieval_top1: 0.0469} | 788 KiB | 2,412 KiB |
|  | infer | ✅ | 4.4 s / 4.3 s (0.98) | 877 MiB | 0.16 s / 0.15 s | 11 MiB | LM loss only = {ms_per_64_pairs: 3.87}; LLM-JEPA, online target = {ms_per_64_pairs: 6.83}; LLM-JEPA, EMA target = {ms_per_64_pairs: 6.84} | 788 KiB | 2,412 KiB |
| 09 looped index LM (5 stages + decode modes) | train | ✅ | 18.4 s / 18.5 s (1.01) | 1080 MiB | 13.87 s / 14.16 s | 303 MiB |  | 908 KiB | 1,990 KiB |
|  | test | ✅ | 5.8 s / 5.7 s (0.99) | 907 MiB | 1.47 s / 1.50 s | 134 MiB | lean = {loss: 2.27, recall_acc: 0, two_hop_acc: 0}; fp_dense_ref = {loss: 2.19, recall_acc: 0, two_hop_acc: 0} | 908 KiB | 1,990 KiB |
|  | infer | ✅ | 5.2 s / 5.2 s (0.99) | 887 MiB | 1.45 s / 1.52 s | 11 MiB | kernel = native; greedy_distinct_tokens = 2; memory_tiers = {within_token_loop_memory_bytes_per_position: 768, gdn_state_bytes: 40960, summary_bytes: 256} | 908 KiB | 1,990 KiB |
| 09 looped index LM, LUT decode | train | ✅ | 17.0 s / 17.4 s (1.03) | 1085 MiB | 13.06 s / 13.60 s | 306 MiB |  | 908 KiB | 1,990 KiB |
|  | test | ✅ | 5.1 s / 5.1 s (1.00) | 909 MiB | 1.25 s / 1.29 s | 132 MiB | lean = {loss: 2.27, recall_acc: 0, two_hop_acc: 0}; fp_dense_ref = {loss: 2.19, recall_acc: 0, two_hop_acc: 0} | 908 KiB | 1,990 KiB |
|  | infer | ✅ | 5.9 s / 5.9 s (0.99) | 889 MiB | 1.78 s / 1.86 s | 12 MiB | kernel = lut; lut_greedy_same_as_native = True; greedy_distinct_tokens = 2 | 935 KiB | 1,990 KiB |
| 09 looped index LM, greedy bpw allocation | train | ✅ | 18.4 s / 18.8 s (1.02) | 1096 MiB | 14.19 s / 14.76 s | 309 MiB |  | 908 KiB | 1,990 KiB |
|  | test | ✅ | 5.2 s / 5.1 s (0.98) | 906 MiB | 1.32 s / 1.35 s | 132 MiB | lean = {loss: 2.27, recall_acc: 0, two_hop_acc: 0}; fp_dense_ref = {loss: 2.19, recall_acc: 0, two_hop_acc: 0} | 908 KiB | 1,990 KiB |
|  | infer | ✅ | 5.8 s / 5.8 s (0.99) | 887 MiB | 1.68 s / 1.75 s | 11 MiB | kernel = native; greedy_distinct_tokens = 2; memory_tiers = {within_token_loop_memory_bytes_per_position: 768, gdn_state_bytes: 40960, summary_bytes: 256} | 908 KiB | 1,990 KiB |
