| example | phase | ok | process wall / CPU (util) | process peak RSS | phase wall / CPU | phase peak RSS growth | key metrics | deployed | checkpoint file |
|---|---|---|---|---|---|---|---|---|---|
| 01 train+pack, LittleBit | train | ✅ | 10.8 s / 11.0 s (1.02) | 938 MiB | 7.29 s / 7.56 s | 99 MiB | steps = 60; final_train_eval_loss = 3.19 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 3.6 s / 3.5 s (0.99) | 884 MiB | 0.11 s / 0.12 s | 29 MiB | held_out_loss = 3.18; entropy_floor_approx = 0.89 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 4.6 s / 4.5 s (0.98) | 881 MiB | 0.60 s / 0.63 s | 26 MiB | kernel = unpack; packed_held_out_loss = 3.18; ms_per_32x63_batch = 28.8 | 103 KiB | 1,648 KiB |
| 01 train+pack, LittleBit, LUT kernel | train | ✅ | 10.8 s / 10.9 s (1.01) | 941 MiB | 7.17 s / 7.42 s | 102 MiB | steps = 60; final_train_eval_loss = 3.19 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 3.7 s / 3.7 s (0.99) | 883 MiB | 0.11 s / 0.12 s | 29 MiB | held_out_loss = 3.18; entropy_floor_approx = 0.89 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 4.1 s / 4.1 s (0.99) | 882 MiB | 0.39 s / 0.40 s | 27 MiB | kernel = lut; packed_held_out_loss = 3.18; ms_per_32x63_batch = 21.5 | 108 KiB | 1,648 KiB |
| 01 train+pack, BitNet | train | ✅ | 14.1 s / 14.3 s (1.02) | 937 MiB | 9.98 s / 10.33 s | 135 MiB | steps = 60; final_train_eval_loss = 3.26 | 110 KiB | 1,618 KiB |
|  | test | ✅ | 4.5 s / 4.4 s (0.99) | 880 MiB | 0.23 s / 0.23 s | 35 MiB | held_out_loss = 3.26; entropy_floor_approx = 0.89 | 110 KiB | 1,618 KiB |
|  | infer | ✅ | 4.1 s / 4.1 s (0.99) | 875 MiB | 0.70 s / 0.73 s | 29 MiB | kernel = unpack; packed_held_out_loss = 3.26; ms_per_32x63_batch = 39.4 | 130 KiB | 1,618 KiB |
| 01 train+pack, BitNet, LUT kernel | train | ✅ | 14.4 s / 14.6 s (1.02) | 950 MiB | 10.10 s / 10.44 s | 133 MiB | steps = 60; final_train_eval_loss = 3.26 | 110 KiB | 1,618 KiB |
|  | test | ✅ | 3.9 s / 3.9 s (0.99) | 879 MiB | 0.21 s / 0.22 s | 35 MiB | held_out_loss = 3.26; entropy_floor_approx = 0.89 | 110 KiB | 1,618 KiB |
|  | infer | ✅ | 4.2 s / 4.2 s (0.99) | 879 MiB | 0.67 s / 0.70 s | 33 MiB | kernel = lut; packed_held_out_loss = 3.26; ms_per_32x63_batch = 39.6 | 131 KiB | 1,618 KiB |
| 01 train+pack, FP | train | ✅ | 9.2 s / 9.3 s (1.01) | 915 MiB | 5.71 s / 5.88 s | 98 MiB | steps = 60; final_train_eval_loss = 3.14 | 805 KiB | 1,618 KiB |
|  | test | ✅ | 4.2 s / 4.2 s (0.99) | 880 MiB | 0.14 s / 0.14 s | 27 MiB | held_out_loss = 3.13; entropy_floor_approx = 0.89 | 805 KiB | 1,618 KiB |
|  | infer | ✅ | 3.9 s / 3.9 s (0.99) | 880 MiB | 0.38 s / 0.39 s | 26 MiB | kernel = dense; packed_held_out_loss = 3.13; ms_per_32x63_batch = 23 | 805 KiB | 1,618 KiB |
| 02 recurrent mixers (stateful LMs) | train | ✅ | 11.2 s / 11.3 s (1.01) | 953 MiB | 7.26 s / 7.51 s | 95 MiB | last_loss = {gdn: 3.28, mamba2: 3.29, gla: 3.29} | 1,068 KiB | 2,189 KiB |
|  | test | ✅ | 4.3 s / 4.2 s (0.99) | 885 MiB | 0.14 s / 0.15 s | 26 MiB | held_out_loss = {gdn: 3.25, mamba2: 3.25, gla: 3.29} | 1,068 KiB | 2,189 KiB |
|  | infer | ✅ | 4.3 s / 4.3 s (0.99) | 878 MiB | 0.22 s / 0.22 s | 10 MiB | ms_per_decoded_token_incl_prefill = {gdn: 1.89, mamba2: 0.988, gla: 0.536}; state_bytes = {gdn: 8192, mamba2: 16384, gla: 8192}; attention_kv_bytes_same_length = 65536 | 1,068 KiB | 2,189 KiB |
| 03 state rollback | train | ✅ | 6.8 s / 6.8 s (1.01) | 939 MiB | 2.55 s / 2.72 s | 87 MiB | steps = 20; last_loss = 3.16 | 213 KiB | 440 KiB |
|  | test | ✅ | 4.1 s / 4.1 s (0.98) | 891 MiB | 0.05 s / 0.05 s | 26 MiB | held_out_loss = 3.17 | 213 KiB | 440 KiB |
|  | infer | ✅ | 4.0 s / 3.9 s (0.98) | 876 MiB | 0.09 s / 0.09 s | 10 MiB | drafts = 5; accepted = 2; rolled_back_state_max_diff_vs_clean = 1.01e-07 | 213 KiB | 440 KiB |
| 04 decision, LittleBit GDN | train | ✅ | 22.3 s / 22.6 s (1.02) | 1233 MiB | 17.58 s / 18.16 s | 403 MiB | steps = 30; last_loss = 3.82 | 282 KiB | 1,003 KiB |
|  | test | ✅ | 5.6 s / 5.6 s (0.99) | 927 MiB | 1.07 s / 1.10 s | 71 MiB | choice_acc = 0.219; choice_ece = 0.0627; choice_brier = 0.753 | 282 KiB | 1,003 KiB |
|  | infer | ✅ | 4.7 s / 4.7 s (1.00) | 889 MiB | 0.34 s / 0.35 s | 22 MiB | prefill_ms_per_state = 16.2; ms_per_question_batched = 0.778 | 282 KiB | 1,003 KiB |
| 04 decision, BitNet GDN | train | ✅ | 20.4 s / 20.9 s (1.03) | 1252 MiB | 16.09 s / 16.78 s | 412 MiB | steps = 30; last_loss = 3.83 | 299 KiB | 1,461 KiB |
|  | test | ✅ | 5.2 s / 5.1 s (0.99) | 887 MiB | 0.97 s / 1.01 s | 99 MiB | choice_acc = 0.246; choice_ece = 0.0367; choice_brier = 0.755 | 299 KiB | 1,461 KiB |
|  | infer | ✅ | 3.6 s / 3.6 s (0.99) | 886 MiB | 0.29 s / 0.30 s | 30 MiB | prefill_ms_per_state = 15.8; ms_per_question_batched = 0.633 | 299 KiB | 1,461 KiB |
| 04 decision, FP GDN | train | ✅ | 16.4 s / 16.8 s (1.02) | 1223 MiB | 12.73 s / 13.24 s | 368 MiB | steps = 30; last_loss = 3.85 | 721 KiB | 1,461 KiB |
|  | test | ✅ | 4.1 s / 4.1 s (1.00) | 897 MiB | 0.78 s / 0.81 s | 91 MiB | choice_acc = 0.266; choice_ece = 0.0349; choice_brier = 0.754 | 721 KiB | 1,461 KiB |
|  | infer | ✅ | 4.3 s / 4.3 s (0.99) | 885 MiB | 0.25 s / 0.26 s | 28 MiB | prefill_ms_per_state = 13.4; ms_per_question_batched = 0.559 | 721 KiB | 1,461 KiB |
| 04 decision, LittleBit Mamba2 | train | ✅ | 16.7 s / 16.9 s (1.02) | 1155 MiB | 12.29 s / 12.69 s | 312 MiB | steps = 30; last_loss = 3.82 | 382 KiB | 1,150 KiB |
|  | test | ✅ | 4.3 s / 4.2 s (0.99) | 929 MiB | 0.68 s / 0.70 s | 87 MiB | choice_acc = 0.281; choice_ece = 0.00828; choice_brier = 0.753 | 382 KiB | 1,150 KiB |
|  | infer | ✅ | 3.7 s / 3.7 s (0.99) | 892 MiB | 0.20 s / 0.21 s | 25 MiB | prefill_ms_per_state = 7.92; ms_per_question_batched = 0.5 | 382 KiB | 1,150 KiB |
| 05 pack + LUT + Helion (interpret) | train | ✅ | 11.4 s / 11.5 s (1.01) | 943 MiB | 6.91 s / 7.18 s | 102 MiB | steps = 60; final_train_eval_loss = 3.29 | 101 KiB | 1,648 KiB |
|  | test | ✅ | 4.1 s / 4.1 s (0.99) | 878 MiB | 0.05 s / 0.05 s | 21 MiB | held_out_loss = 3.28; entropy_floor = 0.891 | 101 KiB | 1,648 KiB |
|  | infer | ✅ | 10.1 s / 10.1 s (1.01) | 995 MiB | 5.81 s / 6.02 s | 142 MiB | unpacked_weight_ram_bytes = 1641024; packed_checkpoint_bytes = 163769; torch = {held_out_loss: 3.3, max_logit_diff: 0.00177, same_greedy: True} | 103 KiB | 1,648 KiB |
| 06 looped LM, FP | train | ✅ | 15.7 s / 16.1 s (1.03) | 1080 MiB | 11.97 s / 12.46 s | 266 MiB | steps = 40; last_loss = 3.09; mean_sampled_depth = 7.3 | 921 KiB | 1,860 KiB |
|  | test | ✅ | 5.6 s / 5.6 s (0.99) | 881 MiB | 1.42 s / 1.47 s | 27 MiB | held_out_loss_by_depth = {1: 3.23, 2: 3.13, 4: 3.12} | 921 KiB | 1,860 KiB |
|  | infer | ✅ | 4.1 s / 4.1 s (0.99) | 876 MiB | 0.23 s / 0.24 s | 29 MiB | adaptive_exit_held_out_loss = 3.12; adaptive_mean_loops = 6.05; r4_latent_loop_mflop = 2.61 | 921 KiB | 1,860 KiB |
| 06 looped LM, LittleBit core | train | ✅ | 17.3 s / 17.7 s (1.02) | 1131 MiB | 13.29 s / 13.79 s | 277 MiB | steps = 40; last_loss = 3.11; mean_sampled_depth = 7.3 | 516 KiB | 1,444 KiB |
|  | test | ✅ | 5.6 s / 5.6 s (0.99) | 881 MiB | 1.31 s / 1.35 s | 21 MiB | held_out_loss_by_depth = {1: 3.27, 2: 3.16, 4: 3.14} | 516 KiB | 1,444 KiB |
|  | infer | ✅ | 4.6 s / 4.6 s (1.00) | 881 MiB | 0.26 s / 0.27 s | 22 MiB | adaptive_exit_held_out_loss = 3.14; adaptive_mean_loops = 6.55; r4_latent_loop_mflop = 2.61 | 516 KiB | 1,444 KiB |
| 06 looped LM, Ouro exit gate | train | ✅ | 18.5 s / 19.0 s (1.03) | 1178 MiB | 14.30 s / 14.94 s | 324 MiB | steps = 40; last_loss = 3.05; mean_sampled_depth = 7.3 | 922 KiB | 1,861 KiB |
|  | test | ✅ | 5.7 s / 5.7 s (1.00) | 882 MiB | 1.41 s / 1.46 s | 27 MiB | held_out_loss_by_depth = {1: 3.2, 2: 3.13, 4: 3.12} | 922 KiB | 1,861 KiB |
|  | infer | ✅ | 3.9 s / 3.9 s (0.99) | 875 MiB | 0.10 s / 0.10 s | 28 MiB | adaptive_exit_held_out_loss = 3.13; adaptive_mean_loops = 2.15; r4_latent_loop_mflop = 2.61 | 922 KiB | 1,861 KiB |
| 07 byte-level: baseline / BLT / H-Net | train | ✅ | 42.4 s / 45.8 s (1.08) | 1170 MiB | 38.73 s / 42.26 s | 376 MiB | steps_per_model = 30 | 5,944 KiB | 11,939 KiB |
|  | test | ✅ | 3.7 s / 3.6 s (0.99) | 916 MiB | 0.25 s / 0.26 s | 39 MiB | byte transformer = {bits_per_byte: 4.48, bytes_per_patch: 1, inner_len: 256}; BLT = {bits_per_byte: 4.97, bytes_per_patch: 4.41, inner_len: 58.1}; H-Net = {bits_per_byte: 4.28, bytes_per_patch: 3.44, inner_len: 93} | 5,944 KiB | 11,939 KiB |
|  | infer | ✅ | 4.5 s / 4.4 s (1.00) | 914 MiB | 1.00 s / 1.03 s | 39 MiB | ms_per_16xseq = {byte transformer: 85.5, BLT (incl. entropy patching): 80.2, H-Net: 81.2}; embedding_params_bytes_vocab = 65536; embedding_params_200k_vocab_at_d128 = 51200000 | 5,944 KiB | 11,939 KiB |
| 08 LLM-JEPA (LM / online / EMA / EMA+dropout) | train | ✅ | 6.3 s / 6.3 s (1.00) | 893 MiB | 2.15 s / 2.26 s | 29 MiB |  | 788 KiB | 2,412 KiB |
|  | test | ✅ | 4.3 s / 4.3 s (0.98) | 879 MiB | 0.04 s / 0.05 s | 13 MiB | LM loss only = {code_loss: 0.0795, code_token_acc: 1, retrieval_top1: 0.0156}; LLM-JEPA, online target = {code_loss: 0.481, code_token_acc: 0.936, retrieval_top1: 0.188}; LLM-JEPA, EMA target = {code_loss: 0.108, code_token_acc: 1, retrieval_top1: 0.0469} | 788 KiB | 2,412 KiB |
|  | infer | ✅ | 4.3 s / 4.3 s (1.00) | 877 MiB | 0.10 s / 0.11 s | 11 MiB | LM loss only = {ms_per_64_pairs: 4.09}; LLM-JEPA, online target = {ms_per_64_pairs: 4.21}; LLM-JEPA, EMA target = {ms_per_64_pairs: 4.17} | 788 KiB | 2,412 KiB |
| 09 looped index LM (5 stages + decode modes) | train | ✅ | 35.0 s / 37.5 s (1.07) | 1204 MiB | 30.99 s / 33.67 s | 384 MiB |  | 908 KiB | 1,990 KiB |
|  | test | ✅ | 4.6 s / 4.6 s (1.00) | 889 MiB | 0.66 s / 0.68 s | 30 MiB | lean = {loss: 2.22, recall_acc: 0, two_hop_acc: 0}; fp_dense_ref = {loss: 2.21, recall_acc: 0, two_hop_acc: 0} | 908 KiB | 1,990 KiB |
|  | infer | ✅ | 5.2 s / 5.1 s (0.99) | 883 MiB | 1.36 s / 1.41 s | 7 MiB | greedy_distinct_tokens = 1; memory_tiers = {within_token_loop_memory_bytes_per_position: 768, gdn_state_bytes: 40960, summary_bytes: 256} | 908 KiB | 1,990 KiB |
