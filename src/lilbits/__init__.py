"""lilbits: primitives for small, lean AI models.

    lilbits.layers     LittleBitLinear, BitLinear (b1.58), SwiGLU
    lilbits.mixers     softmax Attention; GatedDeltaNet, Mamba2, GLA, mLSTM, minGRU (state in / state out)
    lilbits.models     TinyLM + config/factories, decision-model parts (stateful blocks, typed readout)
    lilbits.inference  bit-packed LittleBit, Helion kernels (packed BitNet lives with BitLinear)
    lilbits.training   Fabric training loop, Muon variants, Optuna tuning
    lilbits.data       synthetic Markov and char-level text sources
    lilbits.measure    wall / CPU / peak RSS meter, size reports
"""
