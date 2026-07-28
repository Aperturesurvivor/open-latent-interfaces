# Phase 0 Executive Summary

Open Latent Interfaces now has a tested, reproducible frozen-model
instrumentation stack and one completed local pilot on Qwen2.5-0.5B.

The pilot did **not** demonstrate a deterministic latent graft.

Across 640 pair-disjoint prompts, addition-versus-contrast routing was perfectly
linearly separable at all five inspected boundaries, but the current dataset
does not support a robust-routing claim because lexical template cues were not
held out. Approximate scalar-sum information was present (best test R² 0.915),
yet exact rounded recovery was 0/32.

At the development-selected final residual boundary, the frozen model produced
the correct first result digit on 15/32 test additions. A probe-defined
intervention toward the exact leading digit reached 16/32; a shuffled-result
control reached 14/32. The targeted intervention improved the correct-token
margin by only 0.040 logits relative to shuffled control.

The outcome is an infrastructure success and a weak causal pilot signal, not a
scientific pass. The next gate is a balanced, template-held-out, layer-wide
comparison of probe, donor-patch, logit/Jacobian, and eventually NLA-derived
read/write directions, with all selection confined to development data before
a new frozen audit.

See:

- [complete lab notebook](PHASE0_LAB_NOTEBOOK.md);
- [pilot protocol](protocols/PHASE0_NATIVE_MATH_CHANNELS.md);
- [research ladder](docs/RESEARCH_PROGRAM.md);
- [claims boundary](docs/PRIOR_ART_AND_CLAIMS.md);
- [`results/phase0_qwen05b_pilot.json`](results/phase0_qwen05b_pilot.json).

