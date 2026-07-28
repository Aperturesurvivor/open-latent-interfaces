# Phase 5 Protocol: Qwen Arithmetic-State Replication

## Question

Does the audited Phi arithmetic-state workflow rediscover causally writable
operand and carry coordinates in Qwen2.5-1.5B without transferring any Phi
vector, layer, scale, or fitted artifact?

## Frozen model

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- base-model weights: frozen
- assistant prefill: `Answer=`

## Corpus and isolation

The matched quartet generator and split sizes are identical to Phase 4 so the
causal estimand is comparable. Operand pairs remain disjoint across splits and
were excluded from earlier Qwen answer-channel corpora. Phase 5 fits new Qwen
states only; Phi tensors are evidence and design history, never inputs.

The audit split remains sealed. Behavior is evaluated on fit, selection, and
development only, under the same 90% row and 75% complete-quartet thresholds.

## Replication ladder

1. verify untouched behavior and the Qwen digit-token contract;
2. derive and hash-lock Qwen-specific semantic token positions;
3. scan sequence and token-local causal boundaries with matched no-carry and
   isotropic controls;
4. fit donor-free Qwen operand and carry transports on eligible fit quartets;
5. choose boundaries, writer form, and scales on selection only;
6. validate once on untouched development;
7. authorize exactly one audit only after every artifact and metric is frozen;
8. package vectors only if the sealed audit passes.

## Evidence rule

Workflow portability requires comparable causal and held-out evidence in both
families. It does not require matching layers, vector directions, ranks, or
effect sizes. Direct Phi-to-Qwen vector transfer is outside scope and must not
be attempted or implied.

## Behavior result

The frozen non-audit behavior gate passed in every split: 95.56% fit rows,
95.00% selection rows, and 95.56% development rows were exact, with
158/180, 39/45, and 41/45 complete-correct quartets respectively. The eligible
fit pool therefore advances to causal mapping without a behavior amendment.

## Frozen token-region scan

All 45 selection quartets have one changed first-operand digit token and one
carry-context second-operand digit token under the Qwen renderer. Their
positions and sequence lengths are hash-locked. Hidden-state indices
1, 4, 8, 12, 16, 20, 24, and 27 span the Qwen stack.

At every boundary, the scan separately tests the changed operand token, the
carry-context token, and the downstream tail. Each target has an isotropic
region- and norm-matched control; carry regions additionally face the matched
no-carry `+1` regional delta. Selection gates and thresholds are identical to
the Phase 4 regional map.

## Regional result and bounded follow-up

The generic operand token passed at Qwen hidden-state index 12 with 43/45
exact targets against 0/45 random. The carry-context token localized at index
16 but failed at unit scale: 11/45 target tens against 2/45 matched no-carry
and 0/45 random. The downstream tail did not transfer.

A scale-only follow-up may keep index 16 and the exact carry-context,
matched-no-carry, and isotropic vectors fixed while evaluating a precommitted
bounded scale grid. It may not rescan layers, change token regions, filter
selection, or open development.

The scale-only follow-up selected scale 2.0 as the smallest passing value:
26/45 target tens against 7/45 matched no-carry and 0/45 random. Qwen
donor-free fitting is therefore fixed at operand index 12 and carry-context
index 16. Fit data may determine class prototypes; selection may choose only
from a frozen scale grid and matched controls.

The donor-free viability gate fits full-width mean transport vectors per
source-digit class from the 158 complete-correct fit quartets. Qwen selection
evaluates scales 0.5, 1.0, 1.5, 2.0, and 3.0. The operand target must beat
wrong-class and isotropic controls; the carry target must beat matched
no-carry, wrong-class, and isotropic controls. The smallest scale satisfying
all gates is selected. Development remains unopened.

The donor-free operand writer passed at index 12 and scale 1.0 with 40/45
exact targets against 1/45 wrong-class exact and 0/45 random. The
digit-conditioned carry writer failed because wrong source-digit classes
transferred equally well. A bounded universal-vector follow-up may aggregate
the already fitted Qwen class vectors by their fit counts and compare them
with the equivalently aggregated no-carry vector and isotropic control.

The universal follow-up introduces no new fitted weights. It evaluates the
fit-count-weighted mean Qwen carry vector at fixed index 16 over scales 0.5,
1.0, 1.5, 2.0, and 3.0, selecting the smallest complete pass against the
universal no-carry and isotropic controls.
