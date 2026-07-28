# Phase 4 Donor-Free Prototype Summary

## Outcome

The full-width donor-free operand writer passed. The class-specific
carry-context writer did not pass its complete control gate, despite producing
strong target behavior.

### Operand writer: pass

The selected source-digit prototype at hidden-state index 1 and scale 1.5
reached 41/45 exact target results, versus 0/45 exact for the norm-matched
wrong-class prototype and 1/45 for the isotropic norm-matched control. All
45 outputs were parseable. The writer uses only four fit-derived vectors and
a source digit; no donor activation is required during inference.

### Carry-context writer: class-specific non-pass

At hidden-state index 13 and scale 1.5, the fit-derived carry prototype reached
39/45 target tens digits and 37/45 exact target results. The matched no-carry
prototype reached 20/45 tens and 18/45 exact, while the isotropic control
remained at 1/45.

However, a wrong source-digit carry class reached 39/45 tens and 36/45 exact.
The carry target therefore did not outperform every precommitted control, and
the class-specific carry hypothesis fails.

## Interpretation

Wrong-class carry prototypes generalize almost perfectly across source digits.
This argues against four distinct digit-conditioned carry writers and favors
a simpler, approximately class-invariant carry direction. Source-digit identity
is essential for the early operand edit but appears largely irrelevant to the
later carry-context transport.

The next gate should fit one universal mean carry vector, compare it with a
universal matched no-carry vector and isotropic controls, and select causal
rank only if that simpler writer passes. This class-specific result cannot be
relabeled as a pass after seeing the data.

## Frozen provenance

- frozen experiment commit: `ea02091`
- eligible fit quartets: 127
- fit counts by source digit: 41, 30, 24, and 32
- selection quartets: 45, unfiltered
- configuration SHA-256:
  `06c2e1407d96844ab330e34240e1a6013f8dbdc89e6cecb8813df8f5887e0c3f`
- result SHA-256:
  `08ad571f8f28de1327efdc8b03e101294f864ef54dbb496ae9cb1fe97a5a72ca`
- prototype artifact SHA-256:
  `fb58ac55f804945d060e077a44a32861da0bd1b719ef0d3709e37da7fe615227`
- elapsed evaluation time: 240.95 seconds

## Claim boundary

The operand result is a donor-free selection pass. The carry result supports
a universal-direction follow-up but is a class-specific non-pass. Neither
result establishes development generalization, an individual neuron, or a
natural-language thought transcript.
