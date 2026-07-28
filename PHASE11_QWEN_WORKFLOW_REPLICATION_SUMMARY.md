# Phase 11 Qwen Workflow Replication Summary

## Outcome

Phase 11 established a complete Qwen-specific read → deterministic compute →
hybrid write pipeline without transferring any Phi tensor, layer, centroid,
basis, margin, norm cap, or selected iteration count.

The exposed 180-example development pipeline achieved 180/180 exact outputs.
The one-shot 90-example audit achieved 90/90 exact outputs on the true
arithmetic task but did not pass the broader arbitrary-write control gate.
Phase 11 is therefore a non-passing replication audit.

## Frozen components

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- operand reader: hidden-state index 1
- host computation: exact integer addition
- leading compiler: hidden-state index 23, margin 16, relative-norm cap 0.25,
  two relinearization steps
- suffix writer: hidden-state index 27, rank 16
- tens scale: 1.25
- ones scale: 2.0

## Component selection

The reader selected the earliest passing boundary:

- held-out pairs: 180/180
- held-out digits: 988/988
- rotated-label pairs: 0/180

The one-step leading compiler did not pass. Its deterministic fallback was
frozen before iterative selection. Two relinearization steps were the earliest
passing depth:

- target: 85/90
- identity: 90/90
- strongest semantic control: 11/90
- mean relative norm: 0.2021

Iterations three and four reached 90/90 but were not selected under the frozen
earliest-passing rule.

## Exposed integration

On all 180 Phase 7 development examples:

- reader and deterministic compute: 180/180
- base: 165/180
- latent and oracle hybrid output: 180/180
- recovered base errors: 15/15
- preserved base-correct outputs: 165/165
- shuffled semantic target following: 179/180
- shuffled random target following: 7/180
- semantic target-following advantage: 172/180

The original development gate remained a non-pass because of one redundant
absolute wrong-target ceiling. A hash-locked, no-inference correction removed
only that ceiling while retaining the already frozen 50-point paired recovery
advantage. The correction authorized a fresh audit, not an audit claim.

## One-shot audit

The audit used 90 new examples:

- zero canonical-pair overlap with Phases 3, 4, 6, 7, and the Phi Phase 9E
  audit
- three unseen template families
- balanced leading, tens, and ones digits
- 45 carry and 45 non-carry examples
- exactly one authorized run

Result:

- result SHA-256:
  `1840165adbcc0083fb937fa8407fe64ca23dc3a51de14bd14b3cc423ee61d692`
- reader and deterministic compute: 90/90
- base: 85/90
- latent and oracle true-task output: 90/90
- recovered base errors: 5/5
- preserved base-correct outputs: 85/85
- random and wrong-target recovery: 1/5 each
- shuffled semantic target following: 70/90
- shuffled random target following: 0/90

The frozen audit required at least 81/90 shuffled target following. The sole
failed check was therefore substantive and is not eligible for correction.

## Failure localization

The counterfactual write failures localize primarily to the leading compiler:

- leading: 72/90
- tens: 90/90
- ones: 88/90
- 18/20 failed full outputs differed only at the leading position
- failures spanned all three new template families

The true-task path needs little intervention because the base model is usually
already correct. Arbitrary shuffled targets force large interventions and
expose template sensitivity in the selected two-step local compiler.

## Supported claim

Phase 11 supports the following narrower conclusion:

> The open workflow independently recovered a perfect Qwen latent
> read–compute–correct pipeline on fresh arithmetic examples, with strong
> semantic controls, but the selected leading compiler did not generalize
> arbitrary target writing strongly enough to pass the one-shot
> template-disjoint audit.

It does not support a model-general deterministic reasoning-implant claim.

## Next experiment

The next phase should treat the Phase 11 audit as exposed failure-analysis data
and improve compiler robustness without rerunning or rewriting that audit.
Candidate mechanisms must be selected on new development prompt families and
then judged on another pair- and template-disjoint one-shot audit.

The first justified candidate is a stronger stopping rule or iteration budget
for the leading compiler. Phase 11 already showed that three iterations reached
90/90 on selection while the earliest-passing two-step version reached 85/90.
Any new phase must reselect this mechanism prospectively and cannot simply
substitute iteration three into the completed Phase 11 audit.
