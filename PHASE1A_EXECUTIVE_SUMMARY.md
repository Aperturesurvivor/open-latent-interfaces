# Phase 1A Executive Summary

## Outcome

**Engineering pass; value-decoding non-pass.**

A real Jacobian lens was fit for every internal layer of frozen
Qwen2.5-0.5B-Instruct using 24 precommitted generic WikiText passages. The
resulting matrices are finite, provenance-hashed, and usable through the common
interpretability artifact schema.

On 72 balanced, template-held-out three-digit additions, the lens often
recognized that a numeric answer should follow. It did not reliably identify
the correct leading digit:

| Readout | Selected development layer | Correct leading digit | Median vocabulary rank | Mean correct-vs-best-wrong digit margin |
|---|---:|---:|---:|---:|
| Jacobian lens | block 19 / HF hidden state 20 | 12/72 (16.7%) | 5 | -1.582 |
| Vanilla logit lens | block 15 / HF hidden state 16 | 9/72 (12.5%) | 9,412.5 | -1.612 |
| Actual final model | block 23 / HF hidden state 24 | 7/72 (9.7%) | 5 | -1.998 |
| Balanced chance | — | 8/72 (11.1%) expected | — | — |

The Jacobian-lens 12/72 result has a one-sided binomial value of approximately
0.099 against balanced chance, before accounting for selection across 23
layers. Every layer has a negative mean correct-digit margin. This is not a
pass.

## The useful discovery

At blocks 1 and 2, the correct digit was in the top 10 vocabulary items on all
72 examples, yet digit identity accuracy was exactly chance. The lens had
decoded a **numeric-output context**, not the **specific computed value**.

That distinction is central to this research program. A readout can look
semantically impressive while carrying no discriminative information about the
typed variable we actually need.

## Behavioral gate

The frozen base model itself did not solve this task regime reliably at the
measured next-token boundary. Its correct-leading-digit top-1 accuracy was
9.7%. A correct latent result should not be presumed when the model's behavior
does not demonstrate the computation.

Phase 1 must therefore begin with a frozen capability sweep and restrict causal
cartography to task regimes where the untouched model passes a behavioral
competence threshold. Three-digit word problems remain useful as a hard
negative/control regime.

## Artifact integrity

- Corpus selection was committed in `1eadf9b` before fitting.
- Target model revision:
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Jacobian-lens revision:
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- Development lens SHA-256:
  `529bbc8b261d945bd4bb94785077b9f6bb5e545f20efd4fa236755c74a858c97`.
- Twenty real J-lens observations validate under
  `oli.interpretability-artifact/v1`; all remain `hypothesis`.

The 35 MB lens and 70 MB resumable checkpoint are local ignored artifacts. The
small fit/evaluation manifests and readout records are published in the
repository. Weight publication requires an explicit distribution decision that
accounts for the source corpus's CC-BY-SA-3.0 and GFDL terms.

## Next decision

Run a deterministic capability sweep across operand width, carry/no-carry,
prompt family, and answer format. Freeze the easiest nontrivial regimes where
the base model is reliably correct, then use a separately selected audit lens
and require value-specific discrimination plus causal necessity/sufficiency.
