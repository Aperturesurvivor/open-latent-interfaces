# Phase 1A Lab Notebook

## 2026-07-27 — upstream verification

Pinned `anthropics/jacobian-lens` at
`581d398613e5602a5af361e1c34d3a92ea82ba8e` in an isolated environment.
The upstream suite passed 32/32 tests. An OLI adapter smoke used the release's
real `JacobianLens` class and correctly translated HF hidden-state index 2 to
J-lens block 1.

The pinned NLA inference module was imported directly. The adapter contract
matches:

```text
NLAClient.generate(activation, *, prompt=None, extract_explanation=True, **sampling)
NLACritic.reconstruct(explanation)
```

No real NLA checkpoint was run.

## 2026-07-27 — one-prompt J-lens engineering smoke

Model: `Qwen/Qwen2.5-0.5B-Instruct`.

Environment: Apple MPS, float16. Fitted blocks 3, 7, 11, 15, and 19 toward
block 23 using one 40-token generic prompt, `dim_batch=16`, `skip_first=8`.

Result:

```text
seq_len=40
n_valid=31
elapsed=16 seconds
max ||J|| / sqrt(d)=1.626
```

The lens and checkpoint saved successfully.

## 2026-07-27 — frozen corpus selection

Resolved immutable revisions:

```text
Qwen/Qwen2.5-0.5B-Instruct
  7ae557604adf67be50417f59c2c2f167def9a775
Salesforce/wikitext
  b08601e04326c79dfdd32d625aee71d232d685c3
```

Selected 24 rows from 1,134 eligible WikiText-2 raw training passages using
seed `20260727` and a 40–64 token filter. Stored row indices, token counts, and
text SHA-256 values but not source text. The config was committed and pushed as
`1eadf9b` before fitting.

Dataset license metadata at the pinned revision lists CC-BY-SA-3.0 and GFDL.
The target model and J-lens code are Apache-2.0.

## 2026-07-27 — 24-prompt development fit

Command:

```bash
uv run --project <pinned-jlens-clone> scripts/fit_jacobian_lens.py fit \
  --config configs/phase1_jlens_development.json \
  --output artifacts/jlens/phase1_development_lens.pt \
  --checkpoint checkpoints/phase1_jlens_development.pt \
  --report results/phase1_jlens_development_fit.json \
  --device mps --dtype float16 --checkpoint-every 4 --no-resume
```

Fit:

```text
blocks: 0–22
target: block 23
d_model: 896
prompts: 24
elapsed: 511.98 seconds
lens size: 35 MB
checkpoint size: 70 MB
lens SHA-256: 529bbc8b261d945bd4bb94785077b9f6bb5e545f20efd4fa236755c74a858c97
```

Two precommitted passages produced comparatively high per-prompt Jacobian
norms: 3.795 and 4.415 after division by `sqrt(d)`. They were retained. The
maximum relative shift in the running mean was 0.0752 on the final passage.
This is a development convergence diagnostic, not a frozen stability test.

Every final matrix is finite. Frobenius norm divided by `sqrt(d)` ranges from
0.770 to 1.280. Cosine with the identity map rises from 0.135 at block 0 to
0.889 at block 22, a sensible approach toward the final residual basis.

## 2026-07-28 — development math readout

Environment:

```text
Python 3.12.12
PyTorch 2.12.0
Transformers 5.9.0
MPS float16
```

Evaluated all 72 Phase 0.1 development additions. These use the held-out
`word_problem` template, have eight examples for each leading result digit
1–9, and produce three tokenizer tokens per result.

For each prompt's last token and every internal block:

1. read the correct first-result token's rank under J-lens;
2. read the same rank under vanilla logit lens;
3. rank the correct digit among the nine digit tokens;
4. compute its logit margin over the strongest wrong digit.

The initial vocabulary-rank result looked encouraging: J-lens block 19 had
median rank 5 and hit@10 80.6%, compared with the output-adjacent vanilla
logit-lens block 22 at median rank 7 and hit@10 75.0%.

The balanced digit test falsified the stronger interpretation. J-lens block 19
was correct on 12/72, and all earlier blocks 0–15 were exactly at 8/72 except
for no improvement. All mean digit margins were negative. The actual final
model was correct on 7/72.

Twenty sample readouts spanning four examples and five blocks were written
through the common artifact schema. They validate on reload, have unique
artifact IDs, and retain `hypothesis` status.

## Interpretation

The lens is functioning as a global-workspace-style disposition readout: it
strongly surfaces numeric answer vocabulary. The present task/model/site does
not support a claim that it recovered the computed result. Because the base
model also fails the behavioral task, the next experiment must establish the
competence envelope before more invasive channel mapping.
