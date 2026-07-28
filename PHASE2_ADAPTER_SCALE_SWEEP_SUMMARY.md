# Phase 2 Fixed-Weight Scale-Sweep Summary

## Outcome

**Additional amplitude produced a small causal gain, then saturated while
damaging identity preservation. Amplitude alone is not the missing mechanism.**

The sweep changed no learned weights. Selection chose scales 1.0, 2.0, and 1.5
for the leading, tens, and ones positions. Development exactness increased from
13/90 to 16/90 and tens accuracy from 20/90 to 25/90. Identity preservation
fell from 82/90 to 53/90.

## Provenance

- Frozen protocol commit: `ac2b6b8`.
- Frozen config SHA-256:
  `5721a49db67d013709b2599903aac6b4c08ad2e043728de2a1a45c964294b3fd`.
- Source weights SHA-256:
  `be23dac0cb1d30a1929fc9ee24a839c3c530565f5e7efd3b1505261563c5d7c2`.
- Result SHA-256:
  `0a300ce8e5ed5940547cd9d2c011d68f5fc59defae9375a21f1b8a4d2e649359`.
- New weights trained: none.
- Audit examples evaluated: 0/90.

## Selection

| Position | Selected scale | Target accuracy | Identity accuracy | Target norm |
|---|---:|---:|---:|---:|
| Leading | 1.0 | 88/90 | 88/90 | 34.1% |
| Tens | 2.0 | 31/90 | 69/90 | 48.9% |
| Ones | 1.5 | 54/90 | 74/90 | 48.0% |

The tens curve was non-monotonic:

| Scale | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 | 2.5 | 3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target tens | 16/90 | 18/90 | 18/90 | 25/90 | **31/90** | 28/90 | 23/90 |
| Identity tens | 89/90 | 82/90 | 78/90 | 71/90 | 69/90 | 58/90 | 51/90 |

The decline beyond scale 2.0 shows that the tens result is not simply waiting
for native-donor-sized amplitude.

## Development comparison

| Metric | Scale 1.0 baseline | Selected scales |
|---|---:|---:|
| Digit 1 | 87/90 | 87/90 |
| Digit 2 | 20/90 | 25/90 |
| Digit 3 | 56/90 | 53/90 |
| Exact target | 13/90 | 16/90 |
| Identity preservation | 82/90 | 53/90 |
| Mean norms | 36%, 24%, 37% | 36%, 49%, 56% |
| Parse rate | 100% | 100% |

The strongest exact matched control reached 1/90.

## Frozen gate

| Requirement | Threshold | Observed | Result |
|---|---:|---:|---|
| Exact target | >=50% | 17.8% | Fail |
| Every position | >=70% | 96.7%, 27.8%, 58.9% | Fail |
| Exact advantage over every control | >=25 points | 16.7 points | Fail |
| Identity preservation | >=90% | 58.9% | Fail |
| Relative norm at every position | <=100% | 36.3%, 48.6%, 55.7% | Pass |
| Parse rate | 100% | 100% | Pass |

## Interpretation

The tens adapter contains a real target-specific direction: doubling its scale
adds five development successes and norm-matched controls do not reproduce the
effect. But the direction is incomplete. It peaks far below the native
intervention's tens accuracy, and further scaling reverses the gain.

The preservation collapse also reveals that the same learned map is being
asked to solve two distinct operations:

1. override a computed digit with a requested digit;
2. emit a harmless near-zero update when no override is needed.

Increasing a single scalar cannot improve the first without amplifying errors
in the second.

## Next experiment

Follow the preregistered decision rule with a tens-only representation and
boundary study:

1. establish a balanced synthetic-target native upper bound for the tens
   position using fit-split donor states;
2. sweep a compact set of late residual boundaries on selection only;
3. measure target accuracy and required norm under teacher forcing;
4. at the best boundary, compare present rank 64 against a higher-rank or
   direct optimized residual oracle;
5. train a new amortized writer only after the causal geometry is identified.

The audit remains sealed.
