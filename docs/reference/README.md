# Technical reference: pricing and model validation

This series documents the numerical pricing and independent model-validation
core: the data, the pipeline, the architecture, the decisions behind them and
their rationale, the underlying theory, and the regulation the project is
measured against. It assumes calculus and probability but not stochastic
calculus, and it introduces each piece of mathematics at the point where the
code first needs it.

The series is a technical reference, not a tutorial. It is written in the
impersonal voice of a practitioner, and it states the reasoning behind each
design choice as a conclusion a prudent person in the industry would draw.

## How to use it

Read the chapters in order. Each chapter follows one pattern:

1. **Why it matters.** The role this piece plays in a pricing or validation
   team.
2. **Theory.** The minimum mathematics the code uses, stated before the code.
3. **Implementation.** Where the code lives and what each function does.
4. **Run it.** The command, the real output, and how to read it.
5. **Decisions and rationale.** Why one route was chosen over another.
6. **Key takeaways.** The facts the rest of the project relies on.

Run the code as you read. The pipeline is offline and deterministic:

```bash
uv sync
uv run python scripts/run_all.py      # full run; about 20 s
uv run pytest -q                      # 165 tests, about 2 min
```

The first run in a fresh process compiles the Numba kernels and takes longer
(about 30 s). Later runs reuse the compiled cache.

## Reading order

| Chapter | Covers |
| --- | --- |
| [00-foundations.md](00-foundations.md) | Probability and calculus foundations |
| [01-data.md](01-data.md) | Dataset, data pipeline, reproducibility |
| [02-pricing-theory.md](02-pricing-theory.md) | Options, no-arbitrage, risk-neutral pricing, closed forms |
| [03-monte-carlo.md](03-monte-carlo.md) | Brownian motion, Ito, GBM, discretization, convergence, variance reduction, QMC |
| [04-greeks.md](04-greeks.md) | Sensitivities, four estimation routes, bias-variance |
| [05-pde.md](05-pde.md) | Feynman-Kac, Black-Scholes PDE, schemes, early exercise |
| [06-anchor.md](06-anchor.md) | Hull-White, Bermudan swaption, LSM vs PDE, hedge and PLA |
| [07-calibration.md](07-calibration.md) | Implied vol, SABR, Heston, real-data fitting |
| [08-validation-regulation.md](08-validation-regulation.md) | SR 26-2 framework and regulation in plain English |

## Conventions

- Code paths are relative to the repo root, for example
  `src/pricing/monte_carlo/engine.py`.
- Terminology follows `docs/domain-language.md`. A price is the discounted
  risk-neutral expectation of a payoff. A benchmark is a known-good reference
  price. A challenger is an independent second implementation used to
  corroborate the primary model.
- Every number quoted in this series comes from the run recorded in
  `data/processed/run_manifest.json` and `data/processed/evidence_tables.csv`.
  The run is reproducible; a fresh run produced the same values.
