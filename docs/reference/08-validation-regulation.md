# 08. Validation and regulation in plain English

This chapter explains what model validation is, the framework the project
follows, and the regulation that framework implements. It closes the loop by
showing how the one-command run produces the report that proves all of it.

## What model validation is

A pricing model is software plus assumptions plus math. Before a bank lets a
model drive real P&L or capital, an independent team must establish that the
model is conceptually sound, that its outputs are correct against references,
that its implementation matches its specification, and that its limits are
known. That process is model validation.

The word "independent" is the whole point. The team that builds the model does
not validate it. An independent validator re-derives, re-prices, and
challenges, precisely so that a shared mistake cannot pass unnoticed.

> **Key takeaways.** Validation is independent re-examination, not the
> builder's self-assessment. Every choice in this project (independent
> challengers, cross-engine agreement) exists to make the independence
> credible.

## The framework: SR 26-2

The project's validation report follows **SR 26-2**, the Federal Reserve's
model risk management guidance that supersedes the older **SR 11-7**. Both are
statements of the same discipline, refined over time: model risk must be
governed, models must be validated independently, and validation must be
documented. SR 26-2 updates the older guidance with explicit expectations for
model risk management across the model lifecycle.

The report also anchors to the European and UK counterparts where they add
specific bars:

- **CRR3 / FRTB** (Regulation 2024/1623) sets the P&L-attribution test the
  hedge leg passes.
- **PRA SS1/23** is the Bank of England's model risk management principles.
- **ECB Guide to Internal Models** governs internal-model approval in the
  euro area.

> **Key takeaways.** SR 26-2 is the modern statement of the discipline that
> SR 11-7 began. The project cites the specific regulation that supplies its
> numeric bars, rather than naming frameworks for decoration.

## The four pillars of the report

The SR 26-2 structure has four parts, and the report fills each:

**Conceptual soundness.** Is the model theoretically defensible? The report
shows each method matches theory: convergence orders, standard-error scaling,
and scheme stability all land where the math says they should.

**Outcomes analysis.** Do the outputs pass the bars? The evidence table pairs
every metric with an acceptance bar and a pass/fail verdict. In the committed
run, 30 hard bars apply and all 30 pass; the remaining 26 metrics are recorded
as information because they are degradation or descriptive results, not
claims.

**Independence.** Was each method checked against something that shares
nothing with it? The report lists every challenger pair: closed forms against
QuantLib, Monte Carlo against closed forms, the LSM anchor against the PDE
anchor, the SABR and Heston fitters against synthetic truth.

**Documentation.** Can a reviewer reproduce everything? Derivations and
assumptions live beside each module; data provenance and contracts live in
`docs/data-sources.md` and `data/manifest.json`; operating procedure lives in
`docs/runbook.md`. Every number regenerates from one offline command.

> **Key takeaways.** The four pillars are a checklist: sound, correct,
> independent, reproducible. A report that fails any one fails validation, no
> matter how good the numbers look.

## The scorecard and findings ledger

Each component gets a verdict: approve, approve with limitations, or reject.
The project's eight components all reach approve or approve-with-limitations;
none reaches red.

A finding is a recorded gap with a severity, an owner, evidence, and a
remediation. The project's findings ledger carries six entries (F-01 through
F-06), from the LSM low bias to the Heston Feller flag. The presence of
findings is a feature: a validation report with no limitations is a report
that was not looked at hard enough.

> **Key takeaways.** Approve-with-limitations is the honest middle verdict,
> and a findings ledger is how limitations are made actionable instead of
> vague. A clean report with no findings is suspicious.

## FRTB P&L attribution, decoded

The FRTB (Fundamental Review of the Trading Book) demands that a desk's risk
model actually explain its P&L. The test compares two P&L series:

- **Risk-theoretical P&L (RTPL)**: the P&L predicted by the model's Greeks
  over a set of market moves.
- **Hypothetical P&L (HPL)**: the P&L from fully re-pricing the portfolio over
  the same moves.

If the model's Greeks are right, RTPL and HPL move together. Two statistics
measure the agreement: **Spearman rank correlation** (do they order the same
way) and **Kolmogorov-Smirnov distance** (are the distributions the same
shape). The bars are Spearman above `0.80` and KS below `0.09`.

The project's hedge leg measures Spearman `1.000` and KS `0.075`, passing both
bars, with the KS margin recorded as thin because first-order RTPL omits
gamma.

> **Key takeaways.** The PLA test is "does my model explain my P&L." Spearman
> tests the ranking, KS tests the distribution. Both must pass, and passing
> narrowly is a finding, not a pass with a clean conscience.

## How the one command produces the report

`scripts/run_all.py` calls `run_all` in `src/pricing/validation/pipeline.py`,
which runs nine experiments and writes two artifacts:

- `data/processed/run_manifest.json`: environment versions, input checksums,
  per-experiment runtimes, and metrics.
- `data/processed/evidence_tables.csv`: one row per metric with component,
  measured value, bar, and verdict.

`evidence_table` in `src/pricing/validation/evidence.py` applies the bars
defined in its `BARS` table. The script exits non-zero if any bar breaches, so
a failing validation cannot look successful. Two consecutive runs produce
identical metrics, enforced by a test.

> **Key takeaways.** The report is not a document someone typed; it is the
> printed output of a deterministic run. The script fails loudly on a breach,
> which is the difference between "the report says pass" and "the pipeline
> proves pass."

## Regulation to metric: the map

| Requirement | Where it is met |
| --- | --- |
| Independent challenge (SR 26-2) | challenger table in the report, section 3 |
| Outcomes vs bars (SR 26-2) | `evidence_tables.csv`, 30 hard bars |
| P&L attribution (FRTB PLA) | `pla_spearman`, `pla_ks_statistic` |
| Reproducibility | `run_manifest.json` checksums, deterministic run |
| Model risk principles (PRA SS1/23, ECB Guide) | scorecard, findings ledger, ownership |

> **Key takeaways.** Every regulatory anchor maps to a concrete, reproducible
> number in the evidence table. This is what separates citing a regulation
> from implementing it.
