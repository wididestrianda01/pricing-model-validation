# Market data: raw data out of the repo; a frozen snapshot committed later

Market data is large and login-gated; committing it would bloat the repo and complicate reproducibility. Decision: fetch raw data (option chains, rate series) once into a local gitignored location, recording source, as-of date, and checksums in the run manifest; later, commit a small frozen snapshot of the processed inputs for reproducibility. Nothing after the initial fetch touches the network.

Considered Options:
- Commit raw data — reproducible but bloaty (the multi-date SPY dataset is ~600 MB) and pollutes the repo.
- Re-fetch on demand — breaks the offline requirement and is non-deterministic.
