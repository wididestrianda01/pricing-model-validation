# Anchor instrument: Bermudan swaption on Hull-White

This project demonstrates many numerical methods, but a model-validation reviewer reads one complete controlled workflow, not a method sampler. Decision: anchor the whole effort on a single Bermudan swaption on a one-factor Hull-White model, priced by Longstaff-Schwartz Monte Carlo and by PDE (ADI), benchmarked against Black-76/Jamshidian/QuantLib, and carried through price → hedge → validation memo. It builds on a Hull-White implementation and is a canonical rates object.

Considered Options:
- Equity autocallable — ties the real-data SABR/Heston calibration into the anchor, but is path-dependent and scope-heavy.
- American put + barrier + Asian basket — pure numerical showcase, less authentic as a traded object.

Consequences:
- The swaption uses a synthetic volatility surface (swaption vol is paid); the real-data proof lives in the separate equity calibration module.
