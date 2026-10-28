# Research gap: do small quantum classifiers help when labels are scarce and the positive class is rare?

## State of the art (from the supplied literature)

Quantum machine learning (QML) surveys agree on the broad picture. Peral-García et al. (2024, doi:10.1016/j.cosrev.2024.100619)
and Zeguendry et al. (2023, doi:10.3390/e25020287) report that most studies re-implement a classical classifier
(support vector machine (SVM), neural network) on a variational circuit and compare it on small public data sets; hardware is the
stated bottleneck. Jerbi et al. (2023, doi:10.1038/s41467-023-36159-y) unify the standard models as *linear models in a
Hilbert space*: fidelity-kernel methods and data re-uploading circuits. In the worst case, the re-uploading circuit needs fewer qubits and less data
than the kernel. Abbas et al. (2024, doi:10.1038/s42254-024-00770-9) call for systematic benchmarking against strong
classical solvers. Two 2026 papers sharpen the question. Bhullar et al. (2027, doi:10.1016/j.compag.2026.112376) conclude that QML "at best matches"
strong classical baselines and that the only plausible near-term niche is *severely label-limited learning*.
Alzboun and Abed-alguni (2027, doi:10.1016/j.ins.2026.124009) show that variational quantum policies fall short of classical agents because of
geometric ill-conditioning and parameter redundancy, even when entanglement is used, and that the gap persists against
parameter-matched classical models.

## Related work table

The brief carries no role field, so we assigned roles from the Scopus metadata: *recent* = published 2026 or later,
*landmark* = more than 200 citations, *review* = survey or systematic review, *adjacent* = neighbouring methods or problem.

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Jerbi et al., doi:10.1038/s41467-023-36159-y | 2023 | landmark | Unifies kernel, linear and data re-uploading quantum models; proves resource separations | No empirical comparison under label scarcity, class imbalance or shot noise |
| Zeguendry et al., doi:10.3390/e25020287 | 2023 | landmark, review | Reviews QML; runs the quantum support vector machine (QSVM), variational quantum classifier (VQC) and quanvolution on Iris, breast cancer, digits | Balanced, mostly easy data; no imbalance, no noise, no seed statistics |
| Peral-García et al., doi:10.1016/j.cosrev.2024.100619 | 2024 | landmark, review | Systematic review of 94 QML studies | Does not test whether reported gains survive equal tuning of the classical model |
| Blekos et al., doi:10.1016/j.physrep.2024.03.002 | 2024 | landmark, review | Reviews the Quantum Approximate Optimization Algorithm (QAOA) and variants, noise resilience | Optimisation, not supervised learning |
| Abbas et al., doi:10.1038/s42254-024-00770-9 | 2024 | landmark, review | Argues for systematic benchmarking in quantum optimisation | Gives principles, not a learning benchmark |
| Ullah and Garcia-Zapirain, doi:10.1109/ACCESS.2024.3353461 | 2024 | landmark, review | Systematic review of QML in healthcare (49 studies) | Does not report a label-budget analysis |
| Bhullar et al., doi:10.1016/j.compag.2026.112376 | 2026 | recent, review | Reviews QML for remote sensing; says label-limited learning is the plausible niche | A hypothesis from the literature, not a measurement |
| Alzboun and Abed-alguni, doi:10.1016/j.ins.2026.124009 | 2026 | recent | Explainability of quantum reinforcement learning; ill-conditioning and redundancy | Reinforcement learning only; no supervised or imbalanced setting |
| Huynh et al., doi:10.1016/j.cosrev.2026.101072 | 2026 | recent, review | Survey of quantum-inspired machine learning; open questions on data embedding | No quantitative learning-curve comparison |
| Férre et al., doi:10.1007/978-3-032-39882-6_45 | 2026 | recent | Open-quantum-system regressor with data re-uploading | Regression on smooth benchmark functions, no classical tuning parity |
| Fan et al., doi:10.1016/j.ins.2026.124264 | 2026 | recent | Hardware-efficient variational circuits for state preparation | State preparation, not classification |
| Mulashani et al., doi:10.1016/j.uncres.2026.100539 | 2026 | adjacent | Simulated quantum neural network for permeability regression, beats classical models | Classical baselines tuned by grid search only; one data set, no seed variance |
| Caleffi et al., doi:10.1016/j.comnet.2024.110672 | 2024 | adjacent, review | Survey of distributed quantum computing | Not about learning |

## The gap

Reviews name label-limited learning as the most plausible near-term niche (Bhullar et al., 2027), and a 2026 diagnostic study
attributes the quantum shortfall to ill-conditioned, redundant parameterisations (Alzboun and Abed-alguni, 2027). Nothing in the supplied
set tests the first claim for the standard quantum baselines of Jerbi et al. (2023) on **imbalanced, fraud-like tabular data**, with
learning curves, equal hyper-parameter tuning, a parameter-matched classical model and a shot-noise budget, or links the outcome to the
parameter-redundancy diagnostic.

**Scale statement.** All experiments are noiseless or effective-noise *simulations* of 4-qubit circuits on a laptop and use a synthetic
fraud-like data generator. Nothing here is a hardware result or a claim about real transaction data.

## Hypotheses and success metrics (fixed before the runs)

Metric: area under the precision-recall curve (average precision, AP) on a clean 2000-sample test set with 15% positives (the no-skill value is 0.15). Seeds: 5.

* **H1 (label scarcity helps quantum models).** On the fraud-like data, at the two smallest label budgets (n = 32, 64), the better of
  {fidelity-kernel SVM, data re-uploading circuit} has a higher mean AP than the better of four tuned classical models, and wins in at least 4 of 5 seeds.
  H1 is **falsified** if the mean paired difference is <= 0 or the quantum model wins in fewer than 4 seeds.
* **H2 (positive control).** On a data set whose labels come from the quantum feature map itself, the quantum kernel beats the radial-basis-function (RBF) kernel support vector machine
  at small n. This checks that the pipeline *can* register a quantum benefit; it is circular by construction and is not evidence of advantage.
* **H3 (noise).** Under two-qubit-gate error and finite shots, the AP of both quantum models drops below that of the classical models before p = 1%.

**Novelty in one sentence:** a label-budget, class-imbalance and shot-noise learning-curve benchmark of the standard quantum classifiers
against tuned and parameter-matched classical models, paired with a Jacobian effective-rank diagnostic of parameter redundancy.
