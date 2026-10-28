# Method

## Data (both synthetic, 4 features, 4 qubits)
* **Fraud-like.** Features x ~ N(0, I) clipped to [-2.5, 2.5] and mapped to [0, pi]. The fraud score is x0*x1 + 0.8*x2*x3 + 0.3*x0,
  so a linear model cannot separate the classes. The top 15% by a fixed threshold are positives. Train labels get 3% flip noise; test labels are clean.
* **Quantum-feature-map control.** Features uniform on [0, pi]^4; the score is the expectation of a fixed random Pauli-sum observable in the
  state of the kernel's own feature map (bandwidth a = 1). Favourable to the quantum kernel by construction (circular).
* Train sizes n in {32, 64, 128, 256, 512} with exactly 15% positives; test set of 2000 (300 positives). A fresh draw per seed.

## Models
| Model | Role | Tuning |
|---|---|---|
| Fidelity-kernel support vector machine (SVM) ("QKernel") | standard quantum baseline: ZZ-type feature map, 2 repetitions, ring of 4 pairs | bandwidth a in {0.5, 1, 2} and C in {0.1, 1, 10, 100}, 3-fold stratified cross-validation on average precision (AP) |
| Data re-uploading variational quantum circuit (VQC) ("ReUpload-VQC") | standard quantum baseline of Jerbi et al. (2023); 3 layers, 50 parameters, observable = mean of Z | Adam, 150 epochs, class-balanced cross-entropy, 2 random restarts |
| Radial-basis-function (RBF) kernel SVM | classical counterpart of QKernel | gamma in {0.05, 0.15, 0.5, 1.5} and the same C grid, same cross-validation |
| Multi-layer perceptron (MLP), 8 tanh units (49 parameters) | parameter-matched counterpart of the circuit | same optimiser, epochs, loss, restarts as the circuit |
| Logistic regression, random forest | strong simple tabular baselines | logistic regression: C by cross-validation; forest: 200 trees, balanced |

## Metrics
AP (main) and area under the receiver-operating-characteristic curve (ROC AUC, secondary). Learning-curve exponent: slope of log(1 - AP) against log(n), fitted on the seed means. "Best classical" is the
per-seed maximum over the four classical models, which favours the classical side. Mean +- standard deviation over 5 seeds.

## Ablation
Data re-uploading circuit with layers in {1, 2, 3} and entangling ring on/off (n = 128). Without entanglement the model is additive in the
features and cannot represent the interaction terms. For every variant we also report the Jacobian effective rank (the exponential of the Shannon entropy of the
normalised squared singular values of d logit / d parameters over the training set) and the numerical rank, as a parameter-redundancy
diagnostic in the spirit of Alzboun and Abed-alguni (2027). The 8-unit MLP is analysed the same way.

## Noise and robustness sweep
Models are trained noise-free and evaluated (kernel: trained and evaluated) under an effective global depolarising model with
fidelity f = (1 - p)^N, where N is the two-qubit gate count (circuit: 8 controlled-Z (CZ) gates; fidelity-kernel compute-uncompute circuit: 32 controlled-NOT (CNOT) gates after decomposing each
ZZ rotation into 2 CNOT gates). Outcome probabilities are f*P + (1 - f)/16. Finite shots are sampled exactly from this distribution
(kernel: binomial per Gram entry; circuit: multinomial per sample). Sweep: p in {0, 0.1%, 0.5%, 1%, 2%, 5%} x shots in {100, 1000, 10000}, n = 128.
A global depolarising channel only rescales a noise-free expectation, so it affects ranking solely through shot noise and kernel regularisation. This is
optimistic against real devices (no coherent or crosstalk errors) and is stated as a limitation.

## Seeds and compute
Seeds 0-4 control data, splits, initialisations and shot sampling. `torch.set_num_threads(6)`; CPU only.
