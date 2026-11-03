# Roadmap

## Question

1. Which architectures and losses best preserve the physical structure of BubbleML fields
   (SDF, temperature, velocity) at a given compression ratio and compute budget?
2. **Does reconstruction quality predict how meaningful the latent is?** Rozet et al. (2025) found
   that emulation stays accurate at high compression even as reconstruction degrades. We test this
   on boiling.

Headline figure: one point per (model, compression ratio), with reconstruction error against a
representation score, and the Spearman rank correlation between the two.

## Phases

### 1. Check-in (Nov 12): baselines and harness
- [x] Framework: data, models, losses, metrics, Lightning CLI, wandb, Slurm scripts
- [x] Test drive on HPC3 (`scripts/conv_ae_smoke.slurm`), reading HDF5 directly: works, but I/O-bound
- [ ] Build the 128×128 float32 cache from the WebDataset shards (`scripts/build_cache.slurm`); rerun the test drive on it
- [ ] Baselines at 2–3 compression ratios: Downsample, POD, ConvAE, the earlier VAE
- [ ] Reconstruction metrics: VRMSE per field; spectral error by wavenumber band; vapor IoU and fraction
- [ ] Latent dump (`predict`) for every model, plus ridge probes with POD and random-encoder controls
- [ ] Slides: baselines, the first reconstruction-vs-probe scatter, the revised plan

### 2. Final (Dec 8 / Dec 11): ablations and representation
- Ablations on FC-72: compression ratio, β, loss. 3 seeds for the final pick.
- LN2 and R515B for generalization; cross-fluid transfer (fit the probe on FC-72, test on LN2).
- Representation metrics, cheapest first:
  - **Readout (trivial probes):** ridge, then a 1-hidden-layer MLP, predicting fluid, wall temperature,
    vapor fraction, heater heat flux / Nusselt number, interface length and bubble count.
    Controls: POD at the same latent size, a random-init encoder, shuffled labels. For spatial
    latents, add a 1×1-conv probe that predicts the vapor mask.
  - **Geometry (no labels):** TwoNN intrinsic dimension, RankMe, VAE active units, kNN
    trustworthiness, CKA between models. Temporal interpolation: decode (z_t + z_{t+2})/2 and
    compare it with x_{t+1}, against pixel-space interpolation.
  - **Usefulness:** the same small latent stepper on each frozen latent (linear least squares,
    then MLP + pushforward); score rollout error, stable rollout length and heat-flux statistics.
- Cost: training GPU-hours, peak memory, encode/decode latency (a column in the tables, not its own analysis).

### 3. Stretch
MDL probes, decoder-Jacobian (pullback) geodesics, left-right flip equivariance of the latent,
sparse/disentanglement analysis, temporal models.

## Protocol (lecture 10 checklist)
- Split by run (`split_runs`), and label results as interpolation (held-out interior wall
  temperatures) or extrapolation.
- Every model gets the same data, the same tuning effort and the same GPU budget. 3+ seeds; report
  the mean and std. Use the test set once.
- Report negative results and failure cases.

## References

✓ checked against arXiv or the publisher during planning · [std] standard reference, not re-checked
· [verify] 2026 preprint, check before citing.

### Data and course
- Hassan et al., *BubbleML: A Multiphase Multiphysics Dataset and Benchmarks for Machine Learning*, NeurIPS 2023 D&B. arXiv:2307.14623 ✓
- *Bubbleformer: Forecasting Boiling with Transformers*, arXiv:2507.21244 (2025). Releases BubbleML 2.0 (FC-72, R515B, LN2) ✓
- Ohana et al., *The Well*, NeurIPS 2024. VRMSE and metrics by frequency band [std]
- McGreivy & Hakim, *Weak baselines and reporting biases…*, Nature Machine Intelligence 2024 [std]

### Latent models for PDEs and fluids
- ★ Rozet, Ohana, McCabe, Louppe, Lanusse, Ho, *Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics Emulation*, NeurIPS 2025. arXiv:2507.02608 ✓
- ★ Solera-Rico et al. (Vinuesa), *β-Variational autoencoders and transformers for reduced-order modelling of fluid flows*, Nature Communications 15 (2024). arXiv:2304.03571 ✓
- Bousquet, Nore, Lucor, *AutoEncoders latent space interpretability in the light of POD*, Computer Physics Communications 315 (2025) 109728 ✓
- Fukami & Taira, *Grasping extreme aerodynamics on a low-dimensional manifold*, Nature Communications 14 (2023). arXiv:2305.08024 ✓
- Na, Seo, Kim, Jeon, Jo, *Decrypting the temperature field in flow boiling with latent diffusion models*, arXiv:2501.16510 (2025). VQ-VAE on BubbleML ✓
- Wu, Maruyama, Leskovec, *Learning to Accelerate PDEs via Latent Global Evolution* (LE-PDE), NeurIPS 2022. arXiv:2206.07681 ✓
- Serrano et al., *AROMA*, NeurIPS 2024. arXiv:2406.02176 ✓
- Hagnberger, Musekamp, Niepert, *CALM-PDE*, NeurIPS 2025. arXiv:2505.12944 ✓
- McCabe, Mukhopadhyay et al., *Walrus: A Cross-Domain Foundation Model for Continuum Dynamics*, arXiv:2511.15684 ✓
- Racca, Doan, Magri, *Predicting turbulent dynamics with the convolutional autoencoder echo state network*, JFM 975 A2 (2023) ✓
- Linot & Graham, *Deep learning to discover and predict dynamics on an inertial manifold*, PRE 101:062209 (2020) ✓
- Zeng, Pérez De Jesús, Fox, Graham, *Autoencoders for discovering manifold dimension and coordinates*, arXiv:2305.01090 ✓
- Lusch, Kutz, Brunton, *Deep learning for universal linear embeddings of nonlinear dynamics*, Nature Communications 9 (2018) ✓
- Champion, Lusch, Kutz, Brunton, *Data-driven discovery of coordinates and governing equations*, PNAS 116 (2019) ✓
- Baldi & Hornik, linear autoencoders span the POD subspace, Neural Networks 2 (1989) [std]
- Giral et al., *AeroJEPA*, arXiv:2605.05586 (2026). Linear probes of physical quantities [verify]
- Rosenfeld & Sonnewald, *Sparse probes and murky physics*, arXiv:2606.11657 (2026). A caution about probing [verify]

### Representation metrics
- ★ Alain & Bengio, *Understanding intermediate layers using linear classifier probes*, arXiv:1610.01644 [std]
- Hewitt & Liang, *Designing and Interpreting Probes with Control Tasks*, EMNLP 2019. arXiv:1909.03368 [std]
- Voita & Titov, *Information-Theoretic Probing with Minimum Description Length*, EMNLP 2020. arXiv:2003.12298 ✓
- ★ Ansuini, Laio, Macke, Zoccolan, *Intrinsic dimension of data representations in deep neural networks*, NeurIPS 2019. arXiv:1905.12784 ✓
- Facco, d'Errico, Rodriguez, Laio, TwoNN intrinsic dimension, Scientific Reports 7 (2017) [std]
- Garrido et al., *RankMe*, ICML 2023. arXiv:2210.02885 ✓
- Agrawal et al., *α-ReQ*, NeurIPS 2022 ✓
- Kornblith et al., *Similarity of Neural Network Representations Revisited* (CKA), ICML 2019. arXiv:1905.00414 [std]
- Raghu et al., *SVCCA*, NeurIPS 2017. arXiv:1706.05806 [std]
- Eastwood & Williams, *DCI disentanglement framework*, ICLR 2018 ✓
- Chen et al., *Isolating Sources of Disentanglement in VAEs* (MIG), NeurIPS 2018. arXiv:1802.04942 [std]
- Locatello et al., *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*, ICML 2019 ✓
- Venna & Kaski, trustworthiness and continuity, ICANN 2001; Lee & Verleysen, co-ranking matrix, Neurocomputing 2009 [std]
- Arvanitidis, Hansen, Hauberg, *Latent Space Oddity*, ICLR 2018. arXiv:1710.11379 ✓
- Karras et al., StyleGAN perceptual path length, CVPR 2019. arXiv:1812.04948 [std]
- Burda, Grosse, Salakhutdinov, IWAE (active units), ICLR 2016. arXiv:1509.00519 [std]

### Rollout and training (for the latent-stepper stage, course lecture 7)
- Brandstetter, Worrall, Welling, *Message Passing Neural PDE Solvers* (pushforward, temporal bundling), ICLR 2022 [std]
- Lippe et al., *PDE-Refiner*, NeurIPS 2023 [std]
- McCabe et al., *Towards Stability of Autoregressive Neural Operators*, TMLR 2023 [std]
- Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph Networks* (noise injection), ICML 2020 [std]
