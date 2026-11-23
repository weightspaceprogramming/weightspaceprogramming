# Implementation and research sources

The channel-norm computation and NPZ loader are adapted from `weight_reading.py` in [weightspaceprogramming/examples](https://github.com/weightspaceprogramming/examples/blob/41e3a63c80c8bfbdda2c3f1c0c47c0d6f33b6f4b/weight_reading.py). That implementation was written for the field guide and tested on four Sparks policy networks. The package generalizes its hidden-unit permutation operation to dense networks.

The seven packaged lessons use synthetic arrays. Download the four Sparks policy networks through the [reproduction instructions](https://weightspaceprogramming.tech/build/code-and-implementations/#weight-reading).

- **Sparks of Weight-Space Programming**, Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab. [Original blog](https://rikiyat.github.io/blog/weight-space-programming/) and [released experiment logs](https://github.com/RikiyaT/lm_understand-logs). The separate four-network experiment retains their checkpoint attribution and CC BY 4.0 terms.
- **Program-as-Weights: A Programming Paradigm for Fuzzy Functions**, Wentao Zhang, Liliana Hotsko, Woojeong Kim, Pengyu Nie, Stuart Shieber and Yuntian Deng. [Paper](https://arxiv.org/abs/2607.02512).
- **LoRA: Low-Rank Adaptation of Large Language Models**, Edward J. Hu and colleagues. [Paper](https://arxiv.org/abs/2106.09685). This package writes the low-rank multiplication in the row-vector convention `X @ (W + scale * A @ B)`.
- **Tracr: Compiled Transformers as a Laboratory for Interpretability**, David Lindner and colleagues. [Paper](https://arxiv.org/abs/2301.05062).
