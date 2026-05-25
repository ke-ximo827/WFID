# WFID: Wavelet-Fourier Identity-Decoupled Conditioning for Thermal-to-Visible Face Translation via Denoising Diffusion

This is the official implementation of the paper:

> **WFID: Wavelet-Fourier Identity-Decoupled Conditioning for Thermal-to-Visible Face Translation via Denoising Diffusion**
>
> Yunan Hu\*, Changmeng Peng\*, Xinyu Yang, Andong Deng, Lu Yang, Han Qin, Ruijie Xie, and Ye Lin†
>
> *Submitted to IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)*
>
> \* Equal contribution &emsp; † Corresponding author

## Overview

WFID addresses thermal-to-visible (T2V) face translation by decoupling identity semantics, local frequency structures, and global spectral statistics into three dedicated conditioning pathways within a denoising diffusion framework.

## Repository Structure

```
├── core/                   # Core modules
├── guided_diffusion/       # U-Net and conditioning modules
├── scripts/                # Training and testing scripts
├── Finetune_ir_se50.py     # Cross-domain fine-tuning for IR-SE50
├── preprocess_test.py      # Test data preprocessing
├── run_experiments.sh      # Ablation experiment launcher
└── README.md
```

## Datasets

Experiments are conducted on two publicly available T2V face datasets:

- [TFW](https://github.com/Kuzdeuov/TFW) (Kuzdeuov et al., IEEE TIFS 2022)
- [Tufts Face Database](http://tdface.ece.tufts.edu/) (Panetta et al., IEEE TPAMI 2020)



## Acknowledgements

This codebase builds upon [T2V-DDPM](https://github.com/Nithin-GK/T2V-DDPM). We thank the authors for releasing their code.
