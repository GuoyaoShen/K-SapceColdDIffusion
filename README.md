# KSpaceColdDiffusion
This is the implementation of k-space cold diffusion model for accelerated MRI reconstruction from:

[Learning to reconstruct accelerated MRI through K-space cold diffusion without noise](https://doi.org/10.1038/s41598-024-72820-2)

## Overview

We present a k-space cold diffusion model for accelerated MRI reconstruction. Different from previous diffusion-based models for MRI reconstruction that utilized Gaussian noise, our model performs degradation in k-space during the diffusion process. A deep neural network is trained to perform the reverse process to recover the original fully sampled image. In such a way, the k-space sampling process is integrated directly into the image degradation process, enhancing the model’s generalizability, especially when the sampling process is similar. This allows for quicker application and better performance in zero-shot or few-shot learning scenarios.

![Overall of our k-space cold diffusion model](/imgs/Figure%201.png)
![K-space cold diffusion degradation process](/imgs/Figure%202.png)