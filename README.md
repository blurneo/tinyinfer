# Tinyinfer

## This is a tiny inference framework mainly for x86, CUDA and ARM platforms.

## Features

- Optimized gemm for convolution using im2col method(faster than numpy/openblas).
    - Reach 80% of the peak computation force on intel i7.
- Implementation of 20+ deep learning networkd layers.
    - Supports Mobilenet/Resnet and all kinds of common neural networks
- Model serialization and deserialization using C++ reflection.
    - Efficient model parse and generation in .ti format
- Optimized optical flow implementation.
- JIT(Just-In-Time) compilation for the performance primitives(coming soon)
