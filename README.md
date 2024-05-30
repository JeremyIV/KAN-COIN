# KAN COIN

Quick and simple implementation of [COIN](https://github.com/EmilienDupont/coin) using [Komolgorov Arnold Networks (KANs)](https://github.com/KindXiaoming/pykan) instead of [SIRENs](https://github.com/vsitzmann/siren).

I did not manage to match the original SIREN-based compression performance.

![image](https://github.com/JeremyIV/KAN-COIN/assets/72421929/3c1e9b1d-de23-42b0-bddd-d052f841ccd7)

![image](https://github.com/JeremyIV/KAN-COIN/assets/72421929/de5b6367-c6a5-481e-acec-f51d2c697926)


I ran a random hyperparameter search of 10,000 KAN networks to compress images from the Kodak dataset to 0.3 BPP with 9-bit post-training quantization, and I found the optimal hyperparameters to be:

- learning_rate: 0.001
- layer_size: 12
- num_layers: 3
- grid_size: 34
