# KAN COIN

Quick and simple implementation of [COIN](https://github.com/EmilienDupont/coin) using [Komolgorov Arnold Networks (KANs)](https://github.com/KindXiaoming/pykan) instead of [SIRENs](https://github.com/vsitzmann/siren).

This implementation does not achieve as good performance as the original SIREN implementation.

Ground truth:

![image](https://github.com/JeremyIV/KAN-COIN/assets/72421929/09138155-710f-4567-b996-bddb258c1ae1)


SIREN COIN 30.9 PSNR @ 0.3 BPP (16-bit quantization):

![image](https://github.com/JeremyIV/KAN-COIN/assets/72421929/5141bbcd-92f0-4a28-a7e5-0ce366ae76f8)


KAN COIN 30.1 PSNR @ 0.3 BPP (9-bit quantization):

![image](https://github.com/JeremyIV/KAN-COIN/assets/72421929/c41e538c-0fd4-45d6-ac61-f20f51ce201d)

I ran a random hyperparameter search of 10,000 KAN networks to compress images from the Kodak dataset to 0.3 BPP with 9-bit post-training quantization, and I found the optimal hyperparameters to be:

- learning_rate: 0.001
- layer_size: 12
- num_layers: 3
- grid_size: 30
