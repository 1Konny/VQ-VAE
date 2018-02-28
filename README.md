## Neural Discrete Representation Learning, VQ-VAE
Pytorch implementation of [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
<br><br>
## Requirements
* python 3.6
* pytorch 0.2.0_4
* visdom
<br><br>
## RESULT : MNIST
![mnist](sample/mnist.gif)
<br>
## RESULT : CIFAR10
reconstruction of randomly selected, fixed images
<br>
![cifar10_fixed](sample/cifar10_fixed.gif)
<br>
reconstruction of random samples
<br>
![cifar10_random](sample/cifar10_random.gif)
<br>
you can reproduce similar results by :
```javascript
python main.py --dataset CIFAR10 --batch_size 100 --k_dim 256 --z_dim 256
```

## To do:
- [x] model save/load
- [x] parameter initializaiton
- [x] image save
- [ ] visdom -> tensorboardX
- [ ] learning prior p(z) using PixelCNN
- [ ] image sampling( dummy input => (PixelCNN) => Z_dec => (Decoder) => image )
- [ ] refactoring
- [ ] add references and acknowledgements
