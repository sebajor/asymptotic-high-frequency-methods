# asymptotic-high-frequency-methods

This repository contains codes to solve large antennas electromagnetic problems.

The main idea is to speed up the computation using GPU capabilities, therefore the codes are in jax, allowing GPU acceleration and autodifferentiation over the main parameters enabling optimization procedures.


For the APEX holography system we parametrized the panels deformations and compute the loss as the difference between the measured beam map and the one created by the propagation of physical optics.

The backpropagation of the errors end up being slow given the huge graphs that the forward function creates. One attempt to diminish it is to create a machine learning model that genrates the mapping between the measured mapand the panel deformation parameters. This can be used as starting point for the optimization procedure of physical optics to speed it up.

Some candidates for the ML model are:
- FFT-> convolutional NN -> multi-perceptron
- FFT-> FNO -> Convolutional NN -> MLP


## TODO
- [x] Physical Optics method
- [x] Fresnel-Kirchhoff method
- [x] Cassegrain geometry
- [ ] Migrate Physical-Optics to JAX
- [x] Create APEX model considering cone-subreflector
- [x] Create py-tree witht the panels geometry and deformations.
- [ ] Test the pure optimization process of pure physical-optics.
- [ ] Create training dataset for ML approach.
- [ ] Create JAX Fourier-neural operator (FNO). This is one of the natural candidates for the ML model.
- [ ] Create JAX convolutional layer.
- [ ] Create JAX U-NET (the idea is to predict the surface error)

