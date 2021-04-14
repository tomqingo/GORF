# GORF

Implementation of the paper *"Towards Unbiased Random Features with Lower Variance For Stationary Indefinite Kernels"* published in *IJCNN2021* 
([https://arxiv.org/abs/2104.06204](https://arxiv.org/abs/2104.06204))

- Variation Reduction

The numerical simulation for the variation reduction effect of orthogonality on two typical stationary indefinite kernels, *polynomial kernel on the sphere* and *Delta-Gaussian kernel* is implemented in *variationReduct.m* and *variationReductGaussian.m*, included in folder *variation*.

- Approximation error and classification ability

GORF method and other baseline methods (RM, TS, SRF, DIGMM) is implemented in the functions *gen_{}*, in the craket is the name of the corresponding methods.

We conduct experiments on three benchmark classification datasets, *letter*, *ijcnn1* and *usps*, and two stationary indefinite kernels, *polynomial kernel on the sphere* and *Delta-Gaussian kernel*. We utilize *liblinear*, a toolkit for linear regression, combined with the extracted random features using the above algorithm.

- Regression error

We conduct the SVR regression research on *housing* dataset, and the evaluation metrit is *RMSE*.
