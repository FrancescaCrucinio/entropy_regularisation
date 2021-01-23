# entropy_regularisation

A Wasserstein gradient flow approach to Fredholm Equations of the First Kind

This repository provides Julia code to reproduce the examples of 

The folder `gaussian_mixture` contains code to reproduce the example in Section 7.4.2. 

The folder `pet` contains code to reproduce the example in Section 7.4.5.

The folder `analytically_tractable` contains code to reproduce the example in Section 7.4.1.
 
The folder `1918flu` contains code to reproduce the example in Section 7.4.4.
 
The folder `deconvolution` contains code to reproduce the example in Section 7.4.3.
 
Dependencies: a recent version of R with packages:
* ks
* ggplot2
* tictoc
* readxl
* incidental (for 1918 pandemic flu example)
* fDKDE (available [here](https://researchers.ms.unimelb.edu.au/~aurored/links.html#Code); for deconvolution examples)
* scales, viridis (for plotting in the PET example)