# entropy_regularisation

A Wasserstein gradient flow approach to Fredholm Equations of the First Kind

This repository provides Julia code to reproduce the examples of Chapter 7 of the PhD thesis "Some Interacting Particle Methods with Non-Standard Interactions".

The folder `analytically_tractable` contains code to reproduce the example in Section 7.4.1.

The folder `deconvolution` contains code to reproduce the first and the second example in Section 7.4.2.
 
The folder `1918flu` contains code to reproduce the third example in Section 7.4.2.

The folder `pet` contains code to reproduce the example in Section 7.4.3.

Dependencies: a recent version of R with packages:
* ks
* tictoc
* readxl
* incidental (for 1918 pandemic flu example)
* fDKDE (available [here](https://researchers.ms.unimelb.edu.au/~aurored/links.html#Code); for deconvolution examples)
