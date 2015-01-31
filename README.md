ElVis
=====

ElVis is a visualization system created for the accurate and interactive visualization of scalar fields produced by high-order spectral/hp finite element simulations.


## Known good configurations 

All configurations listed below are for 64-bit systems and software.

OS | Compiler | Cuda | OptiX | Driver | GPU
---|----------|------|-------|--------|----
Windows 7 | Visual Studio 2010 | 6.0 | 3.5.1 | 322.88 | Quadro K4100M
Windows 7 | Visual Studio 2010 | 6.0 | 3.6.0 | 322.88 | Quadro K4100M
Windows 7 | Visual Studio 2010 | 6.0 | 3.6.3 | 322.88 | Quadro K4100M
Open SUSE 12.04 | gcc 4.6.3 | 5.5 | 3.5.1 | 331.20 | GeForce GTX 465
Open SUSE 13.1 | gcc 4.8.1 | 6.0 | 3.6.0 | 340.46 | GeForce GTX 760
Ubuntu 14.04 | gcc 4.8.2 | 6.0 | 3.6.0 | 340.46 | GeForce GTX 560 Ti

## Known bad configurations

OS | Compiler | Cuda | OptiX | Driver | GPU | Reason 
---|----------|------|-------|--------|----|---
Open SUSE 12.04 | gcc 4.6.3 | 5.5 | 3.6.3 | 331.20 | GeForce GTX 465 | 
Ubuntu 14.04 | gcc 4.8.2 | 6.0 | 3.6.3 | 340.58 | GeForce CTX 770 | A supported NVIDIA GPU could not be found.

