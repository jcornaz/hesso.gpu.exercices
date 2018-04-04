# GPU programming exercices

## Status
This was a school project. It is no longer maintained and no support will ever be provided.

The project is available to anyone interested by studying, modifing or using it. However, be aware that it might contains bugs or errors, and some resources might be missing (data or credentials).

Please note that the language and technologies might have been new for the contributors when developping the project. Time frame and project requirement are also very different than most *real-world* project. 

**Coding style is not representative of current style of the contributors**

## Mesured speeds

### on CPU

| Version                 | Speed     |
|-------------------------|----------:|
| Image acquisition only  | 94.2 fps  |
| Sequential (naive)      | 0.6 fps   |
| Sequential (indices)    | 3.3 fps   |
| OpenMP (naive)          | 5.5 fps   |
| OpenMP (indices)        | 15.3 fps  |

### on GPU
| Version                             | Speed     |
|-------------------------------------|----------:|
| Image acquisition only              | 241.5 fps  |
| First (naive)                       | 94.4 fps  |
| Indices calcul improved             | 159.9 fps |
| Kernel in Constant memory           | 196.0 fps |
| MinMax with multi-GPU               | 151.2 fps |
| All kernels with multi-GPU          | 137.1 fps |
| One-to-One pattern                  | 141.0 fps |
| One-to-One pattern                  | 183.8 fps |


# Exercices

## OMP
### Not image
* Slicing (Saucisson)

### Image
* Rippling
* Mandelbrot and Julia Fractals
* Newton Fractal
* Heat-Transfert
* Ray-Tracing

## CUDA
### Not image
* Slicing (Saucisson)
* Monte Carlo
  * Single GPU version
  * Multi GPU version

### Image
* Rippling
  * With a one dimensional CUDA grid
  * With a two dimensional CUDA grid
  * With the one-to-one pattern
* Mandelbrot and Julia Fractals
* Mandelbrot multi-GPU (not working)
* Newton Fractal
* Heat-Transfert
* Ray-Tracing
