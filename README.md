# Work deepening

## Mesured speeds

### on CPU

| Version     | Speed   |
|-------------|--------:|
| Sequential  | 0.6 fps |
| OpenMP      | 5.8 fps |

### on GPU
| Version                             | Speed     |
|-------------------------------------|----------:|
| First (naive)                       | 87.0 fps  |
| Amplified                           | 79.8 fps  |
| Black&White assumed by the kernel   | 98.1 fps  |
| Black&White computed by the kernel  | 87.8 fps  |
| Indices calcul improved             | 138.5 fps |
| 1D cuda grid (instead of 2D)        | 139.4 fps |
| Cuda grid dimension tuned           | 146.2 fps |
| Kernel in Constant memory           | 167.0 fps |


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
