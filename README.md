
# Field of Junctions

This repository contains code for:

**[Field of Junctions](http://vision.seas.harvard.edu/foj/)**
<br>
[Dor Verbin](https://scholar.harvard.edu/dorverbin) and [Todd Zickler](http://www.eecs.harvard.edu/~zickler/)
<br>


Please contact us by email for questions about our paper or code.



## Requirements

Our code is implemented in pytorch. It has been tested using pytorch 1.6 but it should work for other pytorch 1.x versions. The following packages are required:

- python 3.x
- pytorch 1.x
- numpy >= 1.14.0


## Usage

To analyze an `HxWxC` image into its field of junctions, you can simply run the following code snippet:
```
foj = FieldOfJunctions(img, opts)
foj.optimize()
```

In addition to the input image, the `FieldOfJunctions` class requires an object `opts` with the following fields:
```
R                          Patch size
stride                     Stride for junctions (e.g. opts.stride == 1 is a dense field of junctions)
eta                        Width of Heaviside functions
delta                      Width of boundary maps
lr_angles                  Learning rate of angles
lr_x0y0                    Learning rate of vertex positions
lambda_final               Final value of spatial consistency weight lambda
nvals                      Number of values to query in Algorithm 2 from the paper
num_initialization_iters   Number of initialization iterations
num_refinement_iters       Number of refinement iterations
greedy_step_every_iters    Frequency of "greedy" iteration (applying Algorithm 2 with consistency)
parallel_mode              Whether or not to run Algorithm 2 in parallel over all `nvals` values.
```

Instead of using `foj.optimize()` which executes the entire optimization scheme, it is possible to access the field of junctions
during optimization by using the following equivalent code snippet:
```
foj = FieldOfJunctions(img, opts)
for i in range(foj.num_iters):
    foj.step(i)
```

### Boundary maps

In order to compute the (global) boundary maps for a given field of junctions object `foj`:

```
params = torch.cat([foj.angles, foj.x0y0], dim=1)
dists, patches = foj.get_dists_and_patches(params)
local_boundaries = foj.dists2boundaries(dists)
global_boundaries = foj.local2global(local_boundaries)[0, 0, :, :].detach().cpu().numpy()
```

### Boundary-aware smoothing

In order to compute the boundary aware-smoothing of the input image given `foj`, use:
```
params = torch.cat([foj.angles, foj.x0y0], dim=1)
dists, patches = foj.get_dists_and_patches(params)
smoothed_img = foj.local2global(patches)[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
```        


## Data (coming soon)


## Citation

For citing our paper, please use:
```
@article{verbin2020foj,
author = {Verbin, Dor and Zickler, Todd},
title = {Field of Junctions},
journal = {arXiv preprint arXiv:2011.13866},
year = {2020}
}
```
