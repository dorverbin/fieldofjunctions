<img src="https://user-images.githubusercontent.com/15837806/101306054-f28f2900-3811-11eb-99eb-cf6bc0d56b9c.png"
     alt="Input image"
     style="float: left; margin-right: 10px;" />
<img src="https://user-images.githubusercontent.com/15837806/101306056-f327bf80-3811-11eb-9a38-1fd7d0bc7bae.gif"
     alt="Optimization"
     style="float: left; margin-right: 10px;" />

# Field of Junctions

### [Project Page](http://vision.seas.harvard.edu/foj/) | [Paper](https://arxiv.org/abs/2011.13866) | [Video](https://youtu.be/M0VwBw_aVQA)

This repository contains code for:

**[Field of Junctions: Extracting Boundary Structure at Low SNR](http://vision.seas.harvard.edu/foj/)**
<br>
[Dor Verbin](https://scholar.harvard.edu/dorverbin) and [Todd Zickler](http://www.eecs.harvard.edu/~zickler/)
<br>
International Conference on Computer Vision (ICCV), 2021.


Please contact us by email for questions about our paper or code.



## Requirements

Our code is implemented in pytorch. It has been tested using pytorch 1.6 but it should work for other pytorch 1.x versions. The following packages are required:

- python 3.x
- pytorch 1.x
- numpy >= 1.14.0


## Usage

To analyze an `HxWxC` image into its field of junctions, you can simply run the following code snippet:
```
from field_of_junctions import FieldOfJunctions
foj = FieldOfJunctions(img, opts)
foj.optimize()
```

In addition to the input image, the `FieldOfJunctions` class requires an object `opts` with the following fields:
```
R                          Patch size
stride                     Stride of field of junctions (e.g. opts.stride == 1 is dense)
eta                        Width of Heaviside functions
delta                      Width of boundary maps
lr_angles                  Learning rate of angles
lr_x0y0                    Learning rate of vertex positions
lambda_boundary_final      Final value of spatial boundary consistency weight lambda_B
lambda_color_final         Final value of spatial color consistency weight lambda_C
nvals                      Number of values to query in Algorithm 2 from the paper
num_initialization_iters   Number of initialization iterations
num_refinement_iters       Number of refinement iterations
greedy_step_every_iters    Frequency of "greedy" iteration (applying Algorithm 2 with consistency)
parallel_mode              Whether or not to run Algorithm 2 in parallel over all `nvals` values.
```

Note that setting `parallel_mode` to `True` typically results in faster optimization, but requires more memory during
initialization. For large images on a GPU with limited memory, you might need to set `parallel_mode` to `False`.


Instead of using `foj.optimize()` which executes the entire optimization scheme, it is possible to access the field of junctions
during optimization by using the following equivalent code snippet:
```
foj = FieldOfJunctions(img, opts)
for i in range(foj.num_iters):
    foj.step(i)
```

See Python notebook in the `examples/` folder for a full usage example.

### Boundary maps

In order to compute the (global) boundary maps for a given field of junctions object `foj`:

```
params = torch.cat([foj.angles, foj.x0y0], dim=1)
dists, _, patches = foj.get_dists_and_patches(params)
local_boundaries = foj.dists2boundaries(dists)
global_boundaries = foj.local2global(local_boundaries)[0, 0, :, :].detach().cpu().numpy()
```

### Boundary-aware smoothing

In order to compute the boundary-aware smoothing of the input image given `foj`, use:
```
params = torch.cat([foj.angles, foj.x0y0], dim=1)
dists, _, patches = foj.get_dists_and_patches(params)
smoothed_img = foj.local2global(patches)[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
```        


## Data

A zip file containing all of our synthetic data is available [here](https://vision.seas.harvard.edu/foj/dataset/foj_data.zip). It contains the 300 images we used for quantitatively evaluating our algorithm, as well as ground truth locations of edges and corners/junctions.


## Citation

For citing our paper, please use:
```
@InProceedings{verbin2021foj,
author = {Verbin, Dor and Zickler, Todd},
title = {Field of Junctions: Extracting Boundary Structure at Low {SNR}},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2021},
pages = {6869-6878}
}
```
