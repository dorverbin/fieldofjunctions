import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')

class FieldOfJunctions:
    def __init__(self, img, opts):
        """
        Inputs
        ------
        img    Input image: a numpy array of shape [H, W, C]
        opts   Object with the following attributes:
               R                          Patch size
               stride                     Stride for junctions (e.g. opts.stride == 1 is a dense field of junctions)
               eta                        Width parameter for Heaviside functions
               delta                      Width parameter for boundary maps
               lr_angles                  Angle learning rate
               lr_x0y0                    Vertex position learning rate
               lambda_boundary_final      Final value of spatial boundary consistency term
               lambda_color_final         Final value of spatial color consistency term
               nvals                      Number of values to query in Algorithm 2 from the paper
               num_initialization_iters   Number of initialization iterations
               num_refinement_iters       Number of refinement iterations
               greedy_step_every_iters    Frequency of "greedy" iteration (applying Algorithm 2 with consistency)
               parallel_mode              Whether or not to run Algorithm 2 in parallel over all `nvals` values.
        """

        # Get image dimensions
        self.H, self.W, self.C = img.shape

        # Make sure number of patches in both dimensions is an integer
        assert (self.H - opts.R) % opts.stride == 0 and (self.W - opts.R) % opts.stride == 0, \
                "Number of patches must be an integer."

        # Number of patches (throughout the documentation H_patches and W_patches are denoted by H' and W' resp.)
        self.H_patches = (self.H - opts.R) // opts.stride + 1
        self.W_patches = (self.W - opts.R) // opts.stride + 1

        # Store total number of iterations (initialization + refinement)
        self.num_iters = opts.num_initialization_iters + opts.num_refinement_iters

        # Split image into overlapping patches, creating a tensor of shape [N, C, R, R, H', W']
        t_img = torch.tensor(img, device=dev).permute(2, 0, 1).unsqueeze(0)   # input image, shape [1, C, H, W]
        self.img_patches = nn.Unfold(opts.R, stride=opts.stride)(t_img).view(1, self.C, opts.R, opts.R,
                                                                             self.H_patches, self.W_patches)

        # Create pytorch variables for angles and vertex position for each patch
        self.angles = torch.zeros(1, 3, self.H_patches, self.W_patches, dtype=torch.float32, device=dev)
        self.x0y0   = torch.zeros(1, 2, self.H_patches, self.W_patches, dtype=torch.float32, device=dev)

        # Compute gradients for angles and vertex positions
        self.angles.requires_grad = True
        self.x0y0.requires_grad   = True

        # Compute number of patches containing each pixel: has shape [H, W]
        self.num_patches = torch.nn.Fold(output_size=[self.H, self.W],
                                         kernel_size=opts.R,
                                         stride=opts.stride)(torch.ones(1, opts.R**2,
                                                                        self.H_patches * self.W_patches,
                                                                        device=dev)).view(self.H, self.W)

        # Create local grid within each patch
        y, x = torch.meshgrid([torch.linspace(-1.0, 1.0, opts.R, device=dev),
                               torch.linspace(-1.0, 1.0, opts.R, device=dev)])
        self.x = x.view(1, opts.R, opts.R, 1, 1)
        self.y = y.view(1, opts.R, opts.R, 1, 1)

        # Optimization parameters
        adam_beta1 = 0.5
        adam_beta2 = 0.99
        adam_eps   = 1e-08

        # Create optimizers for angles and vertices
        optimizer_angles = optim.Adam([self.angles],
                                       opts.lr_angles, [adam_beta1, adam_beta2], eps=adam_eps)
        optimizer_x0y0   = optim.Adam([self.x0y0],
                                       opts.lr_x0y0,   [adam_beta1, adam_beta2], eps=adam_eps)
        self.optimizers = [optimizer_angles, optimizer_x0y0]

        # Values to search over in Algorithm 2: [0, 2pi) for angles, [-3, 3] for vertex position.
        self.angle_range = torch.linspace(0.0, 2*np.pi, opts.nvals+1, device=dev)[:opts.nvals]
        self.x0y0_range  = torch.linspace(-3.0, 3.0, opts.nvals, device=dev)

        # Save current global image and boundary map (initially None)
        self.global_image      = None
        self.global_boundaries = None
        
        # Save opts
        self.opts = opts

    def optimize(self):
        """
        Optimize field of junctions.
        """
        for iteration in range(self.num_iters):
            self.step(iteration)

    def step(self, iteration):
        """
        Perform one step (either initialization's coordinate descent, or refinement gradient descent)
        Inputs
        ------
        iteration   Iteration number (integer)
        """

        # Linearly increase lambda from 0 to lambda_boundary_final and lambda_color_final
        if self.opts.num_refinement_iters <= 1:
            factor = 0.0
        else:
            factor = max([0, (iteration - self.opts.num_initialization_iters) / (self.opts.num_refinement_iters - 1)])
        lmbda_boundary = factor * self.opts.lambda_boundary_final
        lmbda_color    = factor * self.opts.lambda_color_final

        if iteration < self.opts.num_initialization_iters or \
               (iteration - self.opts.num_initialization_iters + 1) % self.opts.greedy_step_every_iters == 0:
            self.initialization_step(lmbda_boundary, lmbda_color)
        else:
            self.refinement_step(lmbda_boundary, lmbda_color)

    def initialization_step(self, lmbda_boundary, lmbda_color):
        """
        Perform a single coordinate descent step (using Algorithm 2 from the paper).
        Implements a heuristic for searching along the three junction angles after updating each of
        the five parameters. The original value is included in the search, so the extra step is
        guaranteed to obtain a better (or equally-good) set of parameters.

        Inputs
        ------
        lmbda_boundary    Spatial consistency boundary loss weight
        lmbda_color       Spatial consistency color loss weight
        """
        params = torch.cat([self.angles, self.x0y0], dim=1).detach()

        # Run one step of Algorithm 2, sequentially improving each coordinate
        for i in range(5):
            # Repeat the set of parameters `nvals` times along 0th dimension
            params_query = params.repeat(self.opts.nvals, 1, 1, 1)
            param_range = self.angle_range if i < 3 else self.x0y0_range
            params_query[:, i, :, :] = params_query[:, i, :, :] + param_range.view(-1, 1, 1)
            best_ind = self.get_best_inds(params_query, lmbda_boundary, lmbda_color)

            # Update parameters
            params[0, i, :, :] = params_query[best_ind.view(1, self.H_patches, self.W_patches),
                                              i,
                                              torch.arange(self.H_patches).view(1, -1, 1),
                                              torch.arange(self.W_patches).view(1, 1, -1)]

        # Heuristic for accelerating convergence (not necessary but sometimes helps):
        # Update x0 and y0 along the three optimal angles (search over a line passing through current x0, y0)
        for i in range(3):
            params_query = params.repeat(self.opts.nvals, 1, 1, 1)
            params_query[:, 3, :, :] = params[:, 3, :, :] + torch.cos(params[:, i, :, :]) * self.x0y0_range.view(-1, 1, 1)
            params_query[:, 4, :, :] = params[:, 4, :, :] + torch.sin(params[:, i, :, :]) * self.x0y0_range.view(-1, 1, 1)
            best_ind = self.get_best_inds(params_query, lmbda_boundary, lmbda_color)

            # Update vertex positions of parameters
            for j in range(3, 5):
                params[:, j, :, :] = params_query[best_ind.view(1, self.H_patches, self.W_patches),
                                                  j,
                                                  torch.arange(self.H_patches).view(1, -1, 1),
                                                  torch.arange(self.W_patches).view(1, 1, -1)]

        # Update angles and vertex position using the best values found
        self.angles.data = params[:, :3, :, :].data
        self.x0y0.data   = params[:, 3:, :, :].data
        
        # Update global boundaries and image
        dists, colors, patches = self.get_dists_and_patches(params, lmbda_color)
        self.global_image      = self.local2global(patches)
        self.global_boundaries = self.local2global(self.dists2boundaries(dists))


    def refinement_step(self, lmbda_boundary, lmbda_color):
        """
        Perform a single refinement step

        Inputs
        ------
        lmbda_boundary    Spatial consistency boundary loss weight
        lmbda_color       Spatial consistency color loss weight
        """
        params = torch.cat([self.angles, self.x0y0], dim=1)

        # Compute distance functions, colors, and junction patches
        dists, colors, patches = self.get_dists_and_patches(params, lmbda_color)
        
        # Compute loss
        loss = self.get_loss(dists, colors, patches, lmbda_boundary, lmbda_color).mean()

        # Take gradient step over angles and vertex positions
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()

        # Update global boundaries and image
        dists, colors, patches = self.get_dists_and_patches(params, lmbda_color)
        self.global_image      = self.local2global(patches)
        self.global_boundaries = self.local2global(self.dists2boundaries(dists))
            
            
    def get_loss(self, dists, colors, patches, lmbda_boundary, lmbda_color):
        """
        Compute the objective of our model (see Equation 8 of the paper).

        Inputs
        ------
        dists             Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch
        colors            Tensor of shape [N, C, 3, H', W'] storing the C colors at each patch
        patches           Tensor of shape [N, C, R, R, H', W'] with each patch having color c_i^{(j)} at the jth wedge, for each i
        lmbda_boundary    Spatial consistency boundary loss weight
        lmbda_color       Spatial consistency color loss weight

        Outputs
        -------
                 Tensor of shape [N, H', W'] with the loss at each patch
        """
        # Compute negative log-likelihood for each patch (shape [N, H', W'])
        loss_per_patch = ((self.img_patches - patches) ** 2).mean(-3).mean(-3).sum(1)

        # Add spatial consistency loss for each patch, if lambda > 0
        if lmbda_boundary > 0.0:
            loss_per_patch = loss_per_patch + lmbda_boundary * self.get_boundary_consistency_term(dists)

        if lmbda_color > 0.0:
            loss_per_patch = loss_per_patch + lmbda_color * self.get_color_consistency_term(dists, colors)
        
        return loss_per_patch

    def get_boundary_consistency_term(self, dists):
        """
        Compute the spatial consistency term.

        Inputs
        ------
        dists    Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch

        Outputs
        -------
                 Tensor of shape [N, H', W'] with the consistency loss at each patch
        """
        # Split global boundaries into patches
        curr_global_boundaries_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.global_boundaries.detach()).view(1, 1, self.opts.R,self.opts.R, self.H_patches, self.W_patches)

        # Get local boundaries defined using the queried parameters (defined by `dists`)
        local_boundaries = self.dists2boundaries(dists)

        # Compute consistency term
        consistency = ((local_boundaries - curr_global_boundaries_patches) ** 2).mean(2).mean(2)

        return consistency[:, 0, :, :]

    def get_color_consistency_term(self, dists, colors):
        """
        Compute the spatial consistency term.

        Inputs
        ------
        dists    Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch

        Outputs
        -------
                 Tensor of shape [N, H', W'] with the consistency loss at each patch
        """
        # Split into patches
        curr_global_image_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
            self.global_image.detach()).view(1, self.C, self.opts.R,self.opts.R, self.H_patches, self.W_patches)

        wedges = self.dists2indicators(dists)  # shape [N, 3, R, R, H', W']

        # Compute consistency term
        consistency = (wedges.unsqueeze(1) * (
            colors.unsqueeze(-3).unsqueeze(-3) - curr_global_image_patches.unsqueeze(2)) ** 2).mean(-3).mean(-3).sum(1).sum(1)

        return consistency

    
    
    def get_dists_and_patches(self, params, lmbda_color=0.0):
        """
        Compute distance functions and piecewise-constant patches given junction parameters.

        Inputs
        ------
        params   Tensor of shape [N, 5, H', W'] holding N field of junctions parameters. Each
                 5-vector has format (angle1, angle2, angle3, x0, y0).

        Outputs
        -------
        dists    Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch
        colors   Tensor of shape [N, C, 3, H', W']
        patches  Tensor of shape [N, C, R, R, H', W'] with the constant color function at each of the 3 wedges
        """

        # Get dists
        dists = self.params2dists(params)    # shape [N, 2, R, R, H', W']

        # Get wedge indicator functions
        wedges = self.dists2indicators(dists)  # shape [N, 3, R, R, H', W']

        if lmbda_color >= 0 and self.global_image is not None:
            curr_global_image_patches = nn.Unfold(self.opts.R, stride=self.opts.stride)(
                self.global_image.detach()).view(1, self.C, self.opts.R,self.opts.R, self.H_patches, self.W_patches)
            
            numerator = ((self.img_patches + lmbda_color *
                          curr_global_image_patches).unsqueeze(2) * wedges.unsqueeze(1)).sum(-3).sum(-3)
            denominator = (1.0 + lmbda_color) * wedges.sum(-3).sum(-3).unsqueeze(1)
            
            colors = numerator / (denominator + 1e-10)
        else:
            # Get best color for each wedge and each patch
            colors = (self.img_patches.unsqueeze(2) * wedges.unsqueeze(1)).sum(-3).sum(-3) / \
                     (wedges.sum(-3).sum(-3).unsqueeze(1) + 1e-10)

        # Fill wedges with optimal colors
        patches = (wedges.unsqueeze(1) * colors.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)

        return dists, colors, patches

    def dists2boundaries(self, dists):
        """
        Compute boundary map for each patch, given distance functions. The width of the boundary is determined
        by opts.delta.

        Inputs
        ------
        dists    Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch

        Outputs
        -------
                 Tensor of shape [N, 1, R, R, H', W'] with values of boundary map for every patch
        """
        # Find places where either distance transform is small, except where d1 > 0 and d2 < 0
        d1 = dists[:, 0:1, :, :, :, :]
        d2 = dists[:, 1:2, :, :, :, :]
        minabsdist = torch.where(d1 < 0.0, -d1, torch.where(d2 < 0.0, torch.min(d1, -d2), torch.min(d1, d2)))

        return 1.0 / (1.0 + (minabsdist / self.opts.delta) ** 2)

    def local2global(self, patches):
        """
        Compute average value for each pixel over all patches containing it.
        For example, this can be used to compute the global boundary maps, or the boundary-aware smoothed image.

        Inputs
        ------
        patches   Tensor of shape [N, C, R, R, H', W']. patches[n, :, :, :, i, j] is an RxR C-channel patch
                  at the (i, j)th spatial position of the nth entry.


        Outputs
        -------
                  Tensor of shape [N, C, H, W] of averages over all patches containing each pixel.
        """
        N = patches.shape[0]
        C = patches.shape[1]
        return torch.nn.Fold(output_size=[self.H, self.W], kernel_size=self.opts.R, stride=self.opts.stride)(
            patches.view(N, C*self.opts.R**2, -1)).view(N, C, self.H, self.W) / \
                self.num_patches.unsqueeze(0).unsqueeze(0)

    def get_best_inds(self, params, lmbda_boundary, lmbda_color):
        """
        Compute the best index along the 0th dimension of `params` for each pixel position.
        Has two possible modes determined by self.opts.parallel_mode:
        1) When True, all N values are computed in parallel (generally faster, requires more memory)
        2) When False, the values are computed sequentially (generally slower, requires less memory)

        Inputs
        ------
        params            Tensor of shape [N, 5, H', W'] holding N field of junctions parameters. Each
                          5-vector has format (angle1, angle2, angle3, x0, y0).
        lmbda_boundary    Spatial consistency boundary loss weight
        lmbda_color       Spatial consistency color loss weight

        Outputs
        -------
                          Tensor of shape [H', W'] with each value in {0, ..., N-1} holding the
                          index of the best junction parameters at that position.
        """
        if self.opts.parallel_mode:
            dists, colors, smooth_patches = self.get_dists_and_patches(params, lmbda_color)
            loss_per_patch = self.get_loss(dists, colors, smooth_patches, lmbda_boundary, lmbda_color)
            best_ind = loss_per_patch.argmin(dim=0)

        else:
            # First initialize tensors
            best_ind            = torch.zeros(self.H_patches, self.W_patches, device=dev, dtype=torch.int64)
            best_loss_per_patch = torch.zeros(self.H_patches, self.W_patches, device=dev) + 1e10

            # Now fill tensors by iterating over the junction dimension and choosing the best junction parameters
            for n in range(params.shape[0]):
                dists, colors, smooth_patches = self.get_dists_and_patches(params[n:n+1, :, :, :], lmbda_color)

                loss_per_patch = self.get_loss(dists, colors, smooth_patches, lmbda_boundary, lmbda_color)

                improved_inds       = loss_per_patch[0] < best_loss_per_patch
                best_ind            = torch.where(improved_inds, torch.tensor(n, device=dev, dtype=torch.int64), best_ind)
                best_loss_per_patch = torch.where(improved_inds, loss_per_patch, best_loss_per_patch)

        return best_ind

    def params2dists(self, params, tau=1e-1):
        """
        Compute distance functions from field of junctions.

        Inputs
        ------
        params   Tensor of shape [N, 5, H', W'] holding N field of junctions parameters. Each
                 5-vector has format (angle1, angle2, angle3, x0, y0).
        tau      Constant used for lifting the level set function to be either entirely positive
                 or entirely negative when an angle approaches 0 or 2pi.


        Outputs
        -------
                 Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch
        """
        x0     = params[:, 3, :, :].unsqueeze(1).unsqueeze(1)   # shape [N, 1, 1, H', W']
        y0     = params[:, 4, :, :].unsqueeze(1).unsqueeze(1)   # shape [N, 1, 1, H', W']

        # Sort so angle1 <= angle2 <= angle3 (mod 2pi)
        angles = torch.remainder(params[:, :3, :, :], 2 * np.pi)
        angles = torch.sort(angles, dim=1)[0]

        angle1 = angles[:, 0, :, :].unsqueeze(1).unsqueeze(1)   # shape [N, 1, 1, H', W']
        angle2 = angles[:, 1, :, :].unsqueeze(1).unsqueeze(1)   # shape [N, 1, 1, H', W']
        angle3 = angles[:, 2, :, :].unsqueeze(1).unsqueeze(1)   # shape [N, 1, 1, H', W']

        # Define another angle halfway between angle3 and angle1, clockwise from angle3
        # This isn't critical but it seems a bit more stable for computing gradients
        angle4 = 0.5 * (angle1 + angle3) + \
                     torch.where(torch.remainder(0.5 * (angle1 - angle3), 2 * np.pi) >= np.pi,
                                 torch.ones_like(angle1) * np.pi, torch.zeros_like(angle1))

        def g(dtheta):
            # Map from [0, 2pi] to [-1, 1]
            return (dtheta / np.pi - 1.0) ** 35

        # Compute the two distance functions
        sgn42 = torch.where(torch.remainder(angle2 - angle4, 2 * np.pi) < np.pi,
                            torch.ones_like(angle2), -torch.ones_like(angle2))
        tau42 = g(torch.remainder(angle2 - angle4, 2*np.pi)) * tau

        dist42 = sgn42 * torch.min( sgn42 * (-torch.sin(angle4) * (self.x - x0) + torch.cos(angle4) * (self.y - y0)),
                                   -sgn42 * (-torch.sin(angle2) * (self.x - x0) + torch.cos(angle2) * (self.y - y0))) + tau42

        sgn13 = torch.where(torch.remainder(angle3 - angle1, 2 * np.pi) < np.pi,
                            torch.ones_like(angle3), -torch.ones_like(angle3))
        tau13 = g(torch.remainder(angle3 - angle1, 2*np.pi)) * tau
        dist13 = sgn13 * torch.min( sgn13 * (-torch.sin(angle1) * (self.x - x0) + torch.cos(angle1) * (self.y - y0)),
                                   -sgn13 * (-torch.sin(angle3) * (self.x - x0) + torch.cos(angle3) * (self.y - y0))) + tau13

        return torch.stack([dist13, dist42], dim=1)

    def dists2indicators(self, dists):
        """
        Computes the indicator functions u_1, u_2, u_3 from the distance functions d_{13}, d_{12}

        Inputs
        ------
        dists   Tensor of shape [N, 2, R, R, H', W'] with samples of the two distance functions for every patch

        Outputs
        -------
                Tensor of shape [N, 3, R, R, H', W'] with samples of the three indicator functions for every patch
        """
        # Apply smooth Heaviside function to distance functions
        hdists = 0.5 * (1.0 + (2.0 / np.pi) * torch.atan(dists / self.opts.eta))

        # Convert Heaviside functions into wedge indicator functions
        return torch.stack([1.0 - hdists[:, 0, :, :, :, :],
                                  hdists[:, 0, :, :, :, :] * (1.0 - hdists[:, 1, :, :, :, :]),
                                  hdists[:, 0, :, :, :, :] *        hdists[:, 1, :, :, :, :]], dim=1)
