from types import SimpleNamespace

import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path

from field_of_junctions import FieldOfJunctions


def get_default_configs():
    cofnig = SimpleNamespace()
    cofnig.R = 21
    cofnig.stride = 1
    cofnig.eta = 0.01
    cofnig.delta = 0.05
    cofnig.lr_angles = 0.003
    cofnig.lr_x0y0 = 0.03
    cofnig.lambda_boundary_final = 0.5
    cofnig.lambda_color_final = 0.1
    cofnig.nvals = 31
    cofnig.num_initialization_iters = 30
    cofnig.num_refinement_iters = 1000
    cofnig.greedy_step_every_iters = 50
    cofnig.parallel_mode = True

    return cofnig


def normalize_img(img):
    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn)
    return img


class Predictor(BasePredictor):
    def setup(self):
        self.config = get_default_configs()

    def predict(self,
                input_image: Path = Input(description="Image with a face to search"),
                output_type: str = Input(description="Edge map or smoothed image", default="smooth", choices=['smooth', 'edge']),
                max_height: int = Input(description="Resize image to fit this size and save memory", default=128),
                patch_size: int = Input(default=21),
                stride: int = Input(description="stride of patches", default=1),
                parallel_mode: bool = Input(description="Faster but more memory consuming", default=False)
                ) -> Path:
        self.config.R = int(patch_size)
        self.config.stride = int(stride)
        self.config.parallel_mode = bool(parallel_mode)
        input_image = str(input_image)

        img = cv2.imread(input_image)

        # Resuze and  Normalize image
        h,w = img.shape[:2]
        img = cv2.resize(img, (w * max_height // h, max_height))
        img = img.astype(np.float32)
        img = normalize_img(img)

        # Verify stride ok
        hnp = (img.shape[0] - self.config.R) % self.config.stride
        wnp = (img.shape[1] - self.config.R) % self.config.stride
        if (hnp != 0) or (wnp != 0):
            raise ValueError(f"Stride does not match image size, Image cannot be fully split into strided patches\n"
                             f"(H - r) % s = {hnp}, (W-r) % s = {wnp} they should both be 0.")

        # FOJ
        foj = FieldOfJunctions(img, self.config)

        for i in range(foj.num_iters):
            if i == 0:
                print("Beginning initialization...")
            if i == self.config.num_initialization_iters:
                print("Initialization done. Beginning refinement...")
            if i < self.config.num_initialization_iters:
                if i % 5 == 0:
                    print(f"Initialization iteration {i}/{self.config.num_initialization_iters}")
            else:
                if i % 100 == 0:
                    print(f"Refinement iteration {i}/{self.config.num_refinement_iters}")
            foj.step(i)

        # Compute smoothed image and boundary map
        params = torch.cat([foj.angles, foj.x0y0], dim=1)
        dists, _, patches = foj.get_dists_and_patches(params)

        # Aggregate output
        out_path = 'output.png'
        if output_type == 'smooth':
            out_img = foj.local2global(patches)[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()

        else:
            local_boundaries = foj.dists2boundaries(dists)
            out_img = foj.local2global(local_boundaries)[0, 0, :, :].detach().cpu().numpy()

        # Restore image shape and value range
        out_img = cv2.resize(out_img, (w, h), cv2.INTER_NEAREST)
        out_img = normalize_img(out_img)
        out_img = out_img*255

        cv2.imwrite(out_path, out_img)
        return Path(out_path)
