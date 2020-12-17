import numpy as np
import torch.nn.functional as F


class RandomResizeVideo:
    def __init__(
            self,
            short_side,
            interpolation_mode='bilinear'
    ):
        self.interpolation_mode = interpolation_mode
        self.short_side = short_side

    def __call__(self, video):
        shape = video.shape
        rand_int = np.random.randint(self.short_side[0], self.short_side[1])
        if shape[3] > shape[2]:
            reshape_height = rand_int
            reshape_width = int(float(reshape_height) / shape[2] * shape[3])
        else:
            reshape_width = rand_int
            reshape_height = int(float(reshape_width) / shape[3] * shape[2])
        return F.interpolate(video, size=(reshape_height, reshape_width), mode=self.interpolation_mode,
                             align_corners=False)
