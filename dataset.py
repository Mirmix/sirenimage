import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import laplace, sobel

def generate_coordinates(n):

    rows, cols = np.meshgrid(range(n), range(n), indexing="ij")
    coords_abs = np.stack([rows.ravel(), cols.ravel()], axis=-1)

    return coords_abs

class PixelDataset(Dataset):

    def __init__(self, img):
        if not (img.ndim == 2 and img.shape[0] == img.shape[1]):
            raise ValueError("Only 2D square images are supported.")

        self.img = img
        self.size = img.shape[0]
        self.coords_abs = generate_coordinates(self.size)
        self.grad = np.stack([sobel(img, axis=0), sobel(img, axis=1)], axis=-1)
        self.grad_norm = np.linalg.norm(self.grad, axis=-1)
        self.laplace = laplace(img)

    def __len__(self):
        return self.size ** 2

    def __getitem__(self, idx):
        coords_abs = self.coords_abs[idx]
        r, c = coords_abs

        coords = 2 * ((coords_abs / self.size) - 0.5)

        return {
            "coords": coords,
            "coords_abs": coords_abs,
            "intensity": self.img[r, c],
            "grad_norm": self.grad_norm[r, c],
            "grad": self.grad[r, c],
            "laplace": self.laplace[r, c],
        }
