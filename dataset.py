import scipy
import torch
import numpy as np
from torch.utils.data import Dataset


class HSI(Dataset):
    """ --- """

    DATASET_NAMES = ["IP220", "DC191", "PU103", "SA224", "HO144", "DC191", "G5150", "KS176", "HH270", "HG176"]

    def __init__(self, name, dir_dataset, patch_shape=(7, 7)):
        super(HSI, self).__init__()

        # Load HSI, standardize, transpose and create channel dim.
        hsi_3d = self.load_dataset_by_name(name, dir_dataset)[0]  # data_3d, gt
        hsi_3d = self.standardize(hsi_3d)
        hsi_3d = hsi_3d.transpose(2, 0, 1)  # (h, w, band_weights) => (band_weights, h, w)

        # Crop the un batched pixels. eg: (220, 145, 145) -> (220, 140=7x20, 140=7x20)
        num_bands, h, w = hsi_3d.shape
        px, py = patch_shape
        new_h = int(h / px) * px
        new_w = int(w / py) * py
        hsi_3d = hsi_3d[:, :new_h, :new_w]
        self.num_bands = num_bands
        self.h = new_h
        self.w = new_w

        # Split HSI into patch.
        self.coords = []
        self.patches = []
        for x1 in range(0, new_h, px):
            for y1 in range(0, new_w, py):
                x2 = x1 + px
                y2 = y1 + py
                coord = (x1, x2, y1, y2)
                patch = hsi_3d[:, x1:x2, y1:y2]
                patch = torch.Tensor(patch)
                self.coords.append(coord)
                self.patches.append(patch)

    def __len__(self):
        return len(self.patches)  # self.patches has the same length with coords

    def __getitem__(self, idx):
        coord = self.coords[idx]
        patch = self.patches[idx]
        return coord, patch

    @staticmethod
    def standardize(data_3d):
        val_mean = np.mean(data_3d)
        val_sigma = np.std(data_3d)
        return (data_3d - val_mean) / val_sigma

    @classmethod
    def load_dataset_by_name(cls, name, dir_dataset):
        data_3d, gt = None, None

        if name == "IP220":
            data_3d = scipy.io.loadmat(rf'{dir_dataset}/IP220/Indian_pines.mat')["indian_pines"]
            gt = scipy.io.loadmat(rf'{dir_dataset}/IP220/Indian_pines_gt.mat')["indian_pines_gt"]

        elif name == "HO144":
            mat = scipy.io.loadmat(rf'{dir_dataset}/HO144/Houston_2013.mat')
            data_3d = mat["Houston_img"]
            gt = mat["Houston_gt"]

        elif name == "PU103":
            data_3d = scipy.io.loadmat(rf'{dir_dataset}/PU103/paviaU.mat')["paviaU"]
            gt = scipy.io.loadmat(rf'{dir_dataset}/PU103/paviaU_gt.mat')["paviaU_gt"]

        elif name == "SA224":
            data_3d = scipy.io.loadmat(rf'{dir_dataset}/SA224/Salinas.mat')["salinas"]
            gt = scipy.io.loadmat(rf'{dir_dataset}/SA224/Salinas_gt.mat')["salinas_gt"]

        elif name == "DC191":
            mat = scipy.io.loadmat(rf'{dir_dataset}/DC191/DC_Sub.mat')
            data_3d = mat["DC_Sub"]
            gt = mat["grd"]

        elif name == "KS176":
            data_3d = scipy.io.loadmat(rf'{dir_dataset}/KS176/KSC.mat')["KSC"]
            gt = scipy.io.loadmat(rf'{dir_dataset}/KS176/KSC_gt.mat')["KSC_gt"]

        elif name == "G5150":
            data_3d = scipy.io.loadmat(rf'{dir_dataset}/G5150/gf5.mat')["gf5"]
            gt = scipy.io.loadmat(rf'{dir_dataset}/G5150/gf5_gt.mat')["gf5_gt"]

        elif name == "HH270":
            data_3d = scipy.io.loadmat(rf'{dir_dataset}/HH270/HongHu.mat')["HongHu"]
            gt = scipy.io.loadmat(rf'{dir_dataset}/HH270/HongHu_gt.mat')["HongHu_gt"]

        elif name == "PA176":
            data_3d = scipy.io.loadmat(rf'{dir_dataset}/PA176/QUH-Pingan_Sub.mat')["Haigang_sub"]
            gt = scipy.io.loadmat(rf'{dir_dataset}/PA176/QUH-Pingan_Sub_GT.mat')["HaigangGT_sub"]

        return data_3d, gt
