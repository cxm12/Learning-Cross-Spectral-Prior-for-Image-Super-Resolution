import os
import glob
from data import common
import torch.utils.data as data
import numpy as np
from PIL import Image


class SRData(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.gamma = args.gammacorrect
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.scale = args.scale
        self.idx_scale = 0
        
        if self.train:
            self._set_filesystem(args.dir_data)
    
            list_rgb, list_nir = self._scan()
            self.images_rgb, self.images_nir = list_rgb, list_nir
            
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_rgb)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)
        else:
            self._set_filesystem(args.dir_demo)
    
            list_rgb, list_nir = self._scan()
            self.images_rgb, self.images_nir = list_rgb, list_nir

    # Below functions as used to prepare images
    def _scan(self):
        names_rgb = []
        names_nir = []
        for impath in self.dir_rgb:
            if self.train:
                names_rgb = sorted(
                    glob.glob(os.path.join(self.args.dir_data + '/' + impath, '*' + self.ext[0]))
                )
            else:
                names_rgb = sorted(
                    glob.glob(os.path.join(self.args.dir_demo + '/' + impath, '*' + self.ext[0]))
                )
                names_rgb = names_rgb
                # names_rgb = names_rgb[:5]
        for i in range(len(names_rgb)):
            names_nir.append(names_rgb[i].replace('RGBResize', 'NIRResize'))

        return names_rgb, names_nir

    def get_filenames(self, paths):
        filenames = []
        for path in paths:
            for root, dirs, files in os.walk(path):
                for f in files:
                    filenames.append(os.path.join(root, f))
        return filenames

    def _set_filesystem(self, dir_data):
        self.apath = os.listdir(dir_data)
        self.dir_rgb = []
        self.dir_nir = []
        if self.train:
            for file in self.apath:
                self.dir_rgb.append(os.path.join(file, 'RGBResize'))
                self.dir_nir.append(os.path.join(file, 'NIRResize'))
        else:
            for file in self.apath:
                self.dir_rgb.append(os.path.join(file, 'RGBResize/GT'))
                self.dir_nir.append(os.path.join(file, 'NIRResize/GT'))
                
        self.ext = ('.png', '.png')

    def __getitem__(self, idx):
        hrrgb, hrnir, filename = self._load_file(idx)
        # hrnir = np.expand_dims(hrnir, -1)
        pair, pairnir = self.get_patch(hrrgb, hrnir)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pairnir = common.set_channel(*pairnir, n_channels=1)
        
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        pair_tnir = common.np2Tensor(*pairnir, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], pair_tnir[0], pair_tnir[1], filename

    def __len__(self):
        if self.train:
            return len(self.images_rgb) * self.repeat
        else:
            return len(self.images_rgb)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_rgb)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_rgb = self.images_rgb[idx]
        f_nir = self.images_nir[idx]
        
        filename, _ = os.path.splitext(os.path.basename(f_rgb))
        hrrgb = np.array(Image.open(f_rgb))
        hrnir = np.array(Image.open(f_nir))
        
        if self.gamma:
            hrrgb = np.uint8(np.power(hrrgb / 255.0, 0.55) * 255.0)
            
        # size = tuple(((np.array(Image.fromarray(hrrgb).size)).astype(int) / self.scale).astype(int))
        # lrrgb = np.array(Image.fromarray(hrrgb).resize(size, Image.BICUBIC))
        # lrnir = np.array(Image.fromarray(hrnir).resize(size, Image.BICUBIC))

        return hrrgb, hrnir, filename

    def get_patch(self, hrrgb, hrnir):
        scale = self.scale[self.idx_scale]
        if self.train:
            hrrgb, lrrgb, hrnir, lrnir = common.get_patchrgb(
                hrrgb, hrnir,
                scale=scale,
                patch_size=self.args.patch_size
            )
            hrnir = np.expand_dims(hrnir, -1)
            lrnir = np.expand_dims(lrnir, -1)
            if not self.args.no_augment: hrrgb, lrrgb, hrnir, lrnir = common.augment(hrrgb, lrrgb, hrnir, lrnir)
        else:
            ih, iw = hrrgb.shape[:2]
            size = tuple(((np.array(Image.fromarray(hrrgb).size)).astype(int) / scale).astype(int))
            lrrgb = np.array(Image.fromarray(hrrgb).resize(size, Image.BICUBIC))
            lrnir = np.array(Image.fromarray(hrnir).resize(size, Image.BICUBIC))
    
            hrnir = hrnir[0:(ih // scale) * scale, 0:(iw // scale) * scale]
            hrrgb = hrrgb[0:(ih // scale) * scale, 0:(iw // scale) * scale]
    
        return (hrrgb, lrrgb), (hrnir, lrnir)
    
    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
