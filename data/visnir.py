from data.srdata import *
import os
import glob


class VISNIR(SRData):
    def __init__(self, args, train=True, valid=True, oneim=False, cropHRfirst=False):
        super(VISNIR, self).__init__(
            args, train=train
        )
        self.cropHRfirst = cropHRfirst
        self.args = args
        self.train = train
        self.valid = valid
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.scale = args.scale
        self.idx_scale = 0
    
        if self.train:
            self._set_filesystem(args.dir_data)
            list_rgb, list_nir = self._scan()
            self.images_rgb, self.images_nir = list_rgb, list_nir
            print('len Train patches (self.images_rgb)', len(self.images_rgb))
            if len(self.images_rgb) == 0:
                exit()
            self.repeat = 1
        else:
            self._set_filesystem(args.dir_demo)
            list_rgb, list_nir = self._scan()
            self.images_rgb, self.images_nir = list_rgb, list_nir
            if self.valid:
                self.images_rgb, self.images_nir = list_rgb[:1], list_nir[:1]
            if oneim:
                self.images_rgb, self.images_nir = list_rgb[20:21], list_nir[20:21]

    # Below functions as used to prepare images
    def _scan(self):
        if self.train:
            hrp = self.args.dir_data
        else:
            hrp = self.args.dir_demo

        names_rgb = []
        names_nir = []
        for impath in self.dir_rgb:
            if 'RANUS' in self.apath:
                names_rgb.extend(sorted(
                    glob.glob(hrp + '/' + impath + '/*.png')
                ))
                names_nir.extend(sorted(
                    glob.glob(hrp.replace('RGB_sub', 'NIR_sub') + '/' + impath + '/*.png')
                ))
                # nmid = names_rgb[i].replace('RGB', 'NIR')[len(self.args.dir_demo + '/' + impath +'/100'):-5]
                # nm = names_rgb[i].replace('RGB', 'NIR')[:len(self.args.dir_demo + '/' + impath +'/')] + '5'+nmid+ '_nir_.png'
                # names_nir.append(nm)
            elif 'nirscene' in self.apath:
                names_rgb.extend(sorted(
                    glob.glob(hrp + '/' + impath + '/*rgb*.tiff')
                ))
                names_nir.extend(sorted(
                    glob.glob(hrp + '/' + impath + '/*nir*.tiff')
                ))
                # names_nir.append(names_rgb[i].replace('_rgb', '_nir'))
    
        return names_rgb, names_nir

    def _set_filesystem(self, dir_data):
        # self.apath = os.listdir(dir_data)
        self.apath = dir_data
        self.dir_rgb = os.listdir(dir_data)
        
    def __getitem__(self, idx):
        hrrgb, hrnir, filename, d = self._load_file(idx)
        # hrnir = np.expand_dims(hrnir, -1)
        pair, pairnir = self.get_patch(hrrgb, hrnir)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)  # uint8 [0, 255]
        pairnir = common.set_channel(*pairnir, n_channels=1)
    
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        pair_tnir = common.np2Tensor(*pairnir, rgb_range=self.args.rgb_range)
    
        return pair_t[0], pair_t[1], pair_tnir[0], pair_tnir[1], filename, d
        
    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_rgb = self.images_rgb[idx]
        f_nir = self.images_nir[idx]
    
        d = os.path.dirname(f_rgb)[len(self.apath)+1:]
        filename, _ = os.path.splitext(os.path.basename(f_rgb))
        hrrgb = np.array(Image.open(f_rgb))  # uint8 [0,255]
        hrnir = np.array(Image.open(f_nir))
        h, w, c = hrrgb.shape
        h = h // (16 * self.scale[0]) * (16 * self.scale[0])  # 429
        w = w // (16 * self.scale[0]) * (16 * self.scale[0])  # 582
        hrrgb = hrrgb[:h, :w, :]
        hrnir = hrnir[:h, :w]
        if 'RANUS' in self.apath:
            hrnir = np.mean(hrnir, -1)
        # if self.gamma:
        #     hrrgb = np.uint8(np.power(hrrgb / 255.0, 0.55) * 255.0)
        return hrrgb, hrnir, filename, d

    def get_patch(self, hrrgb, hrnir):
        scale = self.scale[self.idx_scale]
        if self.train or self.valid:
            if scale == 3:
                hrrgb, lrrgb, hrnir, lrnir = common.get_patchrgb(
                    hrrgb, hrnir, scale=scale, patch_size=self.args.patch_size
                )
            else:
                hrrgb, lrrgb, hrnir, lrnir = common.get_LRrgb(
                    hrrgb, hrnir, scale=scale
                )
           
            hrnir = np.expand_dims(hrnir, -1)
            lrnir = np.expand_dims(lrnir, -1)
            if not self.args.no_augment: hrrgb, lrrgb, hrnir, lrnir = common.augment(hrrgb, lrrgb, hrnir, lrnir)
        else:
            ih, iw = hrrgb.shape[:2]
            size = tuple(((np.array(Image.fromarray(hrrgb).size)).astype(int) / scale).astype(int))
            
            if self.cropHRfirst:
                hrnir = hrnir[0:(ih // scale) * scale, 0:(iw // scale) * scale]
                hrrgb = hrrgb[0:(ih // scale) * scale, 0:(iw // scale) * scale]  # uint8 [0, 255]
    
                lrrgb = np.array(Image.fromarray(hrrgb).resize(size, Image.BICUBIC))
                lrnir = np.array(Image.fromarray(hrnir).resize(size, Image.BICUBIC))
            else:
                lrrgb = np.array(Image.fromarray(hrrgb).resize(size, Image.BICUBIC))
                lrnir = np.array(Image.fromarray(hrnir).resize(size, Image.BICUBIC))
                
                hrnir = hrnir[0:(ih // scale) * scale, 0:(iw // scale) * scale]
                hrrgb = hrrgb[0:(ih // scale) * scale, 0:(iw // scale) * scale]  # uint8 [0, 255]

        return (hrrgb, lrrgb), (hrnir, lrnir)
