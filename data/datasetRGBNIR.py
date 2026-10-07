import os
from pathlib import Path
import numpy as np
import torch.utils.data as data
import cv2
from data import common
import torch
import glob
from PIL import Image


class Set5Data(data.Dataset):
    def __init__(self, args, inputpath):
        self.args = args
        self.gamma = True
        self.do_eval = True
        self.scale = args.scale
        self.idx_scale = 0
        
        self.images_rgb = glob.glob(inputpath)

    def __getitem__(self, idx):
        pair, pairnir, filename = self._load_file(idx)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pairnir = common.set_channel(*pairnir, n_channels=1)
        
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        pair_tnir = common.np2Tensor(*pairnir, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], pair_tnir[0], pair_tnir[1], filename

    def __len__(self):
        return len(self.images_rgb)

    def _load_file(self, idx):
        f_rgb = self.images_rgb[idx]
        
        filename, _ = os.path.splitext(os.path.basename(f_rgb))
        hrrgb = np.array(Image.open(f_rgb))
        hrrgb = np.uint8(np.power(hrrgb / 255.0, 0.55) * 255.0)
        scale = self.scale[self.idx_scale]
        
        ih, iw = hrrgb.shape[:2]
        ih = (ih // 64) * 64
        iw = (iw // 64) * 64
        hrrgb = hrrgb[0:ih, 0:iw]
        hrnir = hrrgb[:, :, :1]
        
        size = tuple(((np.array(Image.fromarray(hrrgb).size)).astype(int) / scale).astype(int))
        lrrgb = np.array(Image.fromarray(hrrgb).resize(size, Image.BICUBIC))
        lrnir = lrrgb[:, :, :1]
    
        return (hrrgb, lrrgb), (hrnir, lrnir), filename


class StereoDataset(data.Dataset):

    def __init__(self, data_path, list_path, splits):
        super(StereoDataset, self).__init__()
        self.data_path = data_path
        self.height = 429
        self.width = 582
        self.im_suff = 'Resize'
        self.mat_suff = 'Material'
        self.records = []
        for split in splits:
            f = open(Path(list_path) / (split + '.txt'), 'r')
            lines = f.readlines()
            f.close()
            for i, line in enumerate(lines):
                splited = line.split()
                collection = splited[0]
                key = splited[1]
                rgb_exp = float(splited[2])
                nir_exp = float(splited[3])
                record = (collection, key, rgb_exp, nir_exp)
                self.records.append(record)
        self.records = sorted(self.records)
        self.n_records = len(self.records)

    def __getitem__(self, index):
        collection, key, rgb_exp, nir_exp = self.records[index]
        rgb = cv2.imread(self.fname(collection, key, self.im_suff, 'RGB', 'png'))
        assert(rgb.shape == (self.height, self.width, 3))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.transpose([2, 0, 1])
        rgb = rgb.astype(np.float32)
        nir = cv2.imread(self.fname(collection, key, self.im_suff, 'NIR', 'png'), cv2.IMREAD_GRAYSCALE)
        assert(nir.shape == (self.height, self.width))
        nir = nir[np.newaxis][:]
        nir = nir.astype(np.float32)
        rgb_exp = np.array([rgb_exp]).astype(np.float32)
        nir_exp = np.array([nir_exp]).astype(np.float32)
        material = np.load(self.fname(collection, key, self.mat_suff, '', 'npz'))
        rgb_mat = material['rgb']
        nir_mat = material['nir']
        return collection, key, rgb, nir, rgb_exp, nir_exp, rgb_mat, nir_mat

    def __len__(self):
        return self.n_records

    def fname(self, collection, key, suffix, camera, ftype):
        return str(Path(self.data_path) / collection / (camera + suffix) / (key + '_' + camera + suffix + '.' + ftype))


class StereoDatasetTestWarp(data.Dataset):
    # 有exposure
    def __init__(self, data_path, list_path, splits, istrain, warpnir=None, scale=4, testonly=False, patch=192):
        super(StereoDatasetTestWarp, self).__init__()
        self.testonly = testonly
        self.scale = [scale[0]]
        if self.scale == 3:
            self.patch_size = 180  # 210  #
        elif self.scale == 2:
            self.patch_size = 120  # 160  #
        else:
            self.patch_size = patch  # 160  # 256  #
        self.train = istrain
        self.data_path = data_path
        self.height = 429
        self.width = 582
        self.im_suff = 'Resize'
        self.records = []
        for split in splits:
            f = open(Path(list_path) / (split + '.txt'), 'r')
            lines = f.readlines()
            f.close()
            if self.train:
                for i, line in enumerate(lines):
                    if '20170222_0951' in line or '20170222_1423' in line or '20170223_1639' in line:
                        if i % 5 == 0:
                            continue
                    splited = line.split()
                    '''collection, imgID, RGBExposureTime, NIRExposureTime, RedGain, BlueGain.
                    （Expusre times are in microseconds. Red gain and blue gain are white balancing parameters）'''
                    collection = splited[0]
                    key = splited[1]
                    rgb_exp = float(splited[2])
                    nir_exp = float(splited[3])
                    record = (collection, key, rgb_exp, nir_exp)
                    self.records.append(record)
            else:
                for i, line in enumerate(lines):
                    if i % 5 == 0:
                        splited = line.split()
                        '''collection, imgID, RGBExposureTime, NIRExposureTime, RedGain, BlueGain.
                        （Expusre times are in microseconds. Red gain and blue gain are white balancing parameters）'''
                        collection = splited[0] + '100'
                        key = splited[1]
                        rgb_exp = float(splited[2])
                        nir_exp = float(splited[3])
                        record = (collection, key, rgb_exp, nir_exp)
                        self.records.append(record)
        self.records = sorted(self.records)
        self.n_records = len(self.records)
        self.warpnir = warpnir
    
    def get_patch(self, hrrgb, hrnir):
        scale = self.scale[0]
        if self.train:
            hrrgb, lrrgb, hrnir, lrnir = common.get_patchrgb(
                hrrgb, hrnir,
                patch_size=self.patch_size,
                scale=scale
            )
            '''
            hrrgb, lrrgb, hrnir, lrnir = common.get_patchrgbCV18(
                hrrgb, hrnir,
                scale=scale,
                patch_size=self.patch_size
            )
            '''
            hrnir = np.expand_dims(hrnir, -1)
            lrnir = np.expand_dims(lrnir, -1)
            hrrgb, lrrgb, hrnir, lrnir = common.augment(hrrgb, lrrgb, hrnir, lrnir)
        else:
            hrrgb, lrrgb, hrnir, lrnir = common.get_LR(
                hrrgb, hrnir,
                scale=scale
            )
            '''
            hrrgb, lrrgb, hrnir, lrnir = common.get_LRCV18(
                hrrgb, hrnir,
                scale=scale
            )
            '''
            hrnir = np.expand_dims(hrnir, -1)
            lrnir = np.expand_dims(lrnir, -1)
            # hrrgb, lrrgb, hrnir, lrnir = common.augment(hrrgb, lrrgb, hrnir, lrnir)
        return hrrgb, lrrgb, hrnir, lrnir
    
    def __getitem__(self, index):
        # index = 82  # 240742_057997
        collection, key, rgb_exp, nir_exp = self.records[index]
        
        rgb = cv2.imread(self.fname(collection, key, self.im_suff, 'RGB', 'png'))
        assert (rgb.shape == (self.height, self.width, 3))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32)
        # rgblr = cv2.resize(rgb, [self.height//self.scale, self.width//self.scale])
        
        if self.warpnir:
            # print(self.warpnir + collection + key + 'png')
            nir = cv2.imread(self.warpnir + collection + '/' + key + '.png', cv2.IMREAD_GRAYSCALE)
        else:  # str(Path(self.data_path) / collection / (camera + suffix) / (key + '_' + camera + suffix + '.' + ftype))
            nir = cv2.imread(self.fname(collection, key, self.im_suff, 'NIR', 'png'), cv2.IMREAD_GRAYSCALE)
        
        assert (nir.shape == (self.height, self.width))
        nir = nir.astype(np.float32)
        rgb, rgblr, nir, nirlr = self.get_patch(rgb, nir)
        rgb = rgb.transpose([2, 0, 1])
        rgblr = rgblr.transpose([2, 0, 1])
        nir = nir.transpose([2, 0, 1])  # [np.newaxis][:]
        nirlr = nirlr.transpose([2, 0, 1])  # [np.newaxis][:]
        
        rgb_exp = np.array([rgb_exp]).astype(np.float32)
        nir_exp = np.array([nir_exp]).astype(np.float32)
        
        rgb = np.ascontiguousarray(rgb).copy()
        rgblr = np.ascontiguousarray(rgblr).copy()
        nir = np.ascontiguousarray(nir).copy()
        nirlr = np.ascontiguousarray(nirlr).copy()
        rgb = torch.from_numpy(rgb).float()
        rgblr = torch.from_numpy(rgblr).float()
        nir = torch.from_numpy(nir).float()
        nirlr = torch.from_numpy(nirlr).float()
        
        return collection, key, rgb, nir, rgblr, nirlr, rgb_exp, nir_exp
    
    def __len__(self):
        if not self.train:
            if self.testonly:
                return self.n_records  # 0
            else:
                return 10
        else:
            return self.n_records
    
    def fname(self, collection, key, suffix, camera, ftype):
        return str(Path(self.data_path) / collection / (camera + suffix) / (key + '_' + camera + suffix + '.' + ftype))


class StereoDatasetTestWarpColorCorrect(data.Dataset):
    # 有exposure
    def __init__(self, data_path, list_path, splits, istrain, warpnir=None, scale=4, testonly=False, patch=192, cropHRfirst=False):
        super(StereoDatasetTestWarpColorCorrect, self).__init__()
        self.testonly = testonly
        self.cropHRfirst = cropHRfirst
        self.scale = [scale[0]]
        print('Scale $$$$$$$$$$$$$$ ', self.scale)
        if self.scale[0] == 3:
            self.patch_size = 120  # 210  #
        elif self.scale[0] == 2:
            self.patch_size = 128  # 120  #
        else:
            self.patch_size = 256  # 256  #
        self.train = istrain
        self.data_path = data_path
        self.height = 429
        self.width = 582
        self.height1 = 429 // (16 * self.scale[0]) * (16 * self.scale[0])  # 429
        self.width1 = 582 // (16 * self.scale[0]) * (16 * self.scale[0])  # 582
        
        self.im_suff = 'Correct'
        self.records = []
        for split in splits:
            f = open(Path(list_path) / (split + '.txt'), 'r')
            lines = f.readlines()
            f.close()
            if self.train:
                for i, line in enumerate(lines):
                    if '20170224_0742' in line or '20170222_0951' in line or '20170222_1423' in line or '20170223_1639' in line:
                        # continue
                        if i % 5 == 0:
                            continue
                    splited = line.split()
                    collection = splited[0]
                    key = splited[1]
                    record = (collection, key)
                    self.records.append(record)
            else:
                for i, line in enumerate(lines):
                    if i % 5 == 0:
                        splited = line.split()
                        '''collection, imgID, RGBExposureTime, NIRExposureTime, RedGain, BlueGain'''
                        collection = splited[0] + '100'
                        key = splited[1]
                        record = (collection, key)
                        self.records.append(record)
        self.records = sorted(self.records)
        if (not self.train) and (not self.testonly):
            self.records = self.records[:10]
        self.n_records = len(self.records)
        print(self.n_records, ' = self.n_records')
        self.warpnir = warpnir
        
    def get_patch(self, hrrgb, hrnir):
        scale = self.scale[0]
        if self.train:
            hrrgb, lrrgb, hrnir, lrnir = common.get_patchrgb(
                hrrgb, hrnir,
                patch_size=self.patch_size,
                scale=scale
            )
            hrnir = np.expand_dims(hrnir, -1)
            lrnir = np.expand_dims(lrnir, -1)
            hrrgb, lrrgb, hrnir, lrnir = common.augment(hrrgb, lrrgb, hrnir, lrnir)
        else:
            if self.cropHRfirst:  # PSNR higher accurate LR
                hrrgb, lrrgb, hrnir, lrnir = common.get_LR(
                    hrrgb, hrnir,
                    scale=scale
                )
            else:
                hrrgb, lrrgb, hrnir, lrnir = common.get_LRwrong(
                    hrrgb, hrnir,
                    scale=scale
                )

            hrrgb = hrrgb[:self.height1, :self.width1, :]  # [416, 576]
            lrrgb = lrrgb[:self.height1 // scale, :self.width1 // scale, :]  # [208, 288]
            hrnir = hrnir[:self.height1, :self.width1]
            lrnir = lrnir[:self.height1 // scale, :self.width1 // scale]
            
            hrnir = np.expand_dims(hrnir, -1)
            lrnir = np.expand_dims(lrnir, -1)
        return hrrgb, lrrgb, hrnir, lrnir
    
    def __getitem__(self, index):
        # print(index, ' index')
        collection, key = self.records[index]
        
        # print(str(Path(self.data_path) / collection / 'RGBCorrect' / (key + '.png')))
        rgb = cv2.imread(str(Path(self.data_path) / collection / 'RGBCorrect' / (key + '.png')))
        assert (rgb.shape == (self.height, self.width, 3))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32)
        
        if self.warpnir:
            # print(self.warpnir + collection + '/' + key + '.png')
            nir = cv2.imread(self.warpnir + collection + '/' + key + '.png', cv2.IMREAD_GRAYSCALE)
        else:
            nir = cv2.imread(str(Path(self.data_path) / collection / 'NIRCorrect' / (key + '.png')), cv2.IMREAD_GRAYSCALE)

        assert (nir.shape == (self.height, self.width))
        nir = nir.astype(np.float32)
        rgb, rgblr, nir, nirlr = self.get_patch(rgb, nir)
        rgb = rgb.transpose([2, 0, 1])
        rgblr = rgblr.transpose([2, 0, 1])
        nir = nir.transpose([2, 0, 1])  # [np.newaxis][:]
        nirlr = nirlr.transpose([2, 0, 1])  # [np.newaxis][:]
        
        rgb = np.ascontiguousarray(rgb).copy()
        rgblr = np.ascontiguousarray(rgblr).copy()
        nir = np.ascontiguousarray(nir).copy()
        nirlr = np.ascontiguousarray(nirlr).copy()
        rgb = torch.from_numpy(rgb).float()
        rgblr = torch.from_numpy(rgblr).float()
        nir = torch.from_numpy(nir).float()
        nirlr = torch.from_numpy(nirlr).float()
        
        return collection, key, rgb, nir, rgblr, nirlr
    
    def __len__(self):
        return self.n_records
        

class StereoDatasetTestWarpY(data.Dataset):
    # 有exposure
    def __init__(self, data_path, list_path, splits, istrain, warpnir=None, scale=4, testonly=False):
        super(StereoDatasetTestWarpY, self).__init__()
        self.testonly = testonly
        self.scale = [scale[0]]
        self.patch_size = 120
        self.train = istrain
        self.data_path = data_path
        self.height = 429
        self.width = 582-64
        self.im_suff = 'Resize'
        self.records = []
        for split in splits:
            f = open(Path(list_path) / (split + '.txt'), 'r')
            lines = f.readlines()
            f.close()
            if self.train:
                for i, line in enumerate(lines):
                    if '20170222_0951' in line or '20170222_1423' in line or '20170223_1639' in line:
                        if i % 5 == 0:
                            continue
                    splited = line.split()
                    collection = splited[0]
                    key = splited[1]
                    rgb_exp = float(splited[2])
                    nir_exp = float(splited[3])
                    record = (collection, key, rgb_exp, nir_exp)
                    self.records.append(record)
            else:
                for i, line in enumerate(lines):
                    if i % 5 == 0:
                        splited = line.split()
                        '''collection, imgID, RGBExposureTime, NIRExposureTime, RedGain, BlueGain.
                        （Expusre times are in microseconds. Red gain and blue gain are white balancing parameters）'''
                        collection = splited[0] + '100'
                        key = splited[1]
                        rgb_exp = float(splited[2])
                        nir_exp = float(splited[3])
                        record = (collection, key, rgb_exp, nir_exp)
                        self.records.append(record)
        self.records = sorted(self.records)
        self.n_records = len(self.records)
        self.warpnir = warpnir
    
    def get_patchbic(self, hrrgb, hrnir, ycbcr):  # (rgb, nir64, y, ycbcr)
        scale = self.scale[0]
        if self.train:
            hrrgb, bicrgb, lrrgb, hrnir, bicnir, lrnir, hrycbcr, bicycbcr, lrycbcr, hry, bicy, lry = \
                common.get_patchrgbbic64(
                hrrgb, hrnir, ycbcr,
                patch_size=self.patch_size,
                scale=scale
            )
            hry = np.expand_dims(hry, -1)
            bicy = np.expand_dims(bicy, -1)
            lry = np.expand_dims(lry, -1)
        else:
            hrrgb, bicrgb, lrrgb, hrnir, bicnir, lrnir, hrycbcr, bicycbcr, lrycbcr, hry, bicy, lry = \
                common.get_LRbic64(
                hrrgb, hrnir, ycbcr,
                scale=scale
            )
            hry = np.expand_dims(hry, -1)
            bicy = np.expand_dims(bicy, -1)
            lry = np.expand_dims(lry, -1)
        return hrrgb, bicrgb, lrrgb, hrnir, bicnir, lrnir, hrycbcr, hry, bicycbcr, bicy, lrycbcr, lry
    
    def __getitem__(self, index):
        collection, key, rgb_exp, nir_exp = self.records[index]
        
        rgb = cv2.imread(str(Path(self.data_path) / collection / ('RGB' + 'Correct') / (key + '.png')))[:, 64:, :]
        assert (rgb.shape == (self.height, self.width, 3))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV).astype(np.float32)
        # from PIL import Image; Image.fromarray(np.uint8(cv2.cvtColor(np.uint8(ycbcr), cv2.COLOR_YUV2RGB))).save('./result/traindata/HRycbcr.png')
        rgb = rgb.astype(np.float32)  # y = ycbcr[:, :, 0]  # y = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        nir = cv2.imread(str(Path(self.data_path) / collection / ('NIRCorrect') / (key + '.png')), cv2.IMREAD_GRAYSCALE)
        nir = np.expand_dims(nir, -1)
        nir64l = []
        for i in range(64):
            nir64l.append(nir[:, i:-(64 - i), :])
        nir64 = np.concatenate(nir64l, -1).astype(np.float32)  # [429,518,64]
        
        assert (nir64.shape == (self.height, self.width, 64))
        rgb, bicrgb, rgblr, nir, bicnir, nirlr, ycbcr, y, ycbcrbc, ybc, ycbcrlr, ylr \
            = self.get_patchbic(rgb, nir64, ycbcr)
       
        ycbcr, ycbcrbc, ycbcrlr = ycbcr, ycbcrbc[:, :, 1:], ycbcrlr[:, :, 1:]
        y, ybc, ylr = y.transpose([2, 0, 1]), ybc.transpose([2, 0, 1]), ylr.transpose([2, 0, 1])
        ycbcr, ycbcrbc, ycbcrlr = ycbcr.transpose([2, 0, 1]), ycbcrbc.transpose([2, 0, 1]), ycbcrlr.transpose([2, 0, 1])
        rgb, bicrgb, rgblr = rgb.transpose([2, 0, 1]), bicrgb.transpose([2, 0, 1]), rgblr.transpose([2, 0, 1])
        nir, bicnir, nirlr = nir.transpose([2, 0, 1]), bicnir.transpose([2, 0, 1]), nirlr.transpose([2, 0, 1])
        
        y, ybc, ylr, ycbcr, ycbcrbc, ycbcrlr, bicrgb, bicnir = np.ascontiguousarray(y).copy(), np.ascontiguousarray(ybc).copy(), \
                    np.ascontiguousarray(ylr).copy(), np.ascontiguousarray(ycbcr).copy(),\
                    np.ascontiguousarray(ycbcrbc).copy(), np.ascontiguousarray(ycbcrlr).copy(), \
                    np.ascontiguousarray(bicrgb).copy(), np.ascontiguousarray(bicnir).copy()
        rgb, rgblr, nir, nirlr = np.ascontiguousarray(rgb).copy(), np.ascontiguousarray(rgblr).copy(), np.ascontiguousarray(nir).copy(), np.ascontiguousarray(nirlr).copy()
        
        y, ybc, ylr, ycbcr, ycbcrbc, ycbcrlr, bicrgb, bicnir = torch.from_numpy(y).float(), torch.from_numpy(ybc).float() \
            , torch.from_numpy(ylr).float(), torch.from_numpy(ycbcr).float(), torch.from_numpy(ycbcrbc).float() \
            , torch.from_numpy(ycbcrlr).float(), torch.from_numpy(bicrgb).float(), torch.from_numpy(bicnir).float()
        rgb, rgblr, nir, nirlr = torch.from_numpy(rgb).float(), torch.from_numpy(rgblr).float(), torch.from_numpy(nir).float(), torch.from_numpy(nirlr).float()
        
        return collection, key, rgb, nir, rgblr, nirlr, bicrgb, bicnir, ycbcr, ycbcrbc, ycbcrlr, y, ybc, ylr  #, rgb_exp, nir_exp
    
    def __len__(self):
        if self.train:  # train dataset
            return self.n_records
        else:
            if self.testonly:
                return self.n_records  #
            else:
                return 10
    
    def fname(self, collection, key, suffix, camera, ftype):
        return str(Path(self.data_path) / collection / (camera + suffix) / (key + '_' + camera + suffix + '.' + ftype))


class StereoDataDepthRGB(data.Dataset):
    # 有exposure
    # TEST_SET_2005_2006 = ['Reindeer', 'Bowling1', 'Bowling2']
    # TEST_SET_2014 = ['Adirondack-perfect', 'Motorcycle-perfect']
    def __init__(self, data_path, disp_path, istrain, scale=4, testonly=False, patch=192):
        super(StereoDataDepthRGB, self).__init__()
        self.testonly = testonly
        self.scale = [scale[0]]
        self.patch_size = patch
        self.imp = data_path  # E:/file/python_project/StereoSR/StereoSR_CVPR20_code/Middlebury/train/fullpng/leftd/
        self.dp = disp_path  # E:/file/python_project/StereoSR/StereoSR_CVPR20_code/Middlebury/train/fullpng/displeftim/
        self.train = istrain
        # self.recordsim = ['F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/20170224_0742100/RGBCorrect/240742_057997.png']
        # self.recordsd = ['F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/20170224_0742100/NIRCorrect/240742_057997.png']
        self.recordsim = sorted(glob.glob(self.imp + '/*'))  #
        self.recordsd = sorted(glob.glob(self.dp + '/*'))
        self.n_records = len(self.recordsd)
    
    def get_patch(self, hrrgb, hrnir):
        scale = self.scale[0]
        if self.train:
            hrrgb, lrrgb, hrnir, lrnir = common.get_patchrgb(
                hrrgb, hrnir,
                patch_size=self.patch_size,
                scale=scale
            )
            hrnir = np.expand_dims(hrnir, -1)
            lrnir = np.expand_dims(lrnir, -1)
            hrrgb, lrrgb, hrnir, lrnir = common.augment(hrrgb, lrrgb, hrnir, lrnir)
        else:
            hrrgb, lrrgb, hrnir, lrnir = common.get_LR(
                hrrgb, hrnir,
                scale=scale
            )
            hrnir = np.expand_dims(hrnir, -1)
            lrnir = np.expand_dims(lrnir, -1)
        return hrrgb, lrrgb, hrnir, lrnir
    
    def __getitem__(self, index):
        keyd = self.recordsd[index]
        nir = cv2.imread(keyd, cv2.IMREAD_GRAYSCALE)
        name = keyd[len(self.dp):-9]
        
        key = self.imp + name + '.png'
        # key = self.recordsim[index]
        if os.path.isfile(key):
            rgb = cv2.imread(key)
        elif os.path.isfile(self.imp + name + '.jpg'):
            rgb = cv2.imread(self.imp + name + '.jpg')
        elif os.path.isfile(self.imp + name + '.bmp'):
            rgb = cv2.imread(self.imp + name + '.bmp')
        else:
            exit()

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32)
        ih, iw = rgb.shape[:2]
        # print(ih, iw, name)
        # ih2, iw2 = nir.shape[:2]
        ih = ih // (16 * self.scale[0]) * (16 * self.scale[0])
        iw = iw // (16 * self.scale[0]) * (16 * self.scale[0])
        rgb = rgb[:ih, :iw, :]
        nir = nir[:ih, :iw]
        
        nir = nir.astype(np.float32)
        rgb, rgblr, nir, nirlr = self.get_patch(rgb, nir)
        rgb = rgb.transpose([2, 0, 1])
        rgblr = rgblr.transpose([2, 0, 1])
        nir = nir.transpose([2, 0, 1])  # [np.newaxis][:]
        nirlr = nirlr.transpose([2, 0, 1])  # [np.newaxis][:]
        
        rgb = np.ascontiguousarray(rgb).copy()
        rgblr = np.ascontiguousarray(rgblr).copy()
        nir = np.ascontiguousarray(nir).copy()
        nirlr = np.ascontiguousarray(nirlr).copy()
        rgb = torch.from_numpy(rgb).float()
        rgblr = torch.from_numpy(rgblr).float()
        nir = torch.from_numpy(nir).float()
        nirlr = torch.from_numpy(nirlr).float()
        
        return key[:len(self.imp)],  key[len(self.imp):], rgb, nir, rgblr, nirlr
    
    def __len__(self):
        if self.train:
            return self.n_records
        else:
            if self.testonly:
                return self.n_records  # 0
            else:
                return 5
