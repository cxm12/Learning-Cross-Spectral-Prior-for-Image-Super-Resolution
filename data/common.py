import random

import numpy as np
import skimage.color as sc

import torch
import cv2
from PIL import Image


def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]
    
    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size
    
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    
    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy
    
    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]
    
    return ret


def get_patchrgbnir(hrrgb, hrnir, patch_size=96, scale=2):
    ih, iw = hrrgb.shape[:2]
    tp = patch_size
    ip = patch_size
    
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    size = tuple(((np.array(Image.fromarray(hrrgb).size)).astype(int) / scale).astype(int))
    lrrgb = np.array(Image.fromarray(hrrgb).resize(size, Image.BICUBIC))
    lrnir = np.array(Image.fromarray(hrnir).resize(size, Image.BICUBIC))

    tx, ty = scale * ix, scale * iy

    ret = [
        hrrgb[iy:iy + ip, ix:ix + ip, :],
        lrrgb[ty:ty + tp, tx:tx + tp, :],
        hrnir[iy:iy + ip, ix:ix + ip],
        lrnir[ty:ty + tp, tx:tx + tp]]
    
    return ret


def get_patchrgb(hrrgb, hrnir, scale, patch_size=96):
    ih, iw = hrrgb.shape[:2]
    hrrgb = hrrgb[:ih//scale*scale, :iw//scale*scale, :]
    # ih2, iw2 = hrnir.shape[:2]
    # if ih != ih2 or iw != iw2:
    #     print(ih, iw, ih2, iw2)
    #     exit()
    lp = patch_size // scale
    ip = patch_size
    
    lx = random.randrange(0, iw // scale - lp + 1)
    ly = random.randrange(0, ih // scale - lp + 1)
    ix, iy = scale * lx, scale * ly
    
    size1 = tuple(((np.array(Image.fromarray(np.uint8(hrrgb)).size)).astype(int) // scale).astype(int))
    lrrgb = np.array(Image.fromarray(np.uint8(hrrgb)).resize(size1, Image.BICUBIC))
    
    size = tuple(((np.array(Image.fromarray(np.uint8(hrnir)).size)).astype(int) // scale).astype(int))
    lrnir = np.array(Image.fromarray(np.uint8(hrnir)).resize(size, Image.BICUBIC))
    
    ret = [hrrgb[iy:iy + ip, ix:ix + ip, :],
           lrrgb[ly:ly + lp, lx:lx + lp, :],
           hrnir[iy:iy + ip, ix:ix + ip],
           lrnir[ly:ly + lp, lx:lx + lp]]
    
    return ret


def get_LRrgb(hrrgb, hrnir, scale):
    size1 = tuple(((np.array(Image.fromarray(np.uint8(hrrgb)).size)).astype(int) // scale).astype(int))
    lrrgb = np.array(Image.fromarray(np.uint8(hrrgb)).resize(size1, Image.BICUBIC))
    
    size = tuple(((np.array(Image.fromarray(np.uint8(hrnir)).size)).astype(int) // scale).astype(int))
    lrnir = np.array(Image.fromarray(np.uint8(hrnir)).resize(size, Image.BICUBIC))
    
    ret = [hrrgb, lrrgb, hrnir, lrnir]
    return ret


def get_patchrgbbic64(hrrgb, hrnir, ycbcr, scale, patch_size=96):
    ih, iw = hrrgb.shape[:2]
    lp = patch_size // scale
    ip = patch_size
    
    lx = random.randrange(0, iw // scale - lp + 1)
    ly = random.randrange(0, ih // scale - lp + 1)
    ix, iy = scale * lx, scale * ly
    
    size1 = tuple(((np.array(Image.fromarray(np.uint8(hrrgb)).size)).astype(int) / scale).astype(int))
    size2 = tuple((np.array(Image.fromarray(np.uint8(hrrgb)).size)).astype(int))
    lrrgb = np.array(Image.fromarray(np.uint8(hrrgb)).resize(size1, Image.BICUBIC))
    bicrgb = np.array(Image.fromarray(np.uint8(lrrgb)).resize(size2, Image.BICUBIC))
    
    size = tuple(((np.array(Image.fromarray(np.uint8(hrnir[:, :, 0])).size)).astype(int) / scale).astype(int))
    size02 = tuple((np.array(Image.fromarray(np.uint8(hrnir[:, :, 0])).size)).astype(int))
    lrnir64l = []
    bcnir64l = []
    for i in range(64):
        hnir = hrnir[:, :, i]
        lnir = Image.fromarray(np.uint8(hnir)).resize(size, Image.BICUBIC)
        lrnir64l.append(np.expand_dims(np.array(lnir), -1))
        bcnir64l.append(np.expand_dims(np.array(lnir.resize(size02, Image.BICUBIC)), -1))
    lrnir = np.concatenate(lrnir64l, -1)
    bicnir = np.concatenate(bcnir64l, -1)

    ycbcr = cv2.cvtColor(np.uint8(ycbcr), cv2.COLOR_YUV2RGB)
    lrycbcr = np.array(Image.fromarray(ycbcr).resize(size1, Image.BICUBIC))
    bicycbcr = np.array(Image.fromarray(lrycbcr).resize(size2, Image.BICUBIC))
    lrycbcr = cv2.cvtColor(lrycbcr, cv2.COLOR_RGB2YUV).astype(np.float32)
    bicycbcr = cv2.cvtColor(bicycbcr, cv2.COLOR_RGB2YUV).astype(np.float32)
    ycbcr = cv2.cvtColor(np.uint8(ycbcr), cv2.COLOR_RGB2YUV).astype(np.float32)
    
    lry = lrycbcr[:,:,0]
    bicy = bicycbcr[:,:,0]
    y = ycbcr[:,:,0]
    
    ret = [hrrgb[iy:iy + ip, ix:ix + ip, :], bicrgb[iy:iy + ip, ix:ix + ip, :],
           lrrgb[ly:ly + lp, lx:lx + lp, :],
           hrnir[iy:iy + ip, ix:ix + ip, :], bicnir[iy:iy + ip, ix:ix + ip, :],
           lrnir[ly:ly + lp, lx:lx + lp, :],
           ycbcr[iy:iy + ip, ix:ix + ip, :], bicycbcr[iy:iy + ip, ix:ix + ip, :],
           lrycbcr[ly:ly + lp, lx:lx + lp, :],
           y[iy:iy + ip, ix:ix + ip], bicy[iy:iy + ip, ix:ix + ip],
           lry[ly:ly + lp, lx:lx + lp]]
    
    return ret


def get_LRwrong(hrrgb, hrnir, scale):
    ih, iw = hrrgb.shape[:2]
    
    size1 = tuple(((np.array(Image.fromarray(np.uint8(hrrgb)).size)).astype(int) / scale).astype(int))
    lrrgb = np.array(Image.fromarray(np.uint8(hrrgb)).resize(size1, Image.BICUBIC))
    
    size = tuple(((np.array(Image.fromarray(np.uint8(hrnir)).size)).astype(int) / scale).astype(int))
    lrnir = np.array(Image.fromarray(np.uint8(hrnir)).resize(size, Image.BICUBIC))
    
    ihl, iwl = lrrgb.shape[:2]
    
    ret = [hrrgb[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2), :],
           lrrgb[:ihl // 2 * 2, :iwl // 2 * 2, :],
           hrnir[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2)],
           lrnir[:ihl // 2 * 2, :iwl // 2 * 2]]
    
    return ret


def get_LR(hrrgb, hrnir, scale):
    ih, iw = hrrgb.shape[:2]
    ih = ih - ih % scale
    iw = iw - iw % scale
    hrrgb = hrrgb[:ih, :iw, :]
    
    size1 = tuple(((np.array(Image.fromarray(np.uint8(hrrgb)).size)).astype(int) / scale).astype(int))
    lrrgb = np.array(Image.fromarray(np.uint8(hrrgb)).resize(size1, Image.BICUBIC))
    
    size = tuple(((np.array(Image.fromarray(np.uint8(hrnir)).size)).astype(int) / scale).astype(int))
    lrnir = np.array(Image.fromarray(np.uint8(hrnir)).resize(size, Image.BICUBIC))
    
    ihl, iwl = lrrgb.shape[:2]
    
    ret = [hrrgb[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2), :],
           lrrgb[:ihl // 2 * 2, :iwl // 2 * 2, :],
           hrnir[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2)],
           lrnir[:ihl // 2 * 2, :iwl // 2 * 2]]
    
    return ret


def get_LRbic(hrrgb, hrnir, scale):
    ih, iw = hrrgb.shape[:2]
    
    size1 = tuple(((np.array(Image.fromarray(np.uint8(hrrgb)).size)).astype(int) / scale).astype(int))
    size2 = tuple((np.array(Image.fromarray(np.uint8(hrrgb)).size)).astype(int))
    lrrgb = np.array(Image.fromarray(np.uint8(hrrgb)).resize(size1, Image.BICUBIC))
    bicrgb = np.array(Image.fromarray(np.uint8(lrrgb)).resize(size2, Image.BICUBIC))
    
    size = tuple(((np.array(Image.fromarray(np.uint8(hrnir)).size)).astype(int) / scale).astype(int))
    size02 = tuple((np.array(Image.fromarray(np.uint8(hrnir)).size)).astype(int))
    lrnir = np.array(Image.fromarray(np.uint8(hrnir)).resize(size, Image.BICUBIC))
    bicnir = np.array(Image.fromarray(np.uint8(lrnir)).resize(size02, Image.BICUBIC))
    
    ihl, iwl = lrrgb.shape[:2]
    
    ret = [hrrgb[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2), :],
           bicrgb[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2), :],
           lrrgb[:ihl // 2 * 2, :iwl // 2 * 2, :],
           hrnir[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2)],
           bicnir[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2)],
           lrnir[:ihl // 2 * 2, :iwl // 2 * 2]]
    
    return ret


def get_LRbic64(hrrgb, hrnir, ycbcr, scale):  # (rgb, nir64, y, ycbcr)
    ih, iw = hrrgb.shape[:2]

    size1 = tuple(((np.array(Image.fromarray(np.uint8(hrrgb)).size)).astype(int) / scale).astype(int))
    size2 = tuple((np.array(Image.fromarray(np.uint8(hrrgb)).size)).astype(int))
    lrrgb = np.array(Image.fromarray(np.uint8(hrrgb)).resize(size1, Image.BICUBIC))
    bicrgb = np.array(Image.fromarray(np.uint8(lrrgb)).resize(size2, Image.BICUBIC))

    size = tuple(((np.array(Image.fromarray(np.uint8(hrnir[:, :, 0])).size)).astype(int) / scale).astype(int))
    size02 = tuple((np.array(Image.fromarray(np.uint8(hrnir[:, :, 0])).size)).astype(int))
    lrnir64l = []
    bcnir64l = []
    for i in range(64):
        hnir = hrnir[:, :, i]
        lnir = Image.fromarray(np.uint8(hnir)).resize(size, Image.BICUBIC)
        lrnir64l.append(np.expand_dims(np.array(lnir), -1))
        bcnir64l.append(np.expand_dims(np.array(lnir.resize(size02, Image.BICUBIC)), -1))
    lrnir = np.concatenate(lrnir64l, -1)
    bicnir = np.concatenate(bcnir64l, -1)
    ihl, iwl = lrrgb.shape[:2]

    ycbcr = cv2.cvtColor(np.uint8(ycbcr), cv2.COLOR_YUV2RGB)  # Image.fromarray(np.uint8(ycbcr)).save('./result/traindata/HRycbcr.png')
    lrycbcr = np.array(Image.fromarray(ycbcr).resize(size1, Image.BICUBIC))
    bicycbcr = np.array(Image.fromarray(lrycbcr).resize(size2, Image.BICUBIC))
    lrycbcr = cv2.cvtColor(lrycbcr, cv2.COLOR_RGB2YUV).astype(np.float32)
    bicycbcr = cv2.cvtColor(bicycbcr, cv2.COLOR_RGB2YUV).astype(np.float32)
    ycbcr = cv2.cvtColor(np.uint8(ycbcr), cv2.COLOR_RGB2YUV).astype(np.float32)
    # Image.fromarray(np.uint8(cv2.cvtColor(np.uint8(ycbcr), cv2.COLOR_YUV2RGB))).save('./result/traindata/HRycbcr.png')

    lry = lrycbcr[:, :, 0]
    bicy = bicycbcr[:, :, 0]
    y = ycbcr[:, :, 0]
    
    ret = [hrrgb[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2), :],
           bicrgb[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2), :],
           lrrgb[:ihl // 2 * 2, :iwl // 2 * 2, :],
           hrnir[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2), :],
           bicnir[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2), :],
           lrnir[:ihl // 2 * 2, :iwl // 2 * 2, :],
          ycbcr[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2), :],
            bicycbcr[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2), :],
            lrycbcr[:ihl // 2 * 2, :iwl // 2 * 2, :],
            y[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2)],
            bicy[:ih // (scale * 2) * (scale * 2), :iw // (scale * 2) * (scale * 2)],
            lry[:ihl // 2 * 2, :iwl // 2 * 2]]
    return ret


def get_patchrgbCV18(hrrgb, hrnir, scale, patch_size=96):
    ih, iw = hrrgb.shape[:2]
    lp = patch_size // scale
    ip = patch_size
    
    lx = random.randrange(0, iw // scale - lp + 1)
    ly = random.randrange(0, ih // scale - lp + 1)
    ix, iy = scale * lx, scale * ly
    
    lrrgb = cv2.resize(hrrgb, (iw // scale, ih // scale), interpolation=cv2.INTER_CUBIC)
    lrnir = cv2.resize(hrnir, (iw // scale, ih // scale), interpolation=cv2.INTER_CUBIC)
    
    ret = [hrrgb[iy:iy + ip, ix:ix + ip, :],
           lrrgb[ly:ly + lp, lx:lx + lp, :],
           hrnir[iy:iy + ip, ix:ix + ip],
           lrnir[ly:ly + lp, lx:lx + lp]]
    
    return ret


def get_LRCV18(hrrgb, hrnir, scale):
    scale = scale
    
    ih, iw = hrrgb.shape[:2]
    
    lrrgb = cv2.resize(hrrgb, (iw // scale, ih // scale), interpolation=cv2.INTER_CUBIC)
    lrnir = cv2.resize(hrnir, (iw // scale, ih // scale), interpolation=cv2.INTER_CUBIC)
    ihl, iwl = lrrgb.shape[:2]

    ret = [hrrgb[:ih//(scale*2)*(scale*2), :iw//(scale*2)*(scale*2), :], lrrgb[:ihl//2*2, :iwl//2*2, :],
           hrnir[:ih//(scale*2)*(scale*2), :iw//(scale*2)*(scale*2)], lrnir[:ihl//2*2, :iwl//2*2]]
    
    return ret


# import skimage
# skimage.transform.resize()

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        
        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)
        
        return img  # uint8 [0, 255]
    
    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)), dtype=np.float32)
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)
        
        return tensor  # float32 [0, 1]
    
    return [_np2Tensor(a) for a in args]


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    
    def _augment(img):
        if len(img.shape)>=3:
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)
        else:
            if hflip: img = img[:, ::-1]
            if vflip: img = img[::-1, :]
            if rot90: img = img.transpose(1, 0)
            
        return img
    
    return [_augment(a) for a in args]
