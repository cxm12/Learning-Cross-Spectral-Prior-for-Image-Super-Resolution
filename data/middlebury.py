# Original Code from: https://github.com/prs-eth/graph-super-resolution
from pathlib import Path
from torchvision.transforms import Normalize

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode
from PIL import Image
# Original Code from: https://github.com/prs-eth/graph-super-resolution
import re
import csv
import random
import warnings
import numpy as np
import torch
from torchvision.transforms import RandomCrop, RandomRotation
import torchvision.transforms.functional as F
from skimage.measure import block_reduce
from scipy import interpolate

ROTATION_EXPAND = False
ROTATION_CENTER = None  # image center
ROTATION_FILL = 0.


def downsample(image, scaling_factor):
    """
    Performs average pooling, ignoring nan values
    :param image: torch tensor or numpy ndarray of shape (B, C, H, W)
    """
    if image.ndim != 4:
        raise ValueError(f'Image should have four dimensions, got {image.ndim}')

    is_tensor = torch.is_tensor(image)
    if is_tensor:
        device = image.device
        image = image.detach().cpu().numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        image = block_reduce(image, (1, 1, scaling_factor, scaling_factor), np.nanmean)

    return torch.from_numpy(image).to(device) if is_tensor else image


def bicubic_with_mask(source, mask, scaling_factor):
    source_size = source.shape[0]
    H, W = source.shape[0] * scaling_factor, source.shape[1] * scaling_factor

    source_r = source.flatten()
    mask_r = mask.flatten()

    x = np.arange((scaling_factor - 1) / 2, W, scaling_factor)  # (0,source.shape[0])
    y = np.arange((scaling_factor - 1) / 2, H, scaling_factor)  # (0,source.shape[1])

    x_g, y_g = np.meshgrid(x, y)  # indexing="ij"
    x_g_r = x_g.flatten()
    y_g_r = y_g.flatten()

    source_r = source_r[mask_r == 1]
    x_g_r = x_g_r[mask_r == 1]
    y_g_r = y_g_r[mask_r == 1]
    xy_g_r = np.concatenate([x_g_r[:, None], y_g_r[:, None]], axis=1)

    x_HR = np.linspace(0, W, endpoint=False, num=W)
    y_HR = np.linspace(0, H, endpoint=False, num=H)

    x_HR_g, y_HR_g = np.meshgrid(x_HR, y_HR)
    x_HR_g, y_HR_g = x_HR_g.flatten(), y_HR_g.flatten()
    xy_HR_g_r = np.concatenate([x_HR_g[:, None], y_HR_g[:, None]], axis=1)

    depth_HR = interpolate.griddata(xy_g_r, source_r, xy_HR_g_r, method="cubic")
    depth_HR_nearest = interpolate.griddata(xy_g_r, source_r, xy_HR_g_r, method="nearest")
    depth_HR[np.isnan(depth_HR)] = depth_HR_nearest[np.isnan(depth_HR)]
    depth_HR = depth_HR.reshape(source_size * scaling_factor, -1)

    return depth_HR


def random_horizontal_flip(images, p=0.5):
    if random.random() < p:
        return [image.flip(-1) for image in images]
    return images


def random_rotate(images, max_rotation_angle, interpolation, crop_valid=False):
    angle = RandomRotation.get_params([-max_rotation_angle, max_rotation_angle])
    if crop_valid:
        rotated = [F.rotate(image, angle, interpolation, True, ROTATION_CENTER, ROTATION_FILL) for image in images]
        crop_params = np.floor(np.asarray(rotated[0].shape[1:3]) - 2. *
                      (np.sin(np.abs(angle * np.pi / 180.)) * np.asarray(images[0].shape[1:3][::-1]))).astype(int)
        return [F.center_crop(image, crop_params) for image in rotated]
    else:
        return [F.rotate(image, angle, interpolation, ROTATION_EXPAND, ROTATION_CENTER, ROTATION_FILL) for image in images]


def random_crop(images, crop_size):
    crop_params = RandomCrop.get_params(images[0], crop_size)
    return [F.crop(image, *crop_params) for image in images]


# Following contents were adapted from https://www.programmersought.com/article/2506939342/.
def _read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode('utf-8').rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(pfm_file.readline().decode('utf-8').rstrip())
        if scale < 0:
            endian = '<'  # little endian
        else:
            endian = '>'  # big endian

        disparity = np.fromfile(pfm_file, endian + 'f')

    return disparity, (height, width, channels)


def read_calibration(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)

    return calib


def create_depth_from_pfm(pfm_file_path, calib=None):
    disparity, shape = _read_pfm(pfm_file_path)

    if calib is None:
        raise Exception('No calibration information available')
    else:
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))
        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])

        depth_map = fx * base_line / (disparity + doffs)
        depth_map = np.reshape(depth_map, newshape=shape)
        depth_map = np.flipud(depth_map).transpose((2, 0, 1)).copy()

        depth_map[depth_map == 0.] = np.nan

        return depth_map


class MiddleburyDataset(Dataset):

    def __init__(
            self,
            data_path,
            split='train',
            crop_size=(128, 128),
            do_horizontal_flip=True,
            max_rotation_angle=15,
            scale_interpolation=InterpolationMode.BILINEAR,
            rotation_interpolation=InterpolationMode.BILINEAR,
            image_transform=None,
            depth_transform=Normalize([0.0], [1122.7]),
            use_ambient_images=False,
            crop_deterministic=False,
            scaling=8
    ):
        if max_rotation_angle > 0 and crop_deterministic:
            max_rotation_angle = 0
            print('Set max_rotation_angle to zero because of deterministic cropping')

        self.split = split
        self.crop_size = crop_size
        self.do_horizontal_flip = do_horizontal_flip
        self.max_rotation_angle = max_rotation_angle
        self.rotation_interpolation = rotation_interpolation
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.data = []
        self.crop_deterministic = crop_deterministic
        self.scaling = scaling

        # read in various Middlebury datasets using the respective global load_{name} function
        data_dir = Path(data_path)
        self.deterministic_map = []
        if split == 'test':  # , '2014', '2006'
            for name in ('2005',):
                self.data.extend(
                    globals()[f'load_{name}'](data_dir / name, 1.0, InterpolationMode.BILINEAR, use_ambient_images, split))
                print(len(self.data))
        else:  # '2001', '2003',  # '2021', '2005', '2014',
            for name in ('2006',):
                self.data.extend(
                    globals()[f'load_{name}'](data_dir / name, 1.0, InterpolationMode.BILINEAR, use_ambient_images, split))
                print(len(self.data))

        if self.crop_deterministic:
            assert not use_ambient_images
            # construct deterministic mapping
            for i, datum in enumerate(self.data):
                H, W = datum[0][0].shape[1:]
                num_crops_h, num_crops_w = H // self.crop_size[0], W // self.crop_size[1]
                if split == 'test':
                    num_crops_h, num_crops_w = 1, 1
                self.deterministic_map.extend(((i, j, k) for j in range(num_crops_h) for k in range(num_crops_w)))

    def __getitem__(self, index, called=0):
        if (self.crop_deterministic):
            im_index, crop_index_h, crop_index_w = self.deterministic_map[index]
        else:
            im_index = index

        image, depth_map = random.choice(self.data[im_index])
        
        image, depth_map = image.clone(), depth_map.clone()
        if self.do_horizontal_flip and not self.crop_deterministic:
            image, depth_map = random_horizontal_flip((image, depth_map))

        if self.max_rotation_angle > 0 and not self.crop_deterministic:
            image, depth_map = random_rotate((image, depth_map), self.max_rotation_angle, self.rotation_interpolation)
            # passing fill=np.nan to rotate sets all pixels to nan, so set it here explicitly
            depth_map[depth_map == 0.] = np.nan

        if self.crop_deterministic:
            if self.split != 'test':
                slice_h = slice(crop_index_h * self.crop_size[0], (crop_index_h + 1) * self.crop_size[0])
                slice_w = slice(crop_index_w * self.crop_size[1], (crop_index_w + 1) * self.crop_size[1])
            else:
                c, h, w = image.shape
                h = h // (16 * self.scaling) * (16 * self.scaling)
                w = w // (16 * self.scaling) * (16 * self.scaling)
                slice_h = slice(0, h)
                slice_w = slice(0, w)
            image, depth_map = image[:, slice_h, slice_w], depth_map[:, slice_h, slice_w]
        else:
            image, depth_map = random_crop((image, depth_map), self.crop_size)
        
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.depth_transform is not None:
            depth_map = self.depth_transform(depth_map)

        depth_maplr = downsample(depth_map.unsqueeze(0), self.scaling).squeeze().unsqueeze(0)
        rgblr = downsample(image.unsqueeze(0), self.scaling).squeeze().unsqueeze(0)

        mask_hr = (~torch.isnan(depth_map)).float()
        mask_lr = (~torch.isnan(depth_maplr)).float()

        depth_map[mask_hr == 0.] = 0.
        depth_maplr[mask_lr == 0.] = 0.

        if self.split == 'train' and (torch.mean(mask_lr) < 0.9 or torch.mean(mask_hr) < 0.8):
            # omit patch due to too many depth holes, try another one
            return self.__getitem__(index, called=called + 1)
        else:
            depth_map_bicubic = torch.from_numpy(bicubic_with_mask(
                depth_maplr.squeeze().numpy(), mask_lr.squeeze().numpy(), self.scaling)).float()
            if self.split != 'test':
                depth_map_bicubic = depth_map_bicubic.reshape((1, self.crop_size[0], self.crop_size[1]))
            else:
                depth_map_bicubic = depth_map_bicubic.reshape((1, h, w))
            return image, depth_map, depth_maplr, rgblr, mask_hr, mask_lr, im_index, depth_map_bicubic
            # return {'guide': image, 'y': depth_map, 'source': source, 'mask_hr': mask_hr, 'mask_lr': mask_lr,
            #                     'im_idx': im_index, 'y_bicubic': y_bicubic}
        
    def __len__(self):
        return len(self.deterministic_map if self.crop_deterministic else self.data)


# VAL_SET_2005_2006 = ['Moebius', 'Lampshade1', 'Lampshade2']
# VAL_SET_2014 = ['Shelves-perfect', 'Playtable-perfect']
TEST_SET_2005_2006 = ['Reindeer-2views', 'Bowling1-2views', 'Bowling2-2views']
TEST_SET_2014 = ['Adirondack-perfect', 'Motorcycle-perfect']

import cv2


def load_2021(data_dir: Path, scale, scale_interpolation, use_ambient_images, split):
    data = []
    for scene in sorted(data_dir.iterdir()):
        # make train val test split
        calibration = read_calibration(scene / 'calib.txt')

        # add left and right view, as well as corresponding depth maps
        for view in (0, 1):
            resize = Resize((int(int(calibration['height']) * scale), int(int(calibration['width']) * scale)), scale_interpolation)
            depth_map = resize(torch.from_numpy(create_depth_from_pfm(scene / f'disp{view}.pfm', calibration)))
            transform = Compose([ToTensor(), resize])
            if use_ambient_images:
                data.append(
                    [(transform(Image.open(path)), depth_map) for path in scene.glob(f'ambient/L*/im{view}*.png')])
            else:
                data.append([(transform(Image.open(scene / f'im{view}.png')), depth_map)])
        
    return data


def load_2001(data_dir: Path, scale, scale_interpolation, use_ambient_images, split):
    data = []
    for scene in sorted(data_dir.iterdir()):
        # add left and right view, as well as corresponding depth maps
        for view in (2, 6):
            depth_map = cv2.imread(str(scene / f'disp{view}.jpg'), cv2.IMREAD_GRAYSCALE)
            depth_map = torch.from_numpy(depth_map)
            transform = Compose([ToTensor()])
            data.append([(transform(Image.open(str(scene / f'im{view}.jpg'))), depth_map)])
    
    return data


def load_2003(data_dir: Path, scale, scale_interpolation, use_ambient_images, split):
    data = []
    for scene in sorted(data_dir.iterdir()):
        # add left and right view, as well as corresponding depth maps
        for view in (2, 6):
            depth_map = cv2.imread(str(scene / f'disp{view}.png'), cv2.IMREAD_GRAYSCALE)
            depth_map = torch.from_numpy(depth_map)
            transform = Compose([ToTensor()])
            data.append([(transform(Image.open(str(scene / f'im{view}.png'))), depth_map)])
    
    return data


def load_2014(data_dir: Path, scale, scale_interpolation, use_ambient_images, split):
    data = []
    for scene in sorted(data_dir.iterdir()):
        # ignore scenes with imperfect rectification, these are only included in the 2014 dataset anyway
        if not scene.is_dir() or scene.name.endswith('-imperfect'):
            continue

        # make train val test split
        last_dir = scene.parts[-1]

        if (split == 'test' and last_dir in TEST_SET_2014) or \
                (split == 'train' and (last_dir not in TEST_SET_2014)):
            calibration = read_calibration(scene / 'calib.txt')

            # add left and right view, as well as corresponding depth maps
            for view in (0, 1):
                resize = Resize((int(int(calibration['height']) * scale), int(int(calibration['width']) * scale)),
                                scale_interpolation)
                depth_map = resize(torch.from_numpy(create_depth_from_pfm(scene / f'disp{view}.pfm', calibration)))
                transform = Compose([ToTensor(), resize])
                if use_ambient_images:
                    data.append(
                        [(transform(Image.open(path)), depth_map) for path in scene.glob(f'ambient/L*/im{view}*.png')])
                else:
                    data.append([(transform(Image.open(scene / f'im{view}.png')), depth_map)])

    return data


def load_2006(data_dir: Path, scale, scale_interpolation, use_ambient_images, split):
    f = 3740  #px
    baseline = 160  #mm

    data = []
    for scene in sorted(data_dir.iterdir()):
        if not scene.is_dir():
            continue

        # make train val test split
        last_dir = scene.parts[-1]
        if (split == 'test' and last_dir in TEST_SET_2005_2006) or (
                split == 'train' and (last_dir not in TEST_SET_2005_2006)):
            scene = scene / last_dir[:-7]
            # add left and right view, as well as corresponding depth maps
            for view in (1, 5):
                disparities = torch.from_numpy(np.array(Image.open(scene / f'disp{view}.png'))).float().unsqueeze(0)
                # zero disparities are to be interpreted as inf, set them to nan so they result in nan depth
                disparities[disparities == 0.] = np.nan
                with open(scene / 'dmin.txt') as fh:
                    dmin = int(fh.read().strip())
                # add dmin to disparities because disparity maps and images have been cropped to the joint field of view
                disparities += dmin

                depth_map = baseline * f / disparities
                resize = Resize((int(depth_map.shape[1] * scale), int(depth_map.shape[2] * scale)), scale_interpolation)
                depth_map = resize(depth_map)
                transform = Compose([ToTensor(), resize])
                if use_ambient_images:
                    data.append(
                        [(transform(Image.open(path)), depth_map) for path in
                         scene.glob(f'Illum*/Exp*/view{view}.png')])
                else:
                    data.append([(transform(Image.open(scene / f'view{view}.png')), depth_map)])

    return data


# 2005 dataset same as 2006
def load_2005(*args, **kwargs):
    return load_2006(*args, **kwargs)



def mse_loss_func(pred, gt, mask):
    return torch.nn.MSELoss()(pred[mask == 1.], gt[mask == 1.])  # F.mse_loss(pred[mask == 1.], gt[mask == 1.])


def l1_loss_func(pred, gt, mask):
    return torch.nn.L1Loss()(pred[mask == 1.], gt[mask == 1.])  # F.l1_loss(pred[mask == 1.], gt[mask == 1.])

