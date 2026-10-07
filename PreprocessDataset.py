import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import lmdb
from os import path as osp
import os
import sys
from tensorflow import __version__ as _tf_version
IS_TF_1 = _tf_version.startswith('1.')
if IS_TF_1:
    # rootpath = 'F:/SRdata/train_data/NIR_VIS/RANUS/Test/'  # Train/'  #
    rootpath = 'F:/SRdata/train_data/NIR_VIS/nirscene1/'  #
    root = os.path.dirname(__file__)
else:
    root = '../..'
    rootpath = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/RANUS/Train/'  #  '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/nirscene1/'
print('root', root)


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def mainsubimages():
    """A multi-thread tool to crop large images to sub-images for faster IO.

    It is used for DIV2K dataset.

    opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9.
            A higher value means a smaller size and longer compression time.
            Use 0 for faster CPU decompression. Default: 3, same in cv2.

        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR
            DIV2K_train_LR_bicubic/X2
            DIV2K_train_LR_bicubic/X3
            DIV2K_train_LR_bicubic/X4
        After process, each sub_folder should have the same number of
        subimages.
        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = 1  # 20
    opt['compression_level'] = 3
    HR = 160
    HRstep = 120
    opt['thresh_size'] = 0

    # # HR images
    opt['input_folder'] = rootpath + '/NIR/'  # Train/'  #
    opt['save_folder'] = rootpath + '/NIR_sub/'  #
    opt['crop_size'] = HR
    opt['step'] = HRstep
    extract_subimages(opt)
    print('extract subimages from %s' % opt['input_folder'])

    # # LRx4 images2, 3,
    # for scale in [4]:
    #     opt['input_folder'] = rootpath + '/TrainLR%d/' % scale
    #     opt['save_folder'] = rootpath + '/TrainLR%d_sub/' % scale
    #     opt['crop_size'] = HR // scale
    #     opt['step'] = HRstep // scale
    #     extract_subimages(opt)
    #     print('extract subimages from %s' % opt['input_folder'])


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    os.makedirs(save_folder, exist_ok=True)
    print(f'mkdir {save_folder} ...')
    dir = os.listdir(input_folder)
    for subdir in dir:
        img_list = list(scandir(input_folder+subdir, full_path=True))
        print(img_list)
        save_folder = os.path.join(opt['save_folder'], subdir)
        os.makedirs(save_folder, exist_ok=True)

        pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
        pool = Pool(opt['n_thread'])
        for path in img_list:
            pool.apply_async(worker, args=(path, opt, subdir), callback=lambda arg: pbar.update(1))
        pool.close()
        pool.join()
        pbar.close()
        print('All processes done.')


def worker(path, opt, subdir=''):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    # img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, c = img.shape
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')

    # if h < 480 or w < 480:
    #     crop_size = 300
    #     step = 150

    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], subdir, f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


from PIL import Image


def generatecleanLR():
    input_folder = rootpath + 'Test/'  # + 'RGB/'  # + 'Train_sub/'  #
    dir = os.listdir(input_folder)
    slst = [2, 3, 4]  # [4]  #,
    print(dir)
    for scale in slst:
        savepath = rootpath + 'Tests%d/' % scale  # + 'RGBs%d/' % scale  # + 'Trains%d_sub/' % scale  #
        os.makedirs(savepath, exist_ok=True)
        for nam in dir:
            hrpath = input_folder + nam
            savepath1 = os.path.join(savepath, nam)
            os.makedirs(savepath1, exist_ok=True)
            print('savepath1', savepath1)
            filenames = list(scandir(input_folder + nam, full_path=True))
            for fi in range(len(filenames)):  #
                name = filenames[fi][len(hrpath)+1:]
                # print(filenames[fi], name)
                cur = Image.open(filenames[fi])
                # cur_lr = cur.resize([40, 40])
                sizelr = tuple(((np.array(cur.size)).astype(int) / scale).astype(int))
                cur_lr = cur.resize(sizelr, Image.BICUBIC)
                cur_lr.save(savepath1 + '/' + name)
                

# *****************************************************************************************************
def make_lmdb_from_imgs(data_path,
                        lmdb_path,
                        img_path_list,
                        keys,
                        batch=5000,
                        compress_level=1,
                        multiprocessing_read=False,
                        n_thread=40,
                        map_size=None):
    """Make lmdb from images.

    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records 1)image name (with extension),
    2)image shape, and 3)compression level, separated by a white space.

    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1

    We use the image name without extension as the lmdb key.

    If `multiprocessing_read` is True, it will read all the images to memory
    using multiprocessing. Thus, your server needs to have enough memory.

    Args:
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_path_list (str): Image path list.
        keys (str): Used for lmdb keys.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. Default: False.
        n_thread (int): For multiprocessing.
        map_size (int | None): Map size for lmdb env. If None, use the
            estimated size from images. Default: None
    """

    assert len(img_path_list) == len(keys), ('img_path_list and keys should have the same length, '
                                             f'but got {len(img_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Totoal images: {len(img_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    if multiprocessing_read:
        # read all the images to memory (multiprocessing)
        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        print(f'Read images with multiprocessing, #thread: {n_thread} ...')
        pbar = tqdm(total=len(img_path_list), unit='image')

        def callback(arg):
            """get the image data and update pbar."""
            key, dataset[key], shapes[key] = arg
            pbar.update(1)
            pbar.set_description(f'Read {key}')

        pool = Pool(n_thread)
        for path, key in zip(img_path_list, keys):
            pool.apply_async(read_img_worker, args=(osp.join(data_path, path), key, compress_level), callback=callback)
        pool.close()
        pool.join()
        pbar.close()
        print(f'Finish reading {len(img_path_list)} images.')

    # create lmdb environment
    if map_size is None:
        # obtain data size for one image
        img = cv2.imread(osp.join(data_path, img_path_list[0]), cv2.IMREAD_UNCHANGED)
        _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        data_size_per_img = img_byte.nbytes
        print('Data size per image is: ', data_size_per_img)
        data_size = data_size_per_img * len(img_path_list)
        map_size = data_size * 10

    env = lmdb.open(lmdb_path, map_size=map_size)

    # write data to lmdb
    pbar = tqdm(total=len(img_path_list), unit='chunk')
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update(1)
        pbar.set_description(f'Write {key}')
        key_byte = key.encode('ascii')
        if multiprocessing_read:
            img_byte = dataset[key]
            h, w, c = shapes[key]
        else:
            _, img_byte, img_shape = read_img_worker(osp.join(data_path, path), key, compress_level)
            h, w, c = img_shape

        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')


def create_lmdb():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # # HR images
    folder_path = rootpath + 'DIV2K/GT_sub/'  # 'datasets/DIV2K/DIV2K_train_HR_sub'
    lmdb_path = root + '/datasets/DIV2K/DIV2K_train_HR_sub2im.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    print('save .lmdb to %s' % lmdb_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    
    for scale in [4]:  # 2, 3,
        folder_path = rootpath + 'DIV2K/GTmiscs%dLR_sub/' % scale  # 'datasets/DF2K/LRbicx2_sub'
        lmdb_path = root + '/datasets/DIV2K/DIV2K_train_LR_bicubic_X%d_sub2im.lmdb' % scale  # 'datasets/DF2K/LRbicx2_sub.lmdb'
        img_path_list, keys = prepare_keys(folder_path)
        make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
        print('save .lmdb to %s' % lmdb_path)


def prepare_keys(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]
    
    return img_path_list, keys


def read_img_worker(path, key, compress_level):
    """Read image worker.

    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.

    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    """

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


from abc import ABCMeta, abstractmethod


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class MemcachedBackend(BaseStorageBackend):
    """Memcached storage backend.

    Attributes:
        server_list_cfg (str): Config file for memcached server list.
        client_cfg (str): Config file for memcached client.
        sys_path (str | None): Additional path to be appended to `sys.path`.
            Default: None.
    """

    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys
            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError('Please install memcached to enable MemcachedBackend.')

        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(self.server_list_cfg, self.client_cfg)
        # mc.pyvector servers as a point which points to a memory cache
        self._mc_buffer = mc.pyvector()

    def get(self, filepath):
        filepath = str(filepath)
        import mc
        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf


class LmdbBackend(BaseStorageBackend):
    """Lmdb storage backend.

    Args:
        db_paths (str | list[str]): Lmdb database paths.
        client_keys (str | list[str]): Lmdb client keys. Default: 'default'.
        readonly (bool, optional): Lmdb environment parameter. If True,
            disallow any write operations. Default: True.
        lock (bool, optional): Lmdb environment parameter. If False, when
            concurrent access occurs, do not lock the database. Default: False.
        readahead (bool, optional): Lmdb environment parameter. If False,
            disable the OS filesystem readahead mechanism, which may improve
            random read performance when a database is larger than RAM.
            Default: False.

    Attributes:
        db_paths (list): Lmdb database path.
        _client (list): A list of several lmdb envs.
    """

    def __init__(self, db_paths, client_keys='default', readonly=True, lock=False, readahead=False, **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError('Please install lmdb to enable LmdbBackend.')

        if isinstance(client_keys, str):
            client_keys = [client_keys]

        if isinstance(db_paths, list):
            self.db_paths = [str(v) for v in db_paths]
        elif isinstance(db_paths, str):
            self.db_paths = [str(db_paths)]
        assert len(client_keys) == len(self.db_paths), ('client_keys and db_paths should have the same length, '
                                                        f'but received {len(client_keys)} and {len(self.db_paths)}.')

        self._client = {}
        for client, path in zip(client_keys, self.db_paths):
            self._client[client] = lmdb.open(path, readonly=readonly, lock=lock, readahead=readahead, **kwargs)

    def get(self, filepath, client_key):
        """Get values according to the filepath from one lmdb named client_key.

        Args:
            filepath (str | obj:`Path`): Here, filepath is the lmdb key.
            client_key (str): Used for distinguishing differnet lmdb envs.
        """
        filepath = str(filepath)
        assert client_key in self._client, (f'client_key {client_key} is not ' 'in lmdb clients.')
        client = self._client[client_key]
        with client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError


class FileClient(object):
    """A general file client to access files in different backend.

    The client loads a file or text in a specified backend from its path
    and return it as a binary file. it can also register other backend
    accessor with a given name and backend class.

    Attributes:
        backend (str): The storage backend type. Options are "disk",
            "memcached" and "lmdb".
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {
        'disk': HardDiskBackend,
        'memcached': MemcachedBackend,
        'lmdb': LmdbBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend not in self._backends:
            raise ValueError(f'Backend {backend} is not supported. Currently supported ones'
                             f' are {list(self._backends.keys())}')
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    def get(self, filepath, client_key='default'):
        # client_key is used only for lmdb, where different fileclients have
        # different lmdb environments.
        if self.backend == 'lmdb':
            return self.client.get(filepath, client_key)
        else:
            return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)


import torch
import random


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, ('The len of keys should be 2 with [input_key, gt_key]. ' f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(f'{input_key} folder and {gt_key} folder should both in lmdb '
                         f'formats. But received {input_key}: {input_folder}; '
                         f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(dict([(f'{input_key}_path', lmdb_key), (f'{gt_key}_path', lmdb_key)]))
        return paths


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


from torch.utils import data as data
from torchvision.transforms.functional import normalize


class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
        self.io_backend_opt['client_keys'] = ['lq', 'gt']
        self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        
    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    # mainsubimages()
    generatecleanLR()
    # create_lmdb()
