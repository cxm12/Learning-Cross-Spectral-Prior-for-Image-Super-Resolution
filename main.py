import torch
import utility
import loss
from option import args
# from optiontest import args
from optionv2 import args as argsv2
from trainer import Trainer, TrainerCV18
import model
from torch.utils.data import DataLoader
from data.datasetRGBNIR import StereoDatasetTestWarp, StereoDatasetTestWarpY, StereoDatasetTestWarpColorCorrect, StereoDataDepthRGB
from data.visnir import VISNIR
# from data.middlebury import MiddleburyDataset

torch.manual_seed(args.seed)

device = 'cuda'
server = 1
args.cpu = False
args.n_GPUs = 1
    
    
def main():
    checkpoint = utility.checkpoint(args)
    
    if server:
        datapath = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/data'
        listpath = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/lists'
        disp_path = '/mnt/home/user1/MCX/StereoSR/StereoSR_CVPR20_code/Middlebury/train/fullpng/displeftim/'
        rgb_path = '/mnt/home/user1/MCX/StereoSR/StereoSR_CVPR20_code/Middlebury/train/fullpng/leftd/'
        dispt_path = '/mnt/home/user1/MCX/StereoSR/StereoSR_CVPR20_code/Middlebury/test/quartersize/dispquarter/'
        rgbt_path = '/mnt/home/user1/MCX/StereoSR/StereoSR_CVPR20_code/Middlebury/test/quartersize/left/'
        trainsplit = ['20170223_1639', '20170222_0951', '20170222_1423', '20170221_1357', '20170222_0715',
                      '20170222_1207', '20170222_1638', '20170223_0920', '20170223_1217', '20170223_1445',
                      '20170224_1022', '20170224_0742']
    else:
        datapath = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data'
        listpath = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/lists'
        disp_path = 'E:/file/python_project/StereoSR/StereoSR_CVPR20_code/Middlebury/train/fullpng/displeftim/'
        rgb_path = 'E:/file/python_project/StereoSR/StereoSR_CVPR20_code/Middlebury/train/fullpng/leftd/'
        
        dispt_path = 'E:/file/python_project/StereoSR/StereoSR_CVPR20_code/Middlebury/test/quartersize/dispquarter/'
        rgbt_path = 'E:/file/python_project/StereoSR/StereoSR_CVPR20_code/Middlebury/test/quartersize/left/'
        trainsplit = ['20170222_0951']
    testsplit = ['20170224_0742100']  # ['20170222_1423100', '20170223_1639100', '20170222_0951100', ]  #
    # ['20170224_0742']  #
    
    if 'NIRw' in args.modelname:
        warpnir = './cs-stereo-master/result/WarpNIR/'  #
    else:
        warpnir = None
    
    if checkpoint.ok:
        if 'StereoSR' in args.modelname:
            train_set = StereoDatasetTestWarpY(datapath, listpath, trainsplit, istrain=True, warpnir=warpnir, scale=args.scale)
            test_set = StereoDatasetTestWarpY(datapath, listpath, testsplit, istrain=False, warpnir=warpnir, scale=args.scale, testonly=args.test_only)
        elif 'cocrtin' in args.modelname:
            train_set = StereoDatasetTestWarpColorCorrect(datapath, listpath, trainsplit, istrain=True, warpnir=warpnir,
                                              scale=args.scale, patch=args.patch_size)
            test_set = StereoDatasetTestWarpColorCorrect(datapath, listpath, testsplit, istrain=False, warpnir=warpnir,
                                             scale=args.scale, testonly=args.test_only, patch=args.patch_size)
        elif 'depthrgb' in args.modelname:
            train_set = StereoDataDepthRGB(rgb_path, disp_path, istrain=True, scale=args.scale, patch=args.patch_size)
            test_set = StereoDataDepthRGB(rgbt_path, dispt_path, istrain=False, scale=args.scale, testonly=args.test_only,
                                                         patch=args.patch_size)
        # else:
        #     train_set = StereoDatasetTestWarp(datapath, listpath, trainsplit, istrain=True, warpnir=warpnir, scale=args.scale, patch=args.patch_size)
        #     test_set = StereoDatasetTestWarp(datapath, listpath, testsplit, istrain=False, warpnir=warpnir, scale=args.scale, testonly=args.test_only, patch=args.patch_size)
        train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=True)
        
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = TrainerCV18(args, train_loader, test_loader, _model, _loss, checkpoint)
        
        # t.testdata()
        # t.testBic(server=server)
        # t.testFeatureVis()
        # exit()
        # t.testFLOP()
        # exit()
        while not t.terminate(server=server):
            if 'MMNet' in args.modelname:
                t.testMMNet_ReadCorrect()
                t.trainMMNet_ReadCorrect()
            elif 'StereoSR' in args.modelname:
                t.trainTWY()
                t.testY()
            elif 'depthrgb' in args.modelname:
                t.trainTWReadCorrect()
            elif 'cocrtin' in args.modelname:
                t.trainTWReadCorrect()
                t.testReadCorrect()
            elif 'cocrt' in args.modelname:
                t.trainTW()
                t.test()
            elif 'Enhance' in args.modelname:
                t.trainTWdark()
                t.testdark()
            else:
                t.traindarkSR()
                t.testdarkSR(server=server)
        
        checkpoint.done()
        
        ''' original
    if checkpoint.ok:
        loader = data.Data(args)
        
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.trainTW()
            # t.train()
            t.test()
    
        checkpoint.done()
        '''


def mainHAT():
    args = argsv2
    checkpoint = utility.checkpoint(args)

    if 'RGBNIR' in args.data_test:
        if server:
            datapath = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/data'
            listpath = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/lists'
            trainsplit = ['20170223_1639', '20170222_0951', '20170222_1423', '20170221_1357', '20170222_0715',
                          '20170222_1207', '20170222_1638', '20170223_0920', '20170223_1217', '20170223_1445',
                          '20170224_1022', '20170224_0742']
        else:
            datapath = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data'
            listpath = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/lists'
            trainsplit = ['20170222_0951']
            args.cpu = True
    
        if args.test_only:
            testsplit = ['20170222_1423100', '20170223_1639100', '20170222_0951100', '20170224_0742100']
        else:
            testsplit = ['20170224_0742100']  # '20170222_1423100', '20170223_1639100', '20170222_0951100', []  #
        warpnir = './cs-stereo-master/result/WarpNIR/'  #
    
        if checkpoint.ok:
            train_set = StereoDatasetTestWarpColorCorrect(datapath, listpath, trainsplit, istrain=True, warpnir=warpnir,
                                                          scale=args.scale, patch=args.patch_size)
            test_set = StereoDatasetTestWarpColorCorrect(datapath, listpath, testsplit, istrain=False, warpnir=warpnir,
                                                         scale=args.scale, testonly=args.test_only,
                                                         patch=args.patch_size)
            train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=True)
        
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = TrainerCV18(args, train_loader, test_loader, _model, _loss, checkpoint)
            # t.testFLOP(device=device)
            # exit()
            if args.test_only:
                num, meanps, mss = t.testReadCorrect()
                exit()
        
            while not t.terminate(server=server):
                t.testReadCorrect()
                t.trainTWReadCorrect()
        
            checkpoint.done()
    # elif 'depth' in args.data_test:
    #     if server:
    #         disp_path = '/mnt/home/user1/MCX/dataset/depthSR/middlebury/train/Depth/'
    #         rgb_path = '/mnt/home/user1/MCX/dataset/depthSR/middlebury/train/RGB/'
    #         dispt_path = '/mnt/home/user1/MCX/dataset/depthSR/middlebury/test/Depth/'
    #         rgbt_path = '/mnt/home/user1/MCX/dataset/depthSR/middlebury/test/RGB/'
    #     else:
    #         args.cpu = True
    #         path = 'E:/file/python_project/StereoSR/StereoSR_CVPR20_code/Middlebury/middlebury/'
    #         rgb_path = 'F:/SRdata/train_data/stereo/middlebury/train/RGB/'
    #         dispt_path = 'F:/SRdata/train_data/stereo/middlebury/test/Depth/'
    #         rgbt_path = 'F:/SRdata/train_data/stereo/middlebury/test/RGB/'
    #
    #     # train_set = StereoDataDepthRGB(rgb_path, disp_path, istrain=True, scale=args.scale, patch=128*2)
    #     # test_set = StereoDataDepthRGB(rgbt_path, dispt_path, istrain=False, scale=args.scale, testonly=args.test_only, patch=128*2)
    #     data_args = {
    #         'crop_size': (args.patch_size, args.patch_size),
    #         'max_rotation_angle': 0,
    #         'do_horizontal_flip': False,
    #         'crop_deterministic': True,
    #         'scaling': args.scale[0]
    #     }
    #     test_set = MiddleburyDataset(data_path=path, split='test', **data_args)
    #     test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=True)
    #     train_set = MiddleburyDataset(data_path=path)
    #     train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=args.batch_size, shuffle=True)
    #
    #     _model = model.Model(args, checkpoint)
    #     _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    #     t = TrainerCV18(args, train_loader, test_loader, _model, _loss, checkpoint)
    #
    #     while not t.terminate(server=server):
    #         t.testMiddlebury()
    #         # t.trainMiddlebury()
    else:
        if checkpoint.ok:
            train_set = VISNIR(argsv2, train=True)
            test_set = VISNIR(argsv2, train=False, valid=False)
            _model = model.Model(argsv2, checkpoint)
            train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=argsv2.batch_size, shuffle=False)
            test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
        
            _loss = loss.Loss(argsv2, checkpoint) if not argsv2.test_only else None
            t = TrainerCV18(argsv2, train_loader, test_loader, _model, _loss, checkpoint)
            if args.test_only:
                t.testVISNIR()
                exit()

            while not t.terminateVISNIR():
                t.trainVISNIR()
                t.testVISNIR()
            checkpoint.done()
       

def main_visnir():
    checkpoint = utility.checkpoint(argsv2)
    if checkpoint.ok:
        train_set = VISNIR(argsv2, train=True)
        test_set = VISNIR(argsv2, train=False, valid=True)
        _model = model.Model(argsv2, checkpoint)
        train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=argsv2.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
        
        _loss = loss.Loss(argsv2, checkpoint) if not argsv2.test_only else None
        t = TrainerCV18(argsv2, train_loader, test_loader, _model, _loss, checkpoint)

        while not t.terminateVISNIR():
            if 'MMNet' in args.modelname:
                # t.testMMNet_VISNIR()
                t.trainMMNet_VISNIR()
            else:
                t.trainVISNIR()
                t.testVISNIR()
        
        checkpoint.done()


def mainSISR():
    # from optionsisr import args as argsv2
    args = argsv2
    checkpoint = utility.checkpoint(argsv2)
    server = 1
    argsv2.cpu = False
    argsv2.n_GPUs = 1
    
    if server:
        datapath = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/data'
        listpath = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/lists'
        trainsplit = ['20170223_1639', '20170222_0951', '20170222_1423', '20170221_1357', '20170222_0715',
                      '20170222_1207', '20170222_1638', '20170223_0920', '20170223_1217', '20170223_1445',
                      '20170224_1022', '20170224_0742']
    
    if args.test_only:
        testsplit = ['20170222_1423100', '20170223_1639100', '20170222_0951100', '20170224_0742100']
    else:
        testsplit = ['20170224_0742100']  # '20170222_1423100', '20170223_1639100', '20170222_0951100', []  #
    if checkpoint.ok:
        if 'RGBNIR' == argsv2.data_test[0]:  # 'RANUS'  # 'RGBNIR'  # 'nirscene'
            train_set = StereoDatasetTestWarpColorCorrect(datapath, listpath, trainsplit, istrain=True, warpnir=None,
                                                          scale=args.scale, patch=args.patch_size)
            test_set = StereoDatasetTestWarpColorCorrect(datapath, listpath, testsplit, istrain=False, warpnir=None,
                                                         scale=args.scale, testonly=args.test_only,
                                                         patch=args.patch_size)
        else:  # 'RANUS'  # 'nirscene'
            train_set = VISNIR(argsv2, train=True)
            test_set = VISNIR(argsv2, train=False, valid=True)
        
        train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=argsv2.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=True)
        
        _model = model.Model(argsv2, checkpoint)
        _loss = loss.Loss(argsv2, checkpoint) if not argsv2.test_only else None
        t = TrainerCV18(argsv2, train_loader, test_loader, _model, _loss, checkpoint)

        # t.testFLOP()
        # if 'RGBNIR' == argsv2.data_test[0]:
        #     t.testReadCorrect()
        # else:
        #     t.testVISNIR()
        # exit()
        while not t.terminate(server=server):
            if 'RGBNIR' == argsv2.data_test[0]:
                t.trainTWReadCorrect()
                t.testReadCorrect()
            else:
                t.trainVISNIR()
                t.testVISNIR()
        checkpoint.done()
        

if __name__ == '__main__':
    # import os
    # print(os.listdir('F:/SRdata/train_data/NIR_VIS/nirscene1/Train/'))
    # exit()
    # main_visnir()
    mainHAT()
    # main()
    # mainSISR()
