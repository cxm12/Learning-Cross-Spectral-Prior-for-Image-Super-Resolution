import numpy as np
import torch
import utility
import loss
from optiontest import args
from optionv2 import args as argsv2
from trainer import Trainer, TrainerCV18
import model
from torch.utils.data import DataLoader
from data.datasetRGBNIR import Set5Data, StereoDatasetTestWarp, StereoDatasetTestWarpY, StereoDatasetTestWarpColorCorrect
from data.visnir import VISNIR


datapath = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/data'
listpath = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/lists'
trainsplit = ['20170223_1639', '20170222_0951', '20170222_1423', '20170221_1357', '20170222_0715', '20170222_1207',
    '20170222_1638', '20170223_0920', '20170223_1217', '20170223_1445', '20170224_1022', '20170224_0742']

testsplit = ['20170224_0742100', '20170222_0951100', '20170222_1423100', '20170223_1639100']  # ['20170224_0742100']
# , ['20170222_1423100']  #


if 'NIRw' in args.modelname:
    warpnir = './cs-stereo-master/result/WarpNIR/'  #
else:
    warpnir = None
    

## test on NIRRGB stereo dataset
def test():
    meanp = []
    means = []
    for testset in testsplit:
        torch.manual_seed(args.seed)
        checkpoint = utility.checkpoint(args)
    
        if checkpoint.ok:
            if 'StereoSR' in args.modelname:
                test_set = StereoDatasetTestWarpY(datapath, listpath, [testset], istrain=False, warpnir=warpnir,
                                                  scale=args.scale, testonly=args.test_only)
            elif 'cocrtin' in args.modelname:
                test_set = StereoDatasetTestWarpColorCorrect(datapath, listpath, [testset], istrain=False,
                     warpnir=warpnir, scale=args.scale, testonly=args.test_only, patch=args.patch_size,
                     cropHRfirst=True)
            else:
                test_set = StereoDatasetTestWarp(datapath, listpath, [testset], istrain=False, warpnir=warpnir,
                                                 scale=args.scale, testonly=args.test_only, patch=args.patch_size)
                
            train_loader = None  # DataLoader(dataset=train_set, num_workers=0, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=True)

            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = TrainerCV18(args, train_loader, test_loader, _model, _loss, checkpoint, server=server)
            if 'MMNet' in args.modelname:
                num, meanps, mss = t.testMMNet_ReadCorrect()
                means.append(mss)
                meanp.append(meanps)
            elif 'StereoSR' in args.modelname:
                num, meanps, mss = t.testY()
            elif 'cocrtin' in args.modelname:
                num, meanps, mss = t.testReadCorrect()
            elif 'cocrt' in args.modelname:
                num, meanps, mss = t.test()
            elif 'Enhance' in args.modelname:
                num, meanps, mss = t.testdark()
            else:
                num, meanps, mss = t.testdarkSR(server=server)

            checkpoint.done()
            print(testset, '-------------- Mean PSNR/SSIM', num, meanps, mss)
    print('meanp, means', meanp, means)


import os

## test on SISR dataset
def testSISR():
    from optiontestset5 import args

    for test in ['Set5']:  # , 'Urban100', 'Set14', 'BSD100'
        savepath = './experiment/RGBNIR_s4/HAT_cocrtin_NIRw/%s/' % test
        os.makedirs(savepath, exist_ok=True)
        torch.manual_seed(args.seed)
        checkpoint = utility.checkpoint(args)
        inputpath = '/mnt/home/user1/MCX/dataset/TestSet/Set5/GT/image/*.bmp'
        # inputpath = '/mnt/home/user1/MCX/dataset/TestSet/Set14/GT/*.bmp'
        # inputpath = '/mnt/home/user1/MCX/dataset/TestSet/Urban100/GT/*.jpg'
    
        if checkpoint.ok:
            test_set = Set5Data(args, inputpath)
            train_loader = None  # DataLoader(dataset=train_set, num_workers=0, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=True)

            _model = model.Model(args, checkpoint)
            _loss = None
            t = TrainerCV18(args, train_loader, test_loader, _model, _loss, checkpoint, server=1)
            if 'cocrtin' in args.modelname:
                num, meanps, mss = t.testReadCorrectSISR(savepath)
            checkpoint.done()
            print(test, '-------------- Mean PSNR/SSIM', num, meanps, mss)


# original 'EDSR_cocrtin_RGBNIRw2branch1' test other 2 VISNIR datasets
def testVISNIR():
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    p = []
    s=[]
    if checkpoint.ok:
        train_set = VISNIR(argsv2, train=True)
        test_set = VISNIR(argsv2, train=False)
        train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=argsv2.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
        
        _model = model.Model(args, checkpoint)
        _loss = None
        t = TrainerCV18(args, train_loader, test_loader, _model, _loss, checkpoint, server=1, argsv2=argsv2)
            
        if 'MMNet' in args.modelname:
            num, meanps, mss = t.testMMNet_VISNIR()
        else:
            num, meanps, mss = t.testVISNIR()

        checkpoint.done()
        print('-------------- Mean PSNR/SSIM', num, meanps, mss)
        p.append(meanps)
        s.append(mss)
    
    print(np.mean(np.array(p)), np.mean(np.array(s)))


# retrained 'EDSR_cocrtin_RGBNIRw2branch1' test other 2 VISNIR datasets
def testVISNIRv2():
    meanp = []
    means = []
        
    argsv2.cpu = True
    torch.manual_seed(argsv2.seed)
    argsv2.test_only = True
    checkpoint = utility.checkpoint(argsv2)
    if checkpoint.ok:
        test_set = VISNIR(argsv2, train=False, valid=False, oneim=False, cropHRfirst=True)
        train_loader = None  # DataLoader(dataset=train_set, num_workers=0, batch_size=argsv2.batch_size, shuffle=False)
        test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
        
        _model = model.Model(argsv2, checkpoint)
        _loss = None
        t = TrainerCV18(argsv2, train_loader, test_loader, _model, _loss, checkpoint, server=server, argsv2=argsv2)
        
        # t.testBicPSNR_VISNIR()
        # exit()
        if 'MMNet' in args.modelname:
            num, meanps, mss = t.testMMNet_VISNIR()
            means.append(mss)
            meanp.append(meanps)
        else:
            num, meanps, mss = t.testVISNIR()

        checkpoint.done()
        print('-------------- Mean PSNR/SSIM', num, meanps, mss)


def colorDisparity():
    import cv2
    path = 'E:/file/python_project/RGBNIRStereo/cs-stereo-master/result/HR.png'  # _220951_040660
    dis = cv2.imread(path)

    # normalize float versions
    norm_img1 = cv2.normalize(dis, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # scale to uint8
    dis = (255 * norm_img1 * 2.4).astype(np.uint8)
    
    show_image = cv2.applyColorMap(dis, cv2.COLORMAP_JET)

    cv2.imwrite('E:/file/python_project/RGBNIRStereo/cs-stereo-master/result/HR_crop.png', show_image)
    exit()
    

def test_FLOP():
    torch.backends.cudnn.benchmark = True
    import torch.backends.cudnn as cudnn
    from torchvision.transforms import ToTensor
    from torch.autograd import Variable
    from optiontestset5 import args
    from thop import profile
    scale = 4
    device = 'cpu'
    checkpoint = utility.checkpoint(args)
    net = model.Model(args, checkpoint).cuda() 
        
    cudnn.benchmark = True
    with torch.no_grad():
        h, w = 128, 128  # 582, 428  #
        LRl = ToTensor()(np.zeros([h // (scale), w // (scale), 3], dtype=np.float32)).view(1, 3, h // (scale),
                                 w // (scale))
        LR_left = Variable(LRl).to(device)
    total = sum([param.nelement() for param in net.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))  # 8.11M
    exit()
    flops, params = profile(net, inputs=LR_left)
    # flops, params = profile(net, inputs=(LR_left, True))
    print(flops / 1e9, params / 1e6)
    exit()


if __name__ == '__main__':
    test_FLOP()
    
    # colorDisparity()
    testSISR()
    # testVISNIRv2()
    # test()
