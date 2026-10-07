import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('QT5Agg')
import numpy as np
from PIL import Image
import os
from scipy.signal import convolve2d
from PIL import ImageEnhance
from skimage.metrics import structural_similarity as compare_ssim


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    # if not im1.shape == im2.shape:
    #     raise ValueError("Input Imagees must have the same dimensions")
    # if len(im1.shape) > 2:
    #     raise ValueError("Please input the images with 1 channel")
    # M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def psnr(im1, im2):
    diff =np.float64(im1[:]) - np.float64(im2[:])
    rmse = np.sqrt(np.mean(diff**2))
    psnr = 20*np.log10(255/rmse)
    return psnr,rmse


def get_filenames(paths):
    filenames = []
    for path in paths:
        for root, dirs, files in os.walk(path):
            for f in files:
                filenames.append(os.path.join(root, f))
    return filenames


def rgb2y(rgb):
    h,w,d = rgb.shape
    rgb=np.float32(rgb)/255.0
    y=rgb*(np.reshape([65.481, 128.553, 24.966],[1,1,3])/255.0)
    y=y[:,:,0]+y[:,:,1]+y[:,:,2]
    y=np.reshape(y,[h,w])+16/255.0
    return np.uint8(y*255+0.5)


def cal_psnr_ssimY(path0='E:/file\python_project/RGBNIRStereo/SR/Testset_20170224_0742/HR/',  # 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/test100/20170221_1357_100/RGBResize/GT/',
                   path1='E:/file/python_project/SISR/EDSR-PyTorch-master/EDSR-PyTorch-master\experiment/test/results-RGBNIR/'):
                   # path1='../experiment/test/results-test/'):
    filenames = get_filenames([path0])
    filenames1 = get_filenames([path1])
    imnum = 10  # len(filenames)
    mean_psnr2 = 0
    meanssim2 = 0
    scale = 4
    num = 0
    gamma = True
    for fi in range(0, imnum):
        num += 1
        im = Image.open(filenames[fi])
        im1 = Image.open(filenames1[fi])
        size = im.size
        sizelr = tuple(((np.array(im.size)).astype(int) / scale).astype(int))
        lrim = im.resize(sizelr, Image.BICUBIC)
        bicim = lrim.resize(size, Image.BICUBIC)
        
        if gamma:
            # Gamma correct
            im = np.power(np.array(im) / 255.0, 0.55) * 255.0
            im1 = np.power(np.array(im1) / 255.0, 0.55) * 255.0
            bicim = np.power(np.array(bicim) / 255.0, 0.55) * 255.0
        
        if len(np.array(im).shape) >= 3:
            c = rgb2y(np.array(im))
            bic = rgb2y(np.array(bicim))
        else:
            c = np.array(im)
            bic = np.array(bicim)
        h, w = c.shape
        # if len(np.array(im1).shape) >= 3:
        #     c1 = rgb2y(np.array(im1))[:h, :w]
        # else:
        #     c1 = np.array(im1)[:h, :w]
        # h, w = c1.shape
        # c = c[:h, :w]

        # psnr1, _ = psnr(c1, c)
        # score = compute_ssim(c1, c)

        psnr1, _ = psnr(bic, c)
        score = compute_ssim(bic, c)
        
        mean_psnr2 += psnr1
        meanssim2 += score
        print(filenames[fi], psnr1, score)

    print('total image %d' % num, mean_psnr2/num, meanssim2/num)


##
import glob


def cal_psnr_ssimRGB():
    test = 'RANUS'  # 'nirscene'  # 'RGBNIR'  #
    if test == 'RANUS':
        pathgt = 'F:/SRdata/train_data/NIR_VIS/%s/Test/RGB/' % test
        # pathgt = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/%s/Test/RGB/' % test
    else:
        pathgt = 'F:/SRdata/train_data/NIR_VIS/%s1/Test/' % test
        # pathgt = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/%s1/Test/' % test
    dir = os.listdir(pathgt)
    
    for scale in [2, 4]:  # [2, 3, 4]:  #
        mean_psnr2 = 0
        meanssim2 = 0
        num = 0
        # path1 = '/opt/home/user1/MCX/CVPR23-HAT-main/experiments/HAT-S_SRx%d/visualization/%s/' % (scale, test)
        # path1 = 'E:/file/python_project/SISR/EDSR-PyTorch-master/EDSR-PyTorch-master/experiment/test/results-%s/s%d/' % (test, scale)
        # path1 = 'E:/file/python_project/SISR/RCAN-master/RCAN_TestCode/SR/BI/RCAN/results-%s/s%d/' % (test, scale)
        # path1 = 'E:/file/python_project/SISR/SMSR-master-CVPR21/experiment/SMSR/results-%s/s%d/' % (test, scale)  # /%s/s%d/' % (test, scale)  #
        # path1 = 'E:/file/python_project/SISR/Non-Local-Sparse-Attention-CVPR21main/experiment/test/results-%s/s%d/' % (test, scale)
        path1 = 'E:/file/python_project/SISR/AAAI22-ENLCA-master/src/experiment/ENLCN/results-%s/s%d/' % (test, scale)  # 256/results-%s/s%d/' % (test, scale)
        # path1 = './experiment/test/EDSR_cocrtin_RGBNIRw2branch1/results-%s/' % test
        # path1 = 'E:/file/python_project/RGBNIRStereo/experiment/%s_s%d/EDSR2branch1Dynamic_nirscenev1/results-%s22/' % (test, scale, test)
        # if test == 'RANUS':
        #     path1 = 'F:/SRdata/train_data/NIR_VIS/%s/Test/TestBics%d/RGB/' % (test, scale)
        # else:
        #     path1 = 'F:/SRdata/train_data/NIR_VIS/%s1/TestBics%d/' % (test, scale)
        
        for subdir in dir:
            print('subdir = ', subdir)
            path0 = pathgt + subdir
            path = path1 + subdir
            if test == 'RANUS':
                filenames = get_filenames([path0])
            else:
                filenames = glob.glob(path0 + '/*rgb.tiff')
            filenames1 = get_filenames([path])
            imnum = len(filenames)
            for fi in range(0, imnum):
                num += 1
                im = Image.open(filenames[fi])
                im1 = Image.open(filenames1[fi])
                
                c = np.array(im)
                h, w, _ = c.shape
                c = c[:h // scale * scale, :w // scale * scale, :]  # [680, 1024]
                c1 = np.array(im1)
            
                psnr1, _ = psnr(c1, c)
                score = compare_ssim(c1, c, multichannel=True)
            
                mean_psnr2 += psnr1
                meanssim2 += score
                # print(filenames[fi], psnr1, score)
    
        print('Scale=%d; total image %d' % (scale, num), mean_psnr2 / num, meanssim2 / num)


def cal_psnr_ssimRGB_HAT():
    test = 'RANUS'  # 'nirscene'  #
    if test == 'RANUS':
        pathgt = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/%s/Test/RGB/' % test
    else:
        pathgt = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/%s1/Test/' % test
    dir = os.listdir(pathgt)
    mean_psnr = []
    meanssim = []
    for scale in [2, 4]:  # [2, 3, 4]:  #
        mean_psnr2 = 0
        meanssim2 = 0
        num = 0
        path1 = '/opt/home/user1/MCX/CVPR23-HAT-main/experiments/HAT-S_SRx%d/visualization/%s/' % (scale, test)

        for subdir in dir:
            print('subdir = ', subdir)
            path0 = pathgt + subdir
            if test == 'RANUS':
                filenames = get_filenames([path0])
            else:
                filenames = glob.glob(path0 + '/*rgb.tiff')  # 0004_rgb.tiff
            imnum = len(filenames)
            for fi in range(0, imnum):
                num += 1
                im = Image.open(filenames[fi])
                name = filenames[fi][len(path0)+1:-4]
                # 50100104__HAT-S_SRx2.png / 100104_.png
                im1 = Image.open(path1 + subdir + name + '_HAT-S_SRx%d.png' % scale)  # indoor0004_rgb_HAT-S_SRx2.png
                
                c = np.array(im)
                h, w, _ = c.shape
                c = c[:h // scale * scale, :w // scale * scale, :]  # [680, 1024]
                c1 = np.array(im1)
                
                psnr1, _ = psnr(c1, c)
                score = compare_ssim(c1, c, multichannel=True)
                
                mean_psnr2 += psnr1
                meanssim2 += score
                print(filenames[fi], psnr1, score)
        mean_psnr.append(mean_psnr2 / num)
        meanssim.append(meanssim2 / num)
        print('Scale=%d; total image %d' % (scale, num), mean_psnr2 / num, meanssim2 / num)
    print('Scale=2/4; ', mean_psnr, meanssim)


def cal_psnr_ssimRGBNIRStereo():
    testlst = ['20170224_0742100', '20170222_0951100', '20170222_1423100', '20170223_1639100']  #
    testlst1 = ['0742', '0951', '1423', '1639']  #
    test = 'RGBNIR'  # 'RANUS'  # 'nirscene'  #
    # if test == 'RANUS':
    #     pathgt = 'F:/SRdata/train_data/NIR_VIS/%s/Test/RGB/' % test
    #     pathgt = ''
    mean_psnr = []
    meanssim = []
    for scale in [2, 4]:  # [2, 3, 4]:  #
        mean_psnr2 = 0
        meanssim2 = 0
        num = 0
        n = 0
        for dir in testlst:
            print('subdir = ', dir)
            dir1 = testlst1[n]
            n += 1
            # pathgt = 'F:/SRdata/train_data/NIR_VIS/%s1/Test/' % test
            pathgt = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/data/%s/RGBCorrect/' % dir
            path1 = '/opt/home/user1/MCX/CVPR23-HAT-main/experiments/HAT-S_SRx%d/visualization/RN%s/' % (scale, dir1)  # 256/results-%s/s%d/' % (test, scale)

            filenames = get_filenames([path1])
            # else:
            #     filenames = glob.glob(path1 + '/*rgb.tiff')
            filenames1 = get_filenames([pathgt])
            
            filenames.sort()
            filenames1.sort()
            
            imnum = len(filenames)
            for fi in range(0, imnum):
                num += 1
                im = Image.open(filenames[fi])
                im1 = Image.open(filenames1[fi])
                
                c = np.array(im)
                h, w, _ = c.shape
                c1 = np.array(im1)
                c = c[:h // scale * scale, :w // scale * scale, :]
                c1 = c1[:h // scale * scale, :w // scale * scale, :]

                psnr1, _ = psnr(c1, c)
                score = compare_ssim(c1, c, multichannel=True)
                
                mean_psnr2 += psnr1
                meanssim2 += score
                print(filenames[fi], psnr1, score)
        mean_psnr.append(mean_psnr2 / num)
        meanssim.append(meanssim2 / num)
        print('Scale=%d; total image %d' % (scale, num), mean_psnr2 / num, meanssim2 / num)
    print('Scale=2/4; ', mean_psnr, meanssim)


def cal_psnr_RGBNIRStereo_ablation():
    testlst = ['20170222_0951100', '20170222_1423100', '20170223_1639100']  # '20170224_0742100',
    testlst1 = ['220951', '221423', '231639']  # '240742',
    methodlst = ['EDSR_cocrt_RGB', 'EDSR_cocrt_RGBNIR', 'EDSR_cocrt_RGBNIRw', 'EDSR_cocrt_RGBNIRw2branch0',
                 'EDSR_cocrt_RGBNIR2branch1', 'EDSR_cocrtin_RGBNIRw2branch1', 'EDSR_cocrtin_RGBNIRw2branch1']
    i = 0
    for test in testlst:
        pathgt = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/%s/RGBCorrect/' % test
        test1 = testlst1[i]
        i += 1
        our = 1
        for method in methodlst:
            if not method == 'EDSR_cocrtin_RGBNIRw2branch1':
                path1 = 'E:/file/python_project/RGBNIRStereo/experiment/test/ablation/%s/results-%s/' % (method, test1)
            else:
                if our == 1:
                    path1 = 'E:/file/python_project/RGBNIRStereo/experiment/test/%s/model-18/results-%s/' % (method, test1)
                    our = 71
                else:
                    path1 = 'E:/file/python_project/RGBNIRStereo/experiment/test/%s/model-71/results-%s/' % (method, test1)
            mean_psnr2 = 0
            meanssim2 = 0
            filenames = glob.glob(path1 + '/*.png')
            filenames1 = glob.glob(pathgt + '/*.png')
            imnum = len(filenames)
            for fi in range(0, imnum):
                im = Image.open(filenames[fi])
                im1 = Image.open(filenames1[fi])
        
                c = np.array(im)
                h, w, _ = c.shape
                c1 = np.array(im1)[:h, :w, :]
                h, w, _ = c1.shape
                c = c[:h, :w, :]
                psnr1, _ = psnr(c1, c)
                score = compare_ssim(c1, c, multichannel=True)
                mean_psnr2 += psnr1
                meanssim2 += score
                # print(filenames[fi], psnr1, score)
            print('Testset = %s/ method%s;  total image %d' % (test, method, imnum), mean_psnr2 / imnum, meanssim2 / imnum)


def Resave(path0='F:/SRdata/train_data\stereo/rgbnir/rgbnir_stereo/test100/20170221_1357_100/RGBResize/GT/'):
    filenames = get_filenames([path0])
    imnum = len(filenames)
    savepath = 'F:/SRdata/train_data\stereo/rgbnir/rgbnir_stereo/test100/20170221_1357_100/RGBResize/'
    savebic = 'F:/SRdata/train_data\stereo/rgbnir/rgbnir_stereo/test100/20170221_1357_100/RGBResize/X4/'
    # os.makedirs(savepath, exist_ok=True)
    os.makedirs(savebic, exist_ok=True)
    for fi in range(0, imnum):
        name = filenames[fi][len(path0):]
        im = Image.open(filenames[fi])
        # im.save(savepath+name)
        
        sizelr = tuple(((np.array(im.size)).astype(int) / 4).astype(int))
        im.resize(sizelr, Image.BICUBIC).save(savebic+name)


def Lighter():
    path0='F:/SRdata/train_data\stereo/rgbnir/rgbnir_stereo/test100/20170221_1357_100/RGBResize/GT/'
    filenames = get_filenames([path0])
    savepath = 'F:/SRdata/train_data\stereo/rgbnir/rgbnir_stereo/test100/20170221_1357_100/RGBResize/GTlighten1/'
    os.makedirs(savepath, exist_ok=True)
    for fi in range(0, 2):
        name = filenames[fi][len(path0):]
        im = Image.open(filenames[fi])
        
        # # 然后对其增加亮度对比度等操作
        # enh_con = ImageEnhance.Contrast(im)  # 增加对比度
        # enh_con = enh_con.enhance(1.3)
        # imenh_con = np.array(enh_con)
        #
        # # enh_con.save(savepath + name)
        # enh_bri = ImageEnhance.Brightness(enh_con)  # 增加亮度
        # image_bright = enh_bri.enhance(2.5)
        # image_bright.save(savepath + name)
        
        # Gamma correct
        imageR2p2 = np.power(np.array(im) / 255.0, 0.55)  # 显示器的伽玛值 gamma < 1： 变亮
        Image.fromarray(np.uint8(imageR2p2*255)).save(savepath + name)


from scipy import misc


def generateLR():
    testsplit = ['20170224_0742100', '20170222_0951100', '20170222_1423100', '20170223_1639100']
    hrpath = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/%s/RGBCorrect/'% testsplit[0]  # 'F:/SRdata/testsets/testset10/HR/'
    
    filenames = get_filenames([hrpath])
    for s in range(4, 5):
        scale = s
        savepath = hrpath + 'HR/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        for fi in range(len(filenames)):
            print(filenames[fi])
            name = filenames[fi][len(hrpath):-4] + '.png'
            cur = np.array(Image.open(filenames[fi]))
            if len(cur.shape) == 3:
                h, w, c = cur.shape
                h = h - h % scale
                w = w - w % scale
                cur = cur[:h, :w, :]
                cur_lr = misc.imresize(cur, 1 / scale, 'bicubic')
                # cur_bc = misc.imresize(cur_lr, [h,w], 'bicubic')
            else:
                h, w = cur.shape
                cur_lr = misc.imresize(cur[:h, :w], 1 / scale, 'bicubic')
            im = np.float32(np.maximum(0, np.minimum(255, cur)))
            Image.fromarray(np.uint8(im)).save(savepath + name)


def CalPSNR_Bic():
    testsplit = ['20170224_0742100', '20170222_0951100', '20170222_1423100', '20170223_1639100']
   
    for i in range(1, 4):
        hrpath = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/%s/RGBCorrect' % testsplit[i]
        filenames = get_filenames([hrpath])
        for s in range(2, 5):
            mean_psnr2 = 0
            meanssim2 = 0
            scale = s
            savepath = hrpath + 's%d/' % scale
            os.makedirs(savepath, exist_ok=True)
            num = len(filenames)
            for fi in range(num):
                # print(filenames[fi])
                name = filenames[fi][len(hrpath):-4] + '.png'
                cur = np.array(Image.open(filenames[fi]))
                h, w, c = cur.shape
                h = h - h % scale
                w = w - w % scale
                cur = cur[:h, :w, :]
                cur_lr = misc.imresize(cur, 1 / scale, 'bicubic')
                cur_bc = misc.imresize(cur_lr, [h, w, 3], 'bicubic')
            
                Image.fromarray(np.uint8(cur_lr)).save(savepath + name)
            
                psnr1, _ = psnr(cur_bc, cur)
                score = compare_ssim(cur_bc, cur, multichannel=True)
                mean_psnr2 += psnr1
                meanssim2 += score
                # print(filenames[fi], psnr1, score)
        
            print(testsplit[i], 'Scale=%d; total image %d' % (scale, num), mean_psnr2 / num, meanssim2 / num)


def cal_psnr_ssim2im(path0='F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/20170224_0742100/RGBCorrect/',
                     path1='F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/20170224_0742100/Bics4/'):  # 20170223_1639100
    
    # path0 = 'D:/U_copy/Stereo_RGBNIR/figure/ablation/hr/'
    # path1 = 'D:/U_copy/Stereo_RGBNIR/figure/ablation/bic/'  # sisr/'  # srcat/'  # srcat-cvm/'  # wo-cmft/'  # wo-cvm/' #
    path1 = 'E:/file/python_project/RGBNIRStereo/experiment/RGBNIR_s4/MMNet_cocrtin_NIRw_RGBNIR/'  #_HRnir/results-240742
    # path1 = 'E:/file/python_project/RGBNIRStereo/experiment/test/EDSR_cocrtin_RGBNIRw2branch1/model-71/results-240742/'
    # path1 = 'E:/file/python_project/SISR/AAAI22-ENLCA-master/src/experiment/ENLCN256_x4/results-RGBNIR/0742/'
    # 'E:/file/python_project/StereoSR/Stereo SR_other methods/PASSRnet-CVPR19/results/NIRRGB/'
    # 'E:/file/python_project/StereoSR/Stereo SR_other methods/stereosr-master-cvpr2018/result/RGBNIR/s3/'
    # path1 = 'E:/file/python_project/StereoSR/NAFNet-main_CVPRW22/experiments/results/NIRRGB/240742/RGB/'
    # 'E:/file/python_project/SISR/Non-Local-Sparse-Attention-CVPR21main/experiment/test/results-RGBNIR/'):
    # 'E:/file/python_project/SISR/SMSR-master-CVPR21/experiment/RGBNIR/RGBNIR-S2/'
    # 'E:/file/python_project/SISR/EDSR-PyTorch-master/EDSR-PyTorch-master/experiment/test/results-RGBNIR-s3/'):
    # 'E:/file/python_project/SISR/RCAN-master/RCAN_TestCode/SR/BI/RCAN/RGBNIR/x8/'
    # path1 = 'E:/file/python_project/SISR/CVPR23-HAT-main/experiments/HAT-S_SRx4/visualization/RN1639/'  # RN1423/'  # RN0951/'  # RN0742
    # testname = ['20170224_0742100']  # ['20170223_1639100']  # ['20170222_1423100']  # ['20170222_0951100']  #
    
    path0 = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/%s/RGBCorrect/' % testname[0]
    # path0 = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/data/%s/RGBCorrect/' % testname[0]
    # path1 = '/opt/home/user1/MCX/RGBNIRStereo/SR/experiment/RGBNIR_s4/SISR-cat-ENLCN/results-RGBNIR/model_best30/results-231639/'
    
    # ## NIRSCENE / RANUS dataset
    # path0 = 'F:/SRdata/train_data/NIR_VIS/nirscene1/Test/'  # indoor/indoor #country
    # path0 = 'F:/SRdata/train_data/NIR_VIS/RANUS/Test/RGB/50/'
    # # path1 = 'F:/SRdata/train_data/NIR_VIS/nirscene1/TestBics2/indoor/'
    # # path1 = 'E:/file/python_project/SISR/SMSR-master-CVPR21/experiment/SMSR/results-nirscene/s2/indoor/'
    # # path1 = 'E:/file/python_project/SISR/Non-Local-Sparse-Attention-CVPR21main/experiment/test/results-nirscene/s2/indoor/'
    # # path1 = 'E:/file/python_project/SISR/AAAI22-ENLCA-master/src/experiment/ENLCN/results-nirscene/s2/indoor/'
    # path1 = 'E:/file/python_project/RGBNIRStereo/experiment/nirscene_s2/EDSR2branch1Dynamic_nirscenev1/results-nirscene/indoor/'
    # path1 = 'E:/file/python_project/SISR/CVPR23-HAT-main/experiments/HAT-S_SRx4/visualization/nirscene/'
    # path1 = 'E:/file/python_project/SISR/CVPR23-HAT-main/experiments/HAT-S_SRx4/visualization/RANUS/'

    filenames = get_filenames([path0])
    filenames1 = get_filenames([path1])
    filenames.sort()
    filenames1.sort()
    
    imnum = len(filenames)
    fnm = []
    for fi in range(0, imnum):
        if not '_nir.tiff' in filenames[fi]:
            fnm.append(filenames[fi])
    filenames = fnm
    imnum = 1  # len(filenames)

    mean_psnr2 = 0
    meanssim2 = 0
    num = 0
    for fi in range(0, imnum):
        # if not ('089398' in filenames[fi]):  #  '014164'or '036985' in filenames[fi]):
        #     continue
        num += 1
        name = '240742_005677'  # filenames[fi]  # '0001_rgb'
        im = Image.open(path0 + name + '.png')  # name
        im1 = Image.open(path1 + name + 'SR.png')  # 'SR.png')  # filenames1[fi])
        # try:
        #     im1 = Image.open(filenames1[fi])  # Image.open(path1 + name + 'SR.png')  # '.png')  #
        # except:
        #     try:
        #         im1 = Image.open(path1 + name + '_x2.0_.png')  # 'SR.png')  # filenames1[fi])
        #     except:
        #         im1 = Image.open(path1 + name + '_x4_.png')  # 'SR.png')  # filenames1[fi])

        # im1 = Image.open('./result/traindata/240742_005677HR.png')
        # im = Image.open('./result/traindata/240742_005677HRycbcr.png')
        c = np.array(im)
        bic = np.array(im1)
        h, w, _ = np.array(bic).shape
        # bicim = misc.imresize(bic, [h*4, w*4, 3], 'bicubic')
        # c = c[:h, 64:w+64, :]
        c = c[:h, :w, :]
        bicim = np.array(bic)[:h, :w, :]
        
        psnr1, _ = psnr(bicim, c)
        score = compare_ssim(bicim, c, multichannel=True)
    
        mean_psnr2 += psnr1
        meanssim2 += score
        print(name, psnr1, '/', score)

    print('total image %d' % num, mean_psnr2 / num, meanssim2 / num)


def cal_psnr_ssim2dir(testname, scale, subname):
    # path0 = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/%s/RGBCorrect/' % testname[0]
    path0 = '/mnt/home/user1/MCX/dataset/StereoData/rgbnir/rgbnir_stereo/data/%s/RGBCorrect/' % testname
    path1 = '/mnt/home/user1/MCX/StereoSR/RGBNIRStereo/experiment/RGBNIR_s%d/HAT_cocrtin/results-%s/' % (scale, subname)

    filenames = get_filenames([path0])
    filenames1 = get_filenames([path1])
    filenames.sort()
    filenames1.sort()
    
    imnum = len(filenames)
    fnm = []
    for fi in range(0, imnum):
        if not '_nir.tiff' in filenames[fi]:
            fnm.append(filenames[fi])
    filenames = fnm
    imnum = len(filenames)
    
    mean_psnr2 = 0
    meanssim2 = 0
    num = 0
    for fi in range(0, imnum):
        num += 1
        name = filenames[fi][len(path0):-4]  #
        im = Image.open(path0 + name + '.png')  # name
        im1 = Image.open(path1 + name + 'SR.png')  # 'SR.png')  # filenames1[fi])

        c = np.array(im)
        bic = np.array(im1)
        h, w, _ = np.array(bic).shape
        # bicim = misc.imresize(bic, [h*4, w*4, 3], 'bicubic')
        # c = c[:h, 64:w+64, :]
        c = c[:h, :w, :]
        bicim = np.array(bic)[:h, :w, :]
        
        psnr1, _ = psnr(bicim, c)
        score = compare_ssim(bicim, c, multichannel=True)
        
        mean_psnr2 += psnr1
        meanssim2 += score
        # print(name, psnr1, '/', score)
    
    print('Testset %s, scale%d, total image %d' % (subname, scale, num), mean_psnr2 / num, meanssim2 / num)


def cal_psnr_ssim2imNIR():
    # p = (27.092 + 27.313 + 25.949 + 24.886) / 4
    # s = (0.7696 + 0.8280 + 0.7137 + 0.6892) / 4
    p = (27.313 + 28.921 + 27.021 + 25.961) / 4
    s = (0.7911 + 0.8496 + 0.7673 + 0.7152) / 4
    print(p, s)
    exit()
    
    testname = '20170223_1639100'  # '20170224_0742100'  # '20170222_1423100'  # '20170222_0951100'  #
    path0 = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/%s/NIRCorrect/' % testname
    path1 = 'E:/file/python_project/SISR/AAAI22-ENLCA-master/src/experiment/checkpoints4/231639/'
    
    filenames = get_filenames([path0])
    filenames1 = get_filenames([path1])
    filenames.sort()
    filenames1.sort()
    imnum = len(filenames)

    mean_psnr2 = 0
    meanssim2 = 0
    num = 0
    for fi in range(0, imnum):
        num += 1
        name = filenames[fi]
        im = Image.open(name)
        im1 = Image.open(filenames1[fi])
        
        c = np.array(im)
        bic = np.array(im1)
        h, w, _ = np.array(bic).shape
        
        c = c[:h, :w]
        bicim = np.array(bic)[:h, :w, 0]
        
        psnr1, _ = psnr(bicim, c)
        score = compare_ssim(bicim, c, multichannel=True)
        
        mean_psnr2 += psnr1
        meanssim2 += score
        print(filenames1[fi], filenames[fi], psnr1, '/', score)
    
    print('total image %d' % num, mean_psnr2 / num, meanssim2 / num)


def generateBic(path0='F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/20170224_0742100/RGBCorrect/',
                     path1='F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/20170224_0742100/Bics4/', scale=4):
    # savepathbic = path0.replace('RGBCorrect', 'LRup/Bics%d' % scale)
    savepathbic = path0.replace('NIRCorrect', 'NIRHRdownup/Bics%d' % scale)
    os.makedirs(savepathbic, exist_ok=True)
    
    filenames = get_filenames([path0])
    # filenames1 = get_filenames([path1])
    imnum = len(filenames)
    mean_psnr2 = 0
    meanssim2 = 0
    num = 0
    for fi in range(0, imnum):
        num += 1
        name = filenames[fi][len(path0):]
        im = Image.open(filenames[fi])
        size = im.size
        sizelr = tuple(((np.array(im.size)).astype(int) / scale).astype(int))
        lrim = im.resize(sizelr, Image.BICUBIC)
        bicim = lrim.resize(size, Image.BICUBIC)
        im1 = bicim
        
        # im1 = Image.open(filenames1[fi])
        # im1 = im1.resize(size, Image.BICUBIC)
        
        c = np.array(im)
        bic = np.array(im1)
        bicim = bic
        h, w = np.array(bicim).shape
        c = c[:h, :w]
        bicim = np.array(bicim)[:h, :w]
        Image.fromarray(np.uint8(bicim)).save(savepathbic + name)
        
        psnr1, _ = psnr(bicim, c)
        score = compare_ssim(bicim, c, multichannel=True)

        mean_psnr2 += psnr1
        meanssim2 += score
        print(filenames[fi], psnr1, score)
    print('total image %d' % num, mean_psnr2 / num, meanssim2 / num)


def generateBicVISNIR(path0='F:/SRdata/train_data/NIR_VIS/nirscene1/Test/'):
    path0 = 'F:/SRdata/train_data/NIR_VIS/RANUS/Test/RGB/'
    dir = os.listdir(path0)
    for scale in [2, 3, 4]:  #
        # savepathbic = path0.replace('Test', 'TestBics%d' % scale)
        # savepathlr = path0.replace('Test', 'Tests%d' % scale)
        savepathbic = path0.replace('RGB', 'RGBBics%d' % scale)
        savepathlr = path0.replace('RGB', 'RGBs%d' % scale)
        os.makedirs(savepathbic, exist_ok=True)
        mean_psnr2 = 0
        meanssim2 = 0
        num = 0
        import glob
        for subdir in dir:
            # img_list = glob.glob(path0 + subdir + '/*_rgb.tiff')  # list(scandir(path0 + subdir, full_path=True))
            img_list = glob.glob(path0 + subdir + '/*.png')
            save_folder = os.path.join(savepathbic, subdir)
            save_folderlr = os.path.join(savepathlr, subdir)
            os.makedirs(save_folder, exist_ok=True)
            os.makedirs(save_folderlr, exist_ok=True)
            for fi in img_list:
                num += 1
                name = fi[len(path0 + subdir):]  #'0004_rgb.tiff'  #
                im0 = Image.open(fi)
                # im0 = Image.open('F:/SRdata/train_data/NIR_VIS/nirscene1/Test/indoor/' + name)
                c = np.array(im0)  # [682, 1024]
                h, w, _ = c.shape
                c = c[:h // scale * scale, :w // scale * scale, :]  # [680, 1024]
                im = Image.fromarray(c)
                sizelr = tuple(((np.array(im.size)).astype(int) / scale).astype(int))

                lrim = im.resize(sizelr, Image.BICUBIC)
                bicim = np.array(lrim.resize(im.size, Image.BICUBIC))
                psnr1, _ = psnr(bicim, c)
                score = compare_ssim(bicim, c, multichannel=True)
                # print(fi, psnr1, score)
                
                Image.fromarray(np.uint8(bicim)).save(save_folder + '/' + name)
                Image.fromarray(np.uint8(lrim)).save(save_folderlr + '/' + name)
                mean_psnr2 += psnr1
                meanssim2 += score
        print('Scale %d total image %d' % (scale, num), mean_psnr2 / num, meanssim2 / num)
    

def evaluateBicPSNR2im():
    import torch
    import utility
    from skimage.measure import compare_ssim
    # nirscene/indoor/0004_rgb.png
    name = '0004_rgb.tiff'
    scale = 4
    im = Image.open('F:/SRdata/train_data/NIR_VIS/nirscene1/Test/indoor/' + name)
    sizelr = tuple(((np.array(im.size)).astype(int) / scale).astype(int))
    ###############################################
    # Bicubic 27.50657715996027 0.8395701
    lrim = im.resize(sizelr, Image.BICUBIC)
    im1 = lrim.resize(im.size, Image.BICUBIC)
    c = np.array(im)
    bicim = np.array(im1)
    h, w, _ = bicim.shape
    c = c[:h, :w, :]  # Image.fromarray(np.uint8(bicim)).save(save_folder + '/' + name)
    psnr1, _ = psnr(bicim, c)
    score = compare_ssim(bicim, c, multichannel=True)
    print(psnr1, score)
    ###########################################
    # Bicubic 26.48159371851954 0.8274639
    lrim = im.resize(sizelr, Image.BICUBIC)
    c = np.array(im)  # [682, 1024]
    h, w, _ = c.shape
    c = c[:h//scale * scale, :w//scale * scale, :]  # [680, 1024]
    size = tuple((np.array(Image.fromarray(c).size).astype(int)).astype(int))
    bicim = np.array(lrim.resize(size, Image.BICUBIC))
    psnr1, _ = psnr(bicim, c)
    score = compare_ssim(bicim, c, multichannel=True)
    print(psnr1, score)
    
    # ##################################################
    # im = np.array(im)
    # ih, iw = im.shape[:2]
    # # size = Image.fromarray(im).size
    # # !!!!!!!!! *********************************************** !!!!!!!!!! #
    # im = im[0:(ih // scale) * scale, 0:(iw // scale) * scale]  # 放前面：27.57144251754413 0.84100097
    # lrim = np.array(Image.fromarray(im).resize(sizelr, Image.BICUBIC))  # 放前面：26.48159371851954 0.8274639
    # # !!!!!!!!!! *********************************************** !!!!!!!!! #
    # a = 1 / 255
    # rgb_range = 1
    # lrim = np.ascontiguousarray(lrim.transpose((2, 0, 1)), dtype=np.float32)
    # lrtensor = torch.from_numpy(lrim).float().mul_(a)  # float32 [0, 1]
    # im = np.ascontiguousarray(im.transpose((2, 0, 1)), dtype=np.float32)
    # hrtensor = torch.from_numpy(im).float().mul_(a)  # float32 [0, 1]
    #
    # im2 = np.squeeze(hrtensor.numpy()).transpose(1, 2, 0) * (255 / rgb_range)
    # im2 = np.uint8(np.clip(im2, 0, 255))
    # size1 = tuple(((np.array(Image.fromarray(im2).size)).astype(int)).astype(int))
    # lrnorm = np.clip(np.squeeze(lrtensor.numpy()).transpose(1, 2, 0), 0, rgb_range) * (255 / rgb_range)
    # bc = np.uint8(np.array(Image.fromarray(np.uint8(lrnorm)).resize(size1, Image.BICUBIC)))  # [0, 49]
    # Image.fromarray(bc).save('bic.png')
    # psnr1, _ = utility.psnr(bc, im2)
    # score = compare_ssim(bc, im2, multichannel=True)
    # ####################################################
    # print(psnr1, score)
# 33.147126020857726 0.918004618 33.147126020857726 0.91800461


import cv2


def testYUV():
    rgb = cv2.imread('F:\SRdata/train_data\stereo/rgbnir/rgbnir_stereo/data/20170224_0742100/RGBCorrect/240742_001489.png')
    ## !!!!!!!! Uint8 ！！！！！！！！！！
    ycbcr = cv2.cvtColor(np.uint8(rgb), cv2.COLOR_BGR2YUV)
    
    rgb1 = np.uint8(cv2.cvtColor(ycbcr, cv2.COLOR_YUV2RGB))
    rgb = np.uint8(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    psnr1, _ = psnr(np.around(rgb), np.around(rgb1))
    print(psnr1)
    Image.fromarray(rgb1).save('./result/traindata/HRycbcr.png')
    rgb2 = cv2.cvtColor(ycbcr, cv2.COLOR_YUV2BGR)
    cv2.imwrite('./result/traindata/HRycbcrcv.png', rgb2)


## 柱状图
def drawHis():
    shops = ['Common', 'Light', 'Glass', 'Glossy', 'Vegetation', 'Skin', 'Clothing', 'Mean']  # ['Bicubic', 'SISR', 'SRcat', 'SRcat w/ CVM', 'w/o CMFT', 'w/o CVM', 'Full', 'HR']  # 'Baseline',

    # sales_product_c = [1.3565, 0.6819, 0.5593, 0.5747, 0.5595, 0.5651, 0, 0.5109]
    # sales_product_l = [1.701, 0.6101, 0.3043, 0.3396, 0.2908, 0.2822, 0, 0.293]
    # sales_product_gla = [1.9886, 0.7964, 0.4386, 0.4043, 0.4068, 0.437, 0, 0.3912]
    # sales_product_glo = [2.3736, 1.1294, 0.6633, 0.6302, 0.5824, 0.8161, 0, 0.3965]
    # sales_product_v = [1.6742, 0.9505, 0.6772, 0.7773, 0.6878, 0.7174, 0, 0.5973]
    # sales_product_s = [1.5242, 1.4586, 0.8968, 0.8409, 0.949, 1.0394, 0, 1.0353]
    # sales_product_clo = [2.0174, 0.6759, 0.5612, 0.5441, 0.5718, 0.5418, 0, 0.57]
    # sales_product_m = [1.5795, 0.7879, 0.5126, 0.5139, 0.506, 0.5499, 0, 0.474]
    
    sales_product_b = [1.3565, 1.701, 1.9886, 2.3736, 1.6742, 1.5242, 2.0174, 1.5795]
    sales_product_sisr = [0.6819, 0.6101, 0.7964, 1.1294, 0.9505, 1.4586, 0.6759, 0.7879]
    sales_product_scat = [0.5593, 0.3043, 0.4386, 0.6633, 0.6772, 0.8968, 0.5612, 0.5126]
    sales_product_scatwc = [0.5747, 0.3396, 0.4043, 0.6302, 0.7773, 0.8409, 0.5441, 0.5139]
    sales_product_wocmft = [0.5595, 0.2908, 0.4068, 0.5824, 0.6878, 0.949, 0.5718, 0.506]
    sales_product_wocvm = [0.5651, 0.2822, 0.437, 0.8161, 0.7174, 1.0394, 0.5418, 0.5499]
    sales_product_f = [0.521, 0.2558, 0.3899, 0.594, 0.6113, 0.9846, 0.5131, 0.4837]
    sales_product_h = [0.5109, 0.293, 0.3912, 0.3965, 0.5973, 1.0353, 0.57, 0.474]
    
    
    # 创建分组柱状图，需要自己控制x轴坐标
    xticks = np.arange(len(shops))
    
    fig, ax = plt.subplots(figsize=(12, 5.6))  # w, h  图片内容
    # ['Common','Light', 'Glass', 'Glossy',  'Vegetation', 'Skin', 'Clothing', 'Mean']
    # ax.bar(xticks, sales_product_c, width=0.2, label="Common", color='#b3b3b3')
    # ax.bar(xticks + 0.2, sales_product_l, width=0.2, label="Light", color='#440154')
    # ax.bar(xticks + 0.4, sales_product_gla, width=0.2, label="Glass", color='#31688e')
    # ax.bar(xticks + 0.6, sales_product_glo, width=0.2, label="Glossy", color='#35b779')
    # ax.bar(xticks + 0.6, sales_product_v, width=0.2, label="Vegetation", color='peru')
    # ax.bar(xticks + 0.6, sales_product_s, width=0.2, label="Skin", color='darkorchid')
    # ax.bar(xticks + 0.6, sales_product_clo, width=0.2, label="Clothing", color='darkcyan')
    # ax.bar(xticks + 0.6, sales_product_m, width=0.2, label="Mean", color='darkseagreen')

    # ['Bicubic', 'SISR', 'SRcat', 'SRcat w/ CVM', 'w/o CMFT', 'w/o CVM', 'Full', 'HR']  # 'Baseline',
    # ax.bar(xticks, sales_product_b, width=0.1, label="Bicubic", color='#49759c')
    
    ax.bar(xticks + 0.1, sales_product_sisr, width=0.1, label="SISR", color='#738595')
    ax.bar(xticks + 0.2, sales_product_scat, width=0.1, label="SRcat", color='#c27e79')
    ax.bar(xticks + 0.3, sales_product_scatwc, width=0.1, label="SRcat w/ CVM", color='#fbdd7e')
    ax.bar(xticks + 0.4, sales_product_wocmft, width=0.1, label="w/o CMFT", color='#a6c875')
    ax.bar(xticks + 0.5, sales_product_wocvm, width=0.1, label="w/o CVM", color='#82cbb2')
    ax.bar(xticks + 0.6, sales_product_f, width=0.1, label="Full", color='#d1768f')
    ax.bar(xticks + 0.7, sales_product_h, width=0.1, label="HR", color='#9c6da5')
    
    # ax.set_title("Grouped Bar plot", fontsize=15)
    ax.set_xlabel("Materials", fontsize=20)
    ax.set_ylabel("RMSE", fontsize=20)
    ax.legend()
    
    # 最后调整x轴标签的位置
    ax.set_xticks(xticks + 0.25)
    ax.set_xticklabels(shops, fontsize=16)
    plt.show()
    fig.tight_layout()
    fig.set_size_inches(12.5, 6)  # 画布大小
    fig.savefig('./result/fablation.png', dpi=300)


##  pixel intensity分布折线图
def DrawRGBNIRPixel_distribution():
    # pathRANUS = 'F:/SRdata/train_data/NIR_VIS/RANUS/Test/'
    # file = glob.glob(pathRANUS + 'RGB/50/*.png')
    # filenir = glob.glob(pathRANUS + 'NIR/50/*.png')
    pathnirscene = 'F:/SRdata/train_data/NIR_VIS/nirscene1/Test/country/'
    file = glob.glob(pathnirscene + '*_rgb.tiff')
    filenir = glob.glob(pathnirscene + '*_nir.tiff')
    # pathstereo = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/20170224_0742100/'
    # file = glob.glob(pathstereo + 'RGBResize/*')
    # filenir = glob.glob(pathstereo + 'NIRResize/*')
    
    if 'nirscene1' in file[0]:
        h, w = 670, 1024
    elif 'RANUS' in file[0]:
        h, w = 1024, 1024
    else:
        h, w = 582, 429
        
    i = 0
    rgball = []
    nirall = []
    for name in file[:1]:
        f_rgb = name
        f_nir = filenir[i]
        i += 1
        # f_rgb = pathRANUS + 'RGB/50/100001_.png'
        # f_nir = pathRANUS + 'NIR/50/5001_nir_.png'
        f_rgb = pathnirscene + '0000_nir.tiff'
        f_nir = pathnirscene + '0000_rgb.tiff'
        
        hrrgb = cv2.cvtColor(cv2.imread(f_rgb), cv2.COLOR_BGR2GRAY)  # [1024, 1024]
        hrrgb = hrrgb[:h, :w]  # [1024, 1024]
        rgball.append(np.expand_dims(hrrgb, -1))
        hrnir = cv2.cvtColor(cv2.imread(f_nir), cv2.COLOR_BGR2GRAY)
        hrnir = hrnir[:h, :w]  # [1024, 1024]
        nirall.append(np.expand_dims(hrnir, -1))
    hrrgb = np.uint8(np.mean(np.concatenate(rgball, -1), 2))
    hrnir = np.uint8(np.mean(np.concatenate(nirall, -1), 2))
    
    ##  图
    hist1 = cv2.calcHist([hrnir], [0], None, [30], [0, 255])  # [0, 30])  # 掩膜图直方图，参数需要修改
    hist2 = cv2.calcHist([hrrgb], [0], None, [30], [0, 255])  # [0, 30])  #
    # plt.plot(hist1, color='b', label='NIR', linestyle='--')
    # plt.plot(hist2, color='r', label='RGB', linestyle='-.')

    plt.plot(hist1, color='b', label='RGB', linestyle='--')
    plt.plot(hist2, color='r', label='NIR', linestyle='-.')
    plt.legend()
    plt.savefig('./result/fstereo-his2.png', dpi=300)
    plt.show()


##  沿着一条边的pixel intensity折线图
def DrawRGBNIRPixel_lineEdge():
    pathRANUS = 'F:/SRdata/train_data/NIR_VIS/RANUS/Test/'
    # file = glob.glob(pathRANUS + 'RGB/50/*.png')
    # filenir = glob.glob(pathRANUS + 'NIR/50/*.png')
    pathnirscene = 'F:/SRdata/train_data/NIR_VIS/nirscene1/Test/country/'
    # file = glob.glob(pathnirscene + '*_rgb.tiff')
    # filenir = glob.glob(pathnirscene + '*_nir.tiff')
    pathstereo = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/20170224_0742100/'
    # file = glob.glob(pathstereo + 'RGBResize/*')
    # filenir = glob.glob(pathstereo + 'NIRResize/*')

    f_rgb = pathRANUS + 'RGB/50/100001_.png'
    f_nir = pathRANUS + 'NIR/50/5001_nir_.png'
    f_rgb = pathnirscene + '0005_nir.tiff'
    f_nir = pathnirscene + '0005_rgb.tiff'
    if 'nirscene1' in f_rgb:
        h, w = 670, 1024
    elif 'RANUS' in f_rgb:
        h, w = 1024, 1024
    else:
        h, w = 582, 429

    
    hrrgb = np.uint8(cv2.cvtColor(cv2.imread(f_rgb), cv2.COLOR_BGR2GRAY)[:h, :w])  # [1024, 1024]
    hrnir = np.uint8(cv2.cvtColor(cv2.imread(f_nir), cv2.COLOR_BGR2GRAY)[:h, :w])
    
    pixelrgb = []
    pixelnir = []
    for i in range(hrrgb.shape[1]):
        pixelnir.append(hrnir[271, i])
        pixelrgb.append(hrrgb[271, i])
    
    pixelrgb = []
    pixelnir = []
    # for i in range(hrrgb.shape[0]):
    for i in range(200, 420):
        pixelnir.append(hrnir[i, 615])
        pixelrgb.append(hrrgb[i, 615])
    y = np.arange(0, len(pixelnir))

    ##  图
    plt.plot(y, pixelrgb, color='r', label='RGB')
    plt.plot(y, pixelnir, color='g', label='NIR')
    plt.legend()
    plt.savefig('./result/fline.png', dpi=300)
    plt.show()


def FFT_Image():
    import cv2
    from numpy.fft import ifftshift
    save_multi = False
    path1 = 'E:/file/python_project/RGBNIRStereo/result/FFT/im2/'
    sublst = ['NIR', 'RGB']  # ['DarkNIR', 'DarkRGB', 'NIR', 'RGB']  # ['DarkNIR', 'DarkRGB']  #
    for sub in sublst:
        img1 = cv2.imread(path1 + sub + "014.png")  # "240742_009469p.png")  #
        # img1 = cv2.resize(img1, dsize=None, fx=0.5, fy=0.5)
        img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
        # h, w = img1.shape[:2]
        # print(h, w)
        # cv2.namedWindow("W0")
        # cv2.imshow("W0", img2)
        # cv2.waitKey(delay=0)
        
        # 将图像转化到频域内并绘制频谱图
        ##numpy实现
        plt.rcParams['font.family'] = 'SimHei'  # 将全局中文字体改为黑体
        f = np.fft.fft2(img2)
        fshift = np.fft.fftshift(f)  # 将0频率分量移动到图像的中心
        magnitude_spectrum0 = 20 * np.log(np.abs(fshift))
        # 傅里叶逆变换
        # Numpy实现
        ifshift = np.fft.ifftshift(fshift)
        # 将复数转为浮点数进行傅里叶频谱图显示
        ifimg = np.log(np.abs(ifshift))
        if_img = np.fft.ifft2(ifshift)
        origin_img = np.abs(if_img)
        if save_multi:
            imggroup = [img2, magnitude_spectrum0, ifimg, origin_img]
            titles0 = ['原始图像', '经过移动后的频谱图', '逆变换得到的频谱图', '逆变换得到的原图']
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                plt.xticks([])  # 除去刻度线
                plt.yticks([])
                plt.title(titles0[i])
                plt.imshow(imggroup[i], cmap='gray')
            plt.show()
            ##OpenCV实现
            dft = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum1 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            plt.subplot(121), plt.imshow(img2, cmap='gray')
            plt.title('原图'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(magnitude_spectrum1, cmap='gray')
            plt.title('频谱图'), plt.xticks([]), plt.yticks([])
            plt.savefig(path1 + sub + 'ffreq.png', dpi=300)
            plt.show()
        else:
            dft = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum1 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            print(np.mean(magnitude_spectrum1))
            # plt.imshow(magnitude_spectrum1, cmap='gray')
            # plt.xticks([]), plt.yticks([])  # plt.title('频谱图')
            # plt.savefig(path1 + sub + 'ffreq-p.png', dpi=300)
            # plt.show()
            imcolor = cv2.applyColorMap(np.uint8(magnitude_spectrum1), 6)  # cv2.COLORMAP_JET
            cv2.imwrite(path1 + '%s240742_009469color6.png' % sub, imcolor)
    
    
if __name__ == '__main__':
    # generateBicVISNIR()
    # evaluateBicPSNR()
    # drawHis()
    # testYUV()
    # Lighter()
    # Resave()
    # cal_psnr_ssimY()
    # cal_psnr_ssimRGB_HAT()
    # cal_psnr_RGBNIRStereo_ablation()
    # cal_psnr_ssim2im()
    # cal_psnr_ssim2imNIR()
    # FFT_Image()
    # CalPSNR_Bic()
    # generateLR()
    # DrawRGBNIRPixel_distribution()
    # DrawRGBNIRPixel_lineEdge()
    # exit()
    i = 0
    # for testname in ['20170224_0742100']:  # ['20170222_1423100']:  # ['20170222_0951100']:  # ,, '20170223_1639100', , , ,  []:  #]:  # ,
    for testname in ['20170222_0951100', '20170222_1423100', '20170223_1639100']:
        # for testset in ['bic', 'sisr', 'srcat', 'srcat-cvm', 'wo-cmft', 'wo-cvm', 'full']:
        # # for testset in ['our', 'stereosr', 'nlsn', 'smsr', 'rcan', 'edsr', 'bic']:
        #     cal_psnr_ssim2im(path0='D:/U_copy/Stereo_RGBNIR/figure/ablation/hr/',
        #             path1='D:/U_copy/Stereo_RGBNIR/figure/ablation/%s/' % testset)
        # name = ['240742']  # ['1423']  # ['0951']  # , , '1639',
        name = ['220951', '221423', '231639']  #
        # if testname == '20170224_0742100':
        #     i += 1
        #     continue
        
        for scale in [2, 4]:  # ,3
            cal_psnr_ssim2dir(testname=testname, scale=scale, subname=name[i])
            # cal_psnr_ssim2im(path0='F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/%s/RGBCorrect/' % testname,
            #     # path1='E:/file/python_project/SISR/Non-Local-Sparse-Attention-CVPR21main/experiment/test/results-RGBNIR/S%d-%s/' % (scale, name[i]))
            #     # path1='E:/file/python_project/SISR/reference-based/MASA-SR-main-CVPR21/test_results/RGBNIR/GAN/%s/' % name[i])
            #   # path1='E:/file/python_project/SISR/reference-based/MASA-SR-main-CVPR21/test_results/RGBNIR/GAN/240742bic/')
            # # path1='E:/file/python_project/SISR/AAAI22-ENLCA-master/src/experiment/ENLCN256_x%d/results-RGBNIR/%s/' % (scale, name[i]))
            # # path1='E:/file/python_project/SISR/SMSR-master-CVPR21/experiment/SMSR_X%d/results-%s/' % (scale, testname))
            # # path1='E:/file/python_project/RGBNIRStereo/experiment/test/PASSR_cocrt_RGBNIR/results-%s/' % name[i])
            #  # E:\file\python_project\RGBNIRStereo\experiment\test\PASSR_cocrt_RGBNIR\results-240742
            #  #    path1='E:/file/python_project/StereoSR/Stereo SR_other methods/PASSRnet-CVPR19/results/NIRRGB/')
            #     path1='E:/file/python_project/RGBNIRStereo/experiment/test/EDSR_cocrtin_RGBNIRw2branch1/model-71/results-240742/')
            #     # path1='F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/20170224_0742100/s4/')
            #     # path1='E:/file/python_project/SISR/AAAI22-ENLCA-master/src/experiment/ENLCN256_x4/results-RGBNIR/%s/'%name[0])
            #     # path1='E:/file/python_project/RGBNIRStereo/experiment/test/EDSR_cocrtin_RGBNIRw2branch1/model-71/results-240742/')
            #     # path1='E:/file/python_project/RGBNIRStereo/SR\experiment/test/StereoSR_cocrt_RGBNIR/results-240742')
            # # path1='E:/file/python_project/StereoSR/Stereo SR_other methods/stereosr-master-cvpr2018/result/RGBNIR/%s/s%d/' % (testname, scale))
            # #         # path1='E:/file/python_project/SISR/SMSR-master-CVPR21/experiment/RGBNIR/S%d/results-%s/' % (scale, testname))
            # #         # path1='E:/file/python_project/SISR/Non-Local-Sparse-Attention-CVPR21main/experiment/test/%s/results-RGBNIR/'%testname)
            # #         # path1='E:/file/python_project/SISR/EDSR-PyTorch-master/EDSR-PyTorch-master/experiment/test/%s/results-RGBNIR-s%d/' % (testname, scale))
            # #         # path1='E:/file/python_project/SISR/RCAN-master/RCAN_TestCode/SR/BI/RCAN/RGBNIR/%s/x%d/' % (testname, scale))
            #
            # # generateBic(path0='F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/%s/NIRCorrect/' % testname,
            # #             path1='F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/%s/NIRs%d/' % (testname, scale),
            # #             scale=scale)
            # # print(testname + str(scale) + '+++++++++++++++++++++++++++')
            # # exit()
        i += 1

