from decimal import Decimal
import utility

import torch.nn.functional as F
import torch
import torch.nn.utils as utils
from torch import nn
from tqdm import tqdm
import numpy as np
import cv2
import os
from PIL import Image
import imageio
from tensorflow import __version__ as _tf_version
IS_TF_1 = _tf_version.startswith('1.')
# from data.middlebury import l1_loss_func, mse_loss_func

from sklearn.metrics import mean_absolute_error, mean_squared_error

if not IS_TF_1:
    from skimage.metrics import structural_similarity as compare_ssim
else:
    from skimage.measure import compare_ssim


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        
        self.error_last = 1e8
    
    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        
        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (hrrgb, lrrgb, hrnir, lrnir, _,) in enumerate(self.loader_train):
            hrrgb, lrrgb, hrnir, lrnir = self.prepare(hrrgb, lrrgb, hrnir, lrnir)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            # EDSR_cat
            # sr = self.model([lrrgb, hrnir], 0)
            # EDSR
            sr = self.model(lrrgb, 0)
            loss = self.loss(sr, hrrgb)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            
            timer_model.hold()
            
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            
            timer_data.tic()
        
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
    
    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            d.dataset.set_scale(0)
            num = 0
            for hrrgb, lrrgb, hrnir, lrnir, filename in tqdm(d, ncols=80):
                # for batch, (hrrgb, lrrgb, hrnir, lrnir, filename,) in tqdm(d, ncols=80):  # enumerate(self.loader_test):  #
                num += 1
                hrrgb, lrrgb, hrnir, lrnir = self.prepare(hrrgb, lrrgb, hrnir, lrnir)
                
                # EDSR_cat
                # sr = self.model([lrrgb, hrnir], 0)
                # EDSR
                sr = self.model(lrrgb, 0)
                
                sr = utility.quantize(sr, self.args.rgb_range)
                save_list = [sr]
                
                im1 = np.squeeze(sr.cpu().numpy()).transpose(1, 2, 0) * 255
                im2 = np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0) * 255
                psnr, _ = utility.psnr(im1, im2)
                psnr1 = utility.calc_psnr(
                    sr, hrrgb, self.scale[0], self.args.rgb_range, dataset=d
                )
                self.ckp.log[-1, idx_data, 0] += psnr
                # if self.args.save_gt:
                #     save_list.extend([lr, hr])
                
                if self.args.save_results:
                    self.ckp.save_results(d, filename[0], save_list, self.scale)
            
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.scale,
                    self.ckp.log[-1, idx_data, 0] / num,
                    best[0][idx_data, 0] / num,
                    best[1][idx_data, 0] + 1))
        
        #     for idx_scale, scale in enumerate(self.scale):
        #         d.dataset.set_scale(idx_scale)
        # for lr, hr, filename in tqdm(d, ncols=80):
        #     lr, hr = self.prepare(lr, hr)
        
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        
        if self.args.save_results:
            self.ckp.end_background()
        
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
        
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        
        torch.set_grad_enabled(True)
    
    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
        
        return [_prepare(a) for a in args]
    
    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs


class TrainerCV18():
    def __init__(self, args, loader, loaderTest, my_model, my_loss, ckp, server=1, argsv2=None):
        self.a = args.rgb_range / 255
        self.server = server
        self.maxpsnr = 0
        self.maxepoch = 0
        self.args = args
        self.scale = args.scale
        if argsv2:
            testset = argsv2.data_test[0]
        else:
            testset = self.args.data_test[0]
        if self.server == 1:
            root = './'
        else:
            root = 'E:/file/python_project/RGBNIRStereo/'
        if args.save:
            self.savepath = root + 'experiment/%s/%s/' % (args.save, self.args.modelname) + 'results-{}'.format(testset)
        else:
            self.savepath = root + 'experiment/test/%s/' % self.args.modelname + 'results-{}'.format(testset)
        os.makedirs(self.savepath, exist_ok=True)
        
        self.ckp = ckp
        self.loader_train = loader
        self.loader_test = loaderTest
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizerUDL(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        
        self.error_last = 1e8

    def testFeatureVis(self):
        os.makedirs('./result/feature/', exist_ok=True)
        torch.set_grad_enabled(False)
        self.model.eval()
    
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            if idx_data >= 1:
                exit()
            (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, rgb_exp, nir_exp) = d
            rgb = F.relu((raw_rgb - 2.0) / (255.0 - 2.0)).cuda()
            rgblr = F.relu((raw_rgblr - 2.0) / (255.0 - 2.0)).cuda()
            nir = F.relu((raw_nir - 2.0) / (255.0 - 2.0)).cuda()
            nirlr = F.relu((raw_nirlr - 2.0) / (255.0 - 2.0)).cuda()
            rgb_ratio = 0.5 / (rgb.mean(1).mean(1).mean(1) + 1e-3)
            nir_ratio = 0.5 / (nir.mean(1).mean(1).mean(1) + 1e-3)
            lrrgb = torch.clamp(rgblr * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            lrnir = torch.clamp(nirlr * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)  # [0, 1]

            # EDSR2branch
            if 'EDSR_cocrt_RGBNIRw2branch' in self.args.modelname:
                fsr, fnsr = self.model((lrrgb, lrnir), 0)  # [1, 256, h, w]
            fsr = np.squeeze(fsr.cpu().numpy()).transpose(1, 2, 0) * 255
            fnsr = np.squeeze(fnsr.cpu().numpy()).transpose(1, 2, 0) * 255
            for fi in range(fsr.shape[-1]):
                print(fi)
                im1 = np.uint8(fsr[:, :, fi])
                # # image_np = cv2.applyColorMap(im1, colormap=0)
                # # image_np1 = cv2.applyColorMap(im1, colormap=1)
                # image_np2 = cv2.applyColorMap(im1, colormap=2)
                # # image_np3 = cv2.applyColorMap(im1, colormap=3)
                # # image_np4 = cv2.applyColorMap(im1, colormap=4)
                # # cv2.imwrite('./result/feature/cvRGB-%d.png' % fi, image_np)
                # # cv2.imwrite('./result/feature/cvRGB1-%d.png' % fi, image_np1)
                # cv2.imwrite('./result/feature/cvRGB2-%d.png' % fi, image_np2)
                # # cv2.imwrite('./result/feature/cvRGB3-%d.png' % fi, image_np3)
                # # cv2.imwrite('./result/feature/cvRGB4-%d.png' % fi, image_np4)
                # # Image.fromarray(im1).save('./result/feature/RGB%d.png' % fi)
                savecolorim('./result/feature/cRGB%d.png' % fi, im1)
                
                imn1 = np.uint8(fnsr[:, :, fi])
                # Image.fromarray(imn1).save('./result/feature/NIR%d.png' % fi)
                savecolorim('./result/feature/NcIR%d.png' % fi, imn1)
                # image_np1 = cv2.applyColorMap(im1, colormap=0)
                # cv2.imwrite('./result/feature/cvNIR%d.png' % fi, image_np1)
                
                # if fi >= 10:
                #     exit()
            
    ##------------ light SR ------------
    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        num = 0
        meanps = 0
        meanpsb = 0
        mssb = 0
        mss = 0
        for idx_data, d in enumerate(self.loader_test):
            # d.dataset.set_scale(0)
            # for collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, rgb_exp, nir_exp in tqdm(d, ncols=80):
            (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, rgb_exp, nir_exp) = d
            num += 1
            # Preprocessing
            rgb = F.relu((raw_rgb - 2.0) / (255.0 - 2.0)).cuda()
            rgblr = F.relu((raw_rgblr - 2.0) / (255.0 - 2.0)).cuda()
            nir = F.relu((raw_nir - 2.0) / (255.0 - 2.0)).cuda()
            nirlr = F.relu((raw_nirlr - 2.0) / (255.0 - 2.0)).cuda()
            rgb_ratio = 0.5 / (rgb.mean(1).mean(1).mean(1) + 1e-3)
            nir_ratio = 0.5 / (nir.mean(1).mean(1).mean(1) + 1e-3)
            hrrgb = torch.clamp(rgb * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            lrrgb = torch.clamp(rgblr * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            hrnir = torch.clamp(nir * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            lrnir = torch.clamp(nirlr * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)  # [0, 1]

            # EDSR_cat [EDSR_cocrt_RGBNIR]
            if self.args.modelname == 'EDSR_cocrt_RGBNIR' or self.args.modelname == 'EDSR_cocrt_RGBNIRw':
                sr = self.model(torch.cat([lrrgb, lrnir], 1), 0)
            if self.args.modelname == 'EDSR_cocrt_RGB':
                sr = self.model(lrrgb, 0)
            # EDSR2branch
            elif 'EDSR_cocrt_RGBNIRw2branch' in self.args.modelname or '_cocrt_RGBNIR2branch1' in self.args.modelname:
                sr, nsr = self.model((lrrgb, lrnir), 0)
            if 'PASSR' in self.args.modelname:
                sr = self.model((lrrgb, lrnir), 0)
                
            sr = utility.quantize(sr, self.args.rgb_range)
            save_list = [sr]
            
            # im1 = np.squeeze(sr.cpu().numpy()).transpose(1, 2, 0) * 255
            im2 = np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0) * 255
            im2 = np.uint8(np.clip(im2, 0, 255))  # [0,255]
            im1 = np.uint8(np.squeeze(sr.cpu().numpy()).transpose(1, 2, 0) * 255)
            psnr, _ = utility.psnr(im1, im2)
            # psnr = utility.calc_psnr(
            #     im1, im2, self.scale[0], self.args.rgb_range, dataset=d)
            self.ckp.log[-1, idx_data, 0] += psnr
            
            size1 = tuple(((np.array(
                Image.fromarray(im2).size)).astype(
                int)).astype(int))
            lrnorm = 255 * np.clip(np.squeeze(lrrgb.cpu().numpy()).transpose(1, 2, 0) / 5 * 5, 0, 1)
            bc = np.uint8(np.array(
                Image.fromarray(np.uint8(lrnorm)).resize(size1, Image.BICUBIC)))  # [0, 49]
            psnrb, _ = utility.psnr(bc, im2)
            meanps += psnr
            meanpsb += psnrb
            ss = compare_ssim(im1, im2, multichannel=True)
            ssb = compare_ssim(bc, im2, multichannel=True)  # [0,49] [0,56]

            mss += ss
            mssb += ssb
            print(key, psnr, ss, 'Bicubic', psnrb, ssb)
            # self.ckp.log[-1, idx_data, 1] += psnr1
            # if self.args.save_gt:
            #     save_list.extend([lr, hr])
            if self.args.save_results:
                self.ckp.save_results(d, key[0], save_list, self.scale)
            
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.scale,
                    self.ckp.log[-1].sum() / num,
                    # self.ckp.log[-1, idx_data, 0],  # self.ckp.log[-1, idx_data, 1] / num,
                    best[0].sum() / num,  # best[0][idx_data, 0] / num,  #
                    best[1][idx_data, 0] + 1))
        
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        print('Mean PSNR for %d image = ' % num, meanps / num, mss/num, 'Bicubic', meanpsb/num, mssb/num)
        if meanps/num > self.maxpsnr:
            self.maxpsnr = meanps/num
            self.maxepoch = epoch
        self.ckp.write_log(
            'All Image [x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.scale,
                self.ckp.log[-1].sum() / num,
                self.maxpsnr, self.maxepoch))
        if self.args.save_results:
            self.ckp.end_background()
        
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(self.maxepoch == epoch))
            # self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
        
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)
        return num, meanps, mss
    
    def trainTW(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        
        timer_data, timer_model = utility.timer(), utility.timer()
        
        for batch, (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, rgb_exp, nir_exp) in enumerate(
                self.loader_train):
            # if batch >= 10:
            #     break
            
            # Preprocessing
            # print(raw_rgb.shape)
            rgb = F.relu((raw_rgb - 2.0) / (255.0 - 2.0)).cuda()
            rgblr = F.relu((raw_rgblr - 2.0) / (255.0 - 2.0)).cuda()
            nir = F.relu((raw_nir - 2.0) / (255.0 - 2.0)).cuda()
            nirlr = F.relu((raw_nirlr - 2.0) / (255.0 - 2.0)).cuda()
            rgb_ratio = 0.5 / (rgb.mean(1).mean(1).mean(1) + 1e-3)
            nir_ratio = 0.5 / (nir.mean(1).mean(1).mean(1) + 1e-3)
            hrrgb = torch.clamp(rgb * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            lrrgb = torch.clamp(rgblr * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            hrnir = torch.clamp(nir * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            lrnir = torch.clamp(nirlr * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)  # [0, 1]

            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            # EDSR_cat
            if self.args.modelname == 'EDSR_cocrt_RGBNIR' or self.args.modelname == 'EDSR_cocrt_RGBNIRw':
                sr = self.model(torch.cat([lrrgb, lrnir], 1), 0)
                loss = self.loss(sr, hrrgb)
            # EDSR
            elif self.args.modelname == 'EDSR_cocrt_RGB':
                sr = self.model(lrrgb, 0)
                loss = self.loss(sr, hrrgb)
            # EDSR2branch  [0, 1, '']
            elif 'EDSR_cocrt_RGBNIRw2branch' in self.args.modelname or '_cocrt_RGBNIR2branch1' in self.args.modelname:
                sr, nsr = self.model((lrrgb, lrnir), 0)
                loss = self.loss(sr, hrrgb) + 0.1 * self.loss(nsr, hrnir)
            elif 'EDSR_cocrt_RGBNIRw2HRbranch' in self.args.modelname:
                sr, nsr = self.model((lrrgb, hrnir), 0)
                loss = self.loss(sr, hrrgb) + 0.2 * self.loss(nsr, hrnir)
            if 'PASSR' in self.args.modelname:
                sr = self.model((lrrgb, lrnir), 0)
                loss = self.loss(sr, hrrgb)

            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            
            timer_model.hold()
            
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            
            timer_data.tic()
        
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        
    ## Read in RGBCorrect; 不用训练的时候矫正颜色，加快训练
    def testReadCorrect(self):
        saveVar = True
        torch.set_grad_enabled(False)
        epoch = self.scheduler.last_epoch
        # epoch = self.optimizer.get_last_epoch()
        print('Test epoch', epoch)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        num = 0
        meanps = 0
        meanmae = 0
        meanmaeb = 0
        meanpsb = 0
        mssb = 0
        mss = 0
        for idx_data, d in enumerate(self.loader_test):
            (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr) = d
            num += 1
            if self.args.cpu:
                hrrgb = torch.clamp(((raw_rgb) * self.a), 0.0, self.args.rgb_range)
                lrrgb = torch.clamp(((raw_rgblr) * self.a), 0.0, self.args.rgb_range)
                lrnir = torch.clamp(((raw_nirlr) * self.a), 0.0, self.args.rgb_range)  # [0, 1]
                hrnir = torch.clamp(((raw_nir) * self.a), 0.0, self.args.rgb_range)  # [0, 1]
            else:
                hrrgb = torch.clamp(((raw_rgb) * self.a).cuda(), 0.0, self.args.rgb_range)
                lrrgb = torch.clamp(((raw_rgblr) * self.a).cuda(), 0.0, self.args.rgb_range)
                lrnir = torch.clamp(((raw_nirlr) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]
                hrnir = torch.clamp(((raw_nir) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]
                
            # EDSR2branch
            if 'depthSR' in self.args.modelname:
                if 'depthrgbdepthSR_w2branch1' == self.args.modelname:
                    sr, nsr = self.model((lrrgb, lrnir), 0)

                nsr = utility.quantize(nsr, self.args.rgb_range)
                save_list = [nsr]
                im2 = np.squeeze(hrnir.cpu().numpy()) * 255
                im2 = np.uint8(np.clip(im2, 0, 255))  # [0,255]
                im1 = np.uint8(np.squeeze(nsr.cpu().numpy()) * 255)

                mae = mean_absolute_error(im1, im2)
                meanmae += mae
                mse = mean_squared_error(im1, im2)
                psnr = mse  # psnr = utility.Rmse(im1, im2)
                self.ckp.log[-1, idx_data, 0] += psnr
            else:
                if ('HAT' in self.args.modelname) or ('depthrgbleftSR_w2branch1' == self.args.modelname):
                    if 'SISR' in self.args.modelname:
                        sr = self.model(lrrgb, 0)
                    else:
                        sr, nsr = self.model((lrrgb, lrnir), 0)
                elif 'depthrgb_uncertainty' == self.args.modelname or 'depthrgb_uncertainty_elu' == self.args.modelname \
                        or 'EDSRcocrtinNIRw_uncertainty_elu' == self.args.modelname:
                    sr1, theta1, sr2, theta2, sr, theta3, nsr = self.model((lrrgb, lrnir), 0)
                    th_num = 1
                    for theta in [theta1, theta2, theta3]:
                        if self.args.cpu:
                            imtheta = (np.expand_dims(np.squeeze(theta.numpy()).transpose(1, 2, 0) - np.min(theta.numpy()), -1)) * 10
                        else:
                            imtheta = (np.expand_dims(
                                np.squeeze(theta.cpu().numpy()).transpose(1, 2, 0) - np.min(theta.cpu().numpy()),
                                -1)) * 10  # * 255  # [-2763, -44]
                        imthetac = cv2.applyColorMap(imtheta.astype(np.uint8), cv2.COLORMAP_JET)
                        cv2.imwrite('./experiment/test/%s/results-test/Un%d' % (self.args.modelname, th_num) + key[0] + '.png',
                                    imthetac)
                        th_num += 1
                        if saveVar:
                            var = theta
                            if var.size(1) > 1:
                                convert = var.new(1, 3, 1, 1)
                                convert[0, 0, 0, 0] = 65.738
                                convert[0, 1, 0, 0] = 129.057
                                convert[0, 2, 0, 0] = 25.064
                                var.mul_(convert).div_(256)
                                var = var.sum(dim=1, keepdim=True)
                            else:
                                var = None
                            if self.args.cpu:
                                utility.draw_features(var.numpy(), "./experiment/test/%s/results-test/%s_var%d.png" % (
                                self.args.modelname, key[0], th_num))
                            else:
                                utility.draw_features(var.cpu().numpy(), "./experiment/test/%s/results-test/%s_var%d.png" % (
                                self.args.modelname, key[0], th_num))

                    print('Min/Max', np.min(theta3.cpu().numpy()), np.max(theta3.cpu().numpy()))  # [-1, -1]
                    continue
                    # Image.fromarray(np.uint8(np.clip(imtheta, 0, 255))).save('./experiment/test/%s/results-test/' % self.args.modelname+key[0][:-4] + '.png')
                elif 'RGBNIRw2branch' in self.args.modelname or 'RGBNIR2branch1' in self.args.modelname \
                        or 'Dynamic_nirscene' in self.args.modelname:
                    sr, nsr = self.model((lrrgb, lrnir), 0)
                elif 'EDSRcocrtinNIRw_uncertainty' in self.args.modelname or 'EDSRcocrtinNIRw_UDL' in self.args.modelname:
                    if 'EDSRcocrtinNIRw_UDL' in self.args.modelname:
                        sr, theta = self.model((lrrgb, lrnir), 0)
                    else:
                        sr1, sr2, sr, theta, nsr = self.model((lrrgb, lrnir), 0)

                    imtheta = (np.expand_dims(np.squeeze(theta.cpu().numpy()).transpose(1, 2, 0) - np.min(theta.cpu().numpy()), -1)) * 10  # * 255  # [-2763, -44]
                    imthetac = cv2.applyColorMap(imtheta.astype(np.uint8), cv2.COLORMAP_JET)
                    cv2.imwrite('./experiment/test/%s/results-test/Un' % (self.args.modelname) + key[0] + '.png',
                                imthetac)
                    print('Min/Max', np.min(theta.cpu().numpy()), np.max(theta.cpu().numpy()))  # [-1, -1]

                    var = theta
                    if var.size(1) > 1:
                        convert = var.new(1, 3, 1, 1)
                        convert[0, 0, 0, 0] = 65.738
                        convert[0, 1, 0, 0] = 129.057
                        convert[0, 2, 0, 0] = 25.064
                        var.mul_(convert).div_(256)
                        var = var.sum(dim=1, keepdim=True)
                    else:
                        var = None
                    if self.args.cpu:
                        utility.draw_features(var.numpy(), "./experiment/test/%s/results-test/%s_var.png" % (self.args.modelname, key[0]))
                    else:
                        utility.draw_features(var.cpu().numpy(), "./experiment/test/%s/results-test/%s_var.png" % (self.args.modelname, key[0]))
                elif 'SISR' in self.args.modelname:
                    if 'cat' in self.args.modelname:
                        sr = self.model((lrrgb, lrnir), 0)
                    else:
                        sr = self.model((lrrgb), 0)

                sr = utility.quantize(sr, self.args.rgb_range)
                save_list = [sr]
                if self.args.cpu:
                    im2 = np.squeeze(hrrgb.numpy()).transpose(1, 2, 0) / self.a
                    im1 = np.uint8(np.squeeze(sr.numpy()).transpose(1, 2, 0) / self.a)
                else:
                    im2 = np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0) / self.a
                    im1 = np.uint8(np.squeeze(sr.cpu().numpy()).transpose(1, 2, 0) / self.a)
                im2 = np.uint8(np.clip(im2, 0, 255))  # [0,255]
                psnr, _ = utility.psnr(im1, im2)
                self.ckp.log[-1, idx_data, 0] += psnr
            
            size1 = tuple(np.array(Image.fromarray(im2).size).astype(int))
            if 'depthSR' in self.args.modelname:
                if self.args.cpu:
                    lrnorm = np.clip(np.squeeze(lrnir.numpy()), 0, self.args.rgb_range) / self.a
                else:
                    lrnorm = np.clip(np.squeeze(lrnir.cpu().numpy()), 0, self.args.rgb_range) / self.a
                bc = np.uint8(np.array(
                    Image.fromarray(np.uint8(lrnorm)).resize(size1, Image.BICUBIC)))  # [0, 49]
                maeb = utility.mae(bc, im2)  # mean_absolute_error(bc, im2)
                meanmaeb += maeb
                psnrb = utility.mse(bc, im2)  # mean_squared_error(bc, im2)  # [0, 255]
            else:
                if self.args.cpu:
                    lrnorm = np.clip(np.squeeze(lrrgb.numpy()).transpose(1, 2, 0), 0, self.args.rgb_range) / self.a
                else:
                    lrnorm = np.clip(np.squeeze(lrrgb.cpu().numpy()).transpose(1, 2, 0), 0, self.args.rgb_range) / self.a
                bc = np.uint8(np.array(Image.fromarray(np.uint8(lrnorm)).resize(size1, Image.BICUBIC)))  # [0, 49]
                psnrb, _ = utility.psnr(bc, im2)
            ss = compare_ssim(im1, im2, multichannel=True)
            meanps += psnr
            mss += ss
            
            ssb = compare_ssim(bc, im2, multichannel=True)  # [0,49] [0,56]
            meanpsb += psnrb
            mssb += ssb
            # print(key, 'Bicubic', psnrb, maeb, ssb)
            print(key, psnr, ss, 'Bicubic', psnrb, ssb)
            if self.args.save_results:
                self.ckp.save_results(d, key[0], save_list, self.scale)
        
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.scale,
                    self.ckp.log[-1].sum() / num,
                    best[0].sum() / num,  # best[0][idx_data, 0] / num,  #
                    best[1][idx_data, 0] + 1))
    
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if 'depthSR' in self.args.modelname:
            print('Mean MSE/MAE for %d image = ' % num, meanps / num, meanmae / num, mss / num,
                  'Bicubic', meanpsb / num, meanmaeb / num, mssb / num)
        else:
            print('Mean PSNR for %d image = ' % num, meanps / num, mss / num, 'Bicubic', meanpsb / num, mssb / num)
        if meanps / num > self.maxpsnr:
            self.maxpsnr = meanps / num
            self.maxepoch = epoch
        self.ckp.write_log(
            'All Image [x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.scale,
                self.ckp.log[-1].sum() / num,
                self.maxpsnr, self.maxepoch))
        if self.args.save_results:
            self.ckp.end_background()
    
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(self.maxepoch == epoch))
    
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)
        return num, meanps, mss

    def testReadCorrectSISR(self, savepath):
        torch.set_grad_enabled(False)
        self.model.eval()
        self.model.cuda()
        num = 0
        meanps = 0
        meanpsb = 0
        mssb = 0
        mss = 0
        for idx_data, d in enumerate(self.loader_test):
            (raw_rgb, raw_rgblr, raw_nir, raw_nirlr, filename) = d
            num += 1
            hrrgb = torch.clamp(((raw_rgb) * self.a).cuda(), 0.0, self.args.rgb_range)
            lrrgb = torch.clamp(((raw_rgblr) * self.a).cuda(), 0.0, self.args.rgb_range)
            lrnir = torch.clamp(((raw_nirlr) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]

            # HAT_cocrtin_NIRw
            sr, nsr = self.model((lrrgb, lrnir), 0)
            # print('lrrgb.shape, lrnir.shape, sr.shape = ', lrrgb.shape, lrnir.shape, sr.shape)
            # torch.Size([1, 3, 64, 64]) torch.Size([1, 3, 64, 64]) torch.Size([1, 3, 256, 256])

            sr = utility.quantize(sr, self.args.rgb_range)  # 0-1
            im2 = np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0) / self.a
            im1 = np.uint8(np.squeeze(sr.cpu().numpy()).transpose(1, 2, 0) / self.a)
            im2 = np.uint8(np.clip(im2, 0, 255))  # [0,255]
                
            psnr, _ = utility.psnr(im1, im2)
            ss = compare_ssim(im1, im2, multichannel=True)
            meanps += psnr
            mss += ss
            
            size1 = tuple(np.array(Image.fromarray(im2).size).astype(int))
            lrnorm = np.clip(np.squeeze(lrrgb.cpu().numpy()).transpose(1, 2, 0), 0, self.args.rgb_range) / self.a
            bc = np.uint8(np.array(Image.fromarray(np.uint8(lrnorm)).resize(size1, Image.BICUBIC)))  # [0, 49]
            psnrb, _ = utility.psnr(bc, im2)
            ssb = compare_ssim(bc, im2, multichannel=True)  # [0,49] [0,56]
            meanpsb += psnrb
            mssb += ssb
            print(filename, psnr, ss, 'Bicubic', psnrb, ssb)
            saveto = '{}{}.png'.format(savepath, filename[:-4])
            imageio.imwrite(saveto, im1)

        print('Mean PSNR for %d image = ' % num, meanps / num, mss / num, 'Bicubic', meanpsb / num, mssb / num)
        torch.set_grad_enabled(True)
        return num, meanps, mss

    def trainTWReadCorrect(self):
        # self.loss.step()
        # epoch = self.scheduler.last_epoch  # + 1
        # self.optimizer.set_last_epoch(ep=epoch)  # epoch = self.args.resume + 1
        # print('Start Epoch', epoch)
        # lr = self.optimizer.get_lr()  # lr = 1e-4  #
        # self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        # self.loss.start_log()
        # self.model.train()

        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
    
        timer_data, timer_model = utility.timer(), utility.timer()
    
        for batch, (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr) in enumerate(self.loader_train):
            # Preprocessing
            if self.args.cpu:
                hrrgb = torch.clamp(((raw_rgb) * self.a), 0.0, self.args.rgb_range)
                lrrgb = torch.clamp(((raw_rgblr) * self.a), 0.0, self.args.rgb_range)
                lrnir = torch.clamp(((raw_nirlr) * self.a), 0.0, self.args.rgb_range)  # [0, 1]
                hrnir = torch.clamp(((raw_nir) * self.a), 0.0, self.args.rgb_range)  # [0, 1]
            else:
                hrrgb = torch.clamp(((raw_rgb) * self.a).cuda(), 0.0, self.args.rgb_range)
                lrrgb = torch.clamp(((raw_rgblr) * self.a).cuda(), 0.0, self.args.rgb_range)
                lrnir = torch.clamp(((raw_nirlr) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]
                hrnir = torch.clamp(((raw_nir) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]
        
            timer_data.hold()
            timer_model.tic()
        
            self.optimizer.zero_grad()
            if 'depthSR' in self.args.modelname:
                if 'depthrgbdepthSR_w2branch1' == self.args.modelname:
                    sr, nsr = self.model((lrrgb, lrnir), 0)
                    loss = 0.1 * self.loss(sr, hrrgb) + self.loss(nsr, hrnir)
            else:
                if 'HAT' in self.args.modelname:
                    if 'SISR' in self.args.modelname:
                        sr = self.model(lrrgb, 0)
                    else:
                        sr, nsr = self.model((lrrgb, lrnir), 0)
                    # loss = self.loss(sr, hrrgb) + 0.01 * self.loss(nsr, hrnir)
                    loss = self.loss(sr, hrrgb)  # + 0.0001 * self.loss(nsr, hrnir)
                elif 'depthrgb_uncertainty_elu' == self.args.modelname or 'EDSRcocrtinNIRw_uncertainty_elu' == self.args.modelname:
                    sr1, theta1, sr2, theta2, sr3, theta3, nsr = self.model((lrrgb, lrnir), 0)
                    lossuncer = torch.mean(torch.exp(-theta1) * (sr1 - hrrgb).pow(2)) + torch.mean(2 * theta1)
                    lossuncer += torch.mean(torch.exp(-theta2) * (sr2 - hrrgb).pow(2)) + torch.mean(2 * theta2)
                    lossuncer += torch.mean(torch.exp(-theta3) * (sr3 - hrrgb).pow(2)) + torch.mean(2 * theta3)
                    loss = torch.mean((sr3 - hrrgb).pow(2)) + 0.1 * torch.mean((nsr - hrnir).pow(2)) + lossuncer  #
                elif 'EDSRcocrtinNIRw_UDL' in self.args.modelname or 'EDSRcocrtinNIRw_uncertainty_twostage' == self.args.modelname\
                        or 'EDSRcocrtinNIRw_uncertainty_stage1' == self.args.modelname \
                        or 'EDSRcocrtinNIRw_uncertainty_twostage_f' == self.args.modelname:
                    if 'EDSRcocrtinNIRw_UDL' in self.args.modelname:
                        sr3, theta = self.model((lrrgb, lrnir), 0)
                    else:
                        sr1, sr2, sr3, theta, nsr = self.model((lrrgb, lrnir), 0)
                    if 'stage1' in self.args.modelname:
                        s = torch.exp(-theta)  # [2, 3, 256, 256]
                        sr_ = torch.mul(sr3, s)  # [2, 3, 256, 256]
                        hr_ = torch.mul(hrrgb, s)  # [2, 3, 256, 256]
                        loss = nn.L1Loss()(sr_, hr_) + 2 * torch.mean(theta)
                    else:
                        b, c, h, w = theta.shape
                        s1 = theta.view(b, c, -1)
                        pmin = torch.min(s1, dim=-1)
                        pmin = pmin[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
                        s = theta
                        s = s - pmin + 1
                        sr_ = torch.mul(sr3, s)
                        hr_ = torch.mul(hrrgb, s)
                        loss = nn.L1Loss()(sr_, hr_) + nn.L1Loss()(nsr, hrnir)
                elif 'RGBNIRw2branch' in self.args.modelname or 'RGBNIR2branch1' in self.args.modelname \
                        or 'depthrgb' in self.args.modelname or 'Dynamic_nirscene' in self.args.modelname:
                    sr, nsr = self.model((lrrgb, lrnir), 0)
                    loss = self.loss(sr, hrrgb) + 0.01 * self.loss(nsr, hrnir)
                elif 'EDSR_cocrt_RGBNIRw2HRbranch' in self.args.modelname:
                    sr, nsr = self.model((lrrgb, hrnir), 0)
                    loss = self.loss(sr, hrrgb) + 0.2 * self.loss(nsr, hrnir)
                if 'SISR' in self.args.modelname:
                    if 'cat' in self.args.modelname:
                        sr, _ = self.model((lrrgb, lrnir), 0)
                        loss = self.loss(sr, hrrgb)
                    else:
                        sr, _ = self.model((lrrgb), 0)
                        loss = self.loss(sr, hrrgb)
                    
            loss.backward()
            if self.args.gclip > 0:
                print('self.args.gclip > 0', self.args.gclip > 0)
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
        
            timer_model.hold()
        
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
        
            timer_data.tic()
    
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()  # self.optimizer.schedule()

    def trainMiddlebury(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
    
        timer_data, timer_model = utility.timer(), utility.timer()
    
        for batch, d in enumerate(self.loader_train):
            (hrrgb, hrnir, lrnir, lrrgb, mask_hr, mask_lr, im_index, nir_bicubic) = d
            # # Preprocessing
            # if self.args.cpu:
            #     hrrgb = torch.clamp(((raw_rgb) * self.a), 0.0, self.args.rgb_range)
            #     lrrgb = torch.clamp(((raw_rgblr) * self.a), 0.0, self.args.rgb_range)
            #     lrnir = torch.clamp(((raw_nirlr) * self.a), 0.0, self.args.rgb_range)  # [0, 1]
            #     hrnir = torch.clamp(((raw_nir) * self.a), 0.0, self.args.rgb_range)  # [0, 1]
            # else:
            #     hrrgb = torch.clamp(((raw_rgb) * self.a).cuda(), 0.0, self.args.rgb_range)
            #     lrrgb = torch.clamp(((raw_rgblr) * self.a).cuda(), 0.0, self.args.rgb_range)
            #     lrnir = torch.clamp(((raw_nirlr) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]
            #     hrnir = torch.clamp(((raw_nir) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]
        
            timer_data.hold()
            timer_model.tic()
        
            self.optimizer.zero_grad()
            if 'HAT' in self.args.modelname:
                dsr, rgbsr = self.model((lrnir, lrrgb), 0)
                loss = self.loss(dsr, hrnir) + 0.001 * self.loss(rgbsr, hrrgb)
                
            loss.backward()
            if self.args.gclip > 0:
                print('self.args.gclip > 0', self.args.gclip > 0)
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
        
            timer_model.hold()
        
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
        
            timer_data.tic()
    
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()  # self.optimizer.schedule()

    def testMiddlebury(self):
        torch.set_grad_enabled(False)
        epoch = self.scheduler.last_epoch
        # epoch = self.optimizer.get_last_epoch()
        print('Test epoch', epoch)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
    
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        num = 0
        meanps = 0
        meanmae = 0
        meanmaeb = 0
        meanpsb = 0
        mssb = 0
        mss = 0
        for idx_data, d in enumerate(self.loader_test):
            (hrrgb, hrnir, lrnir, lrrgb, mask_hr, mask_lr, im_index, nir_bicubic) = d
            num += 1
            im2 = np.squeeze(hrnir.cpu().numpy())

            # if 'HAT' in self.args.modelname:
            #     nsr, sr = self.model((lrnir, lrrgb), 0)
            # nsr = utility.quantize(nsr, self.args.rgb_range)
            # im1 = np.uint8(np.squeeze(nsr.cpu().numpy()) * 255)

            # mae = mean_absolute_error(im1, im2)
            # meanmae += mae
            # mse = mean_squared_error(im1, im2)
            # psnr = mse  # psnr = utility.Rmse(im1, im2)
            # self.ckp.log[-1, idx_data, 0] += psnr

            bc = np.squeeze(nir_bicubic.cpu().numpy())
            maeb = utility.mae(bc, im2)  # mean_absolute_error(bc, im2)
            # l1_loss = l1_loss_func(nir_bicubic, hrnir, mask_hr)
            # meanmaeb += l1_loss  # maeb
            psnrb = utility.mse(bc, im2)  # mean_squared_error(bc, im2)  # [0, 255]
            # l2_loss = mse_loss_func(nir_bicubic, hrnir, mask_hr)

            # ss = compare_ssim(im1, im2, multichannel=True)
            # meanps += psnr
            # mss += ss
        
            ssb = compare_ssim(bc, im2, multichannel=True)  # [0,49] [0,56]
            # meanpsb += l2_loss  # psnrb
            mssb += ssb
            print('Bicubic', psnrb, maeb, ssb)
            # print(key, psnr, ss, 'Bicubic', psnrb, ssb)
            # if self.args.save_results:
            #     self.ckp.save_results(d, key[0], [sr], self.scale)
        
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.scale,
                    self.ckp.log[-1].sum() / num,
                    best[0].sum() / num,  # best[0][idx_data, 0] / num,  #
                    best[1][idx_data, 0] + 1))
    
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        if 'depthSR' in self.args.modelname:
            print('Mean MSE/MAE for %d image = ' % num, meanps / num, meanmae / num, mss / num,
                  'Bicubic', meanpsb / num, meanmaeb / num, mssb / num)
        else:
            print('Mean PSNR for %d image = ' % num, meanps / num, mss / num, 'Bicubic', meanpsb / num, mssb / num)
        if meanps / num > self.maxpsnr:
            self.maxpsnr = meanps / num
            self.maxepoch = epoch
        self.ckp.write_log(
            'All Image [x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.scale,
                self.ckp.log[-1].sum() / num,
                self.maxpsnr, self.maxepoch))
        if self.args.save_results:
            self.ckp.end_background()
    
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(self.maxepoch == epoch))
    
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)
        return num, meanps, mss

    def testMMNet_ReadCorrect(self):
        torch.set_grad_enabled(False)
        epoch = self.scheduler.last_epoch
        # epoch = self.optimizer.get_last_epoch()
        print('Test epoch', epoch)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        num = 0
        meanps = 0
        meanpsb = 0
        mssb = 0
        mss = 0
        for idx_data, d in enumerate(self.loader_test):
            (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr) = d
            num += 1
            # Preprocessing
            if self.args.cpu:
                hrrgb = torch.clamp(((raw_rgb) * self.a), 0.0, self.args.rgb_range)
                lrrgb = torch.clamp(((raw_rgblr) * self.a), 0.0, self.args.rgb_range)
                lrnir = torch.clamp(((raw_nirlr) * self.a), 0.0, self.args.rgb_range)  # [0, 1]
                hrnir = torch.clamp(((raw_nir) * self.a), 0.0, self.args.rgb_range)  # [0, 1]
            else:
                hrrgb = torch.clamp(((raw_rgb) * self.a).cuda(), 0.0, self.args.rgb_range)
                lrrgb = torch.clamp(((raw_rgblr) * self.a).cuda(), 0.0, self.args.rgb_range)
                lrnir = torch.clamp(((raw_nirlr) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]
                hrnir = torch.clamp(((raw_nir) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]

            bicnir = F.interpolate(lrnir, [hrrgb.shape[2], hrrgb.shape[3]], mode='bilinear', align_corners=False)

            self.optimizer.zero_grad()
            if 'HRnir' in self.args.modelname:
                sr = self.model((hrnir, lrrgb), 0)
            else:
                sr = self.model((bicnir, lrrgb), 0)
            # sr = F.interpolate(lrrgb, [hrrgb.shape[2], hrrgb.shape[3]], mode='bilinear', align_corners=False)

            sr = utility.quantize(sr, self.args.rgb_range)
            save_list = [sr]
            if self.args.cpu:
                im2 = np.squeeze(hrrgb.numpy()).transpose(1, 2, 0) / self.a
                im1 = np.uint8(np.squeeze(sr.numpy()).transpose(1, 2, 0) / self.a)
            else:
                im2 = np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0) / self.a
                im1 = np.uint8(np.squeeze(sr.cpu().numpy()).transpose(1, 2, 0) / self.a)
            im2 = np.uint8(np.clip(im2, 0, 255))  # [0,255]
            psnr, _ = utility.psnr(im1, im2)
            self.ckp.log[-1, idx_data, 0] += psnr

            size1 = tuple(((np.array(
                Image.fromarray(im2).size)).astype(
                int)).astype(int))
            if self.args.cpu:
                lrnorm = np.clip(np.squeeze(lrrgb.numpy()).transpose(1, 2, 0), 0, self.args.rgb_range) / self.a
            else:
                lrnorm = np.clip(np.squeeze(lrrgb.cpu().numpy()).transpose(1, 2, 0), 0,
                                 self.args.rgb_range) / self.a
            bc = np.uint8(np.array(Image.fromarray(np.uint8(lrnorm)).resize(size1, Image.BICUBIC)))  # [0, 49]
            psnrb, _ = utility.psnr(bc, im2)
            
            meanps += psnr
            meanpsb += psnrb
            ss = compare_ssim(im1, im2, multichannel=True)
            ssb = compare_ssim(bc, im2, multichannel=True)  # [0,49] [0,56]

            mss += ss
            mssb += ssb
            # print(key, psnr, ss, 'Bicubic', psnrb, ssb)
            if self.args.save_results:
                self.ckp.save_results(d, key[0], save_list, self.scale)
        
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.scale,
                    self.ckp.log[-1].sum() / num,
                    best[0].sum() / num,  # best[0][idx_data, 0] / num,  #
                    best[1][idx_data, 0] + 1))
    
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        print('Mean PSNR for %d image = ' % num, meanps / num, mss / num, 'Bicubic', meanpsb / num, mssb / num)
        if meanps / num > self.maxpsnr:
            self.maxpsnr = meanps / num
            self.maxepoch = epoch
        self.ckp.write_log(
            'All Image [x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.scale,
                self.ckp.log[-1].sum() / num,
                self.maxpsnr, self.maxepoch))
        if self.args.save_results:
            self.ckp.end_background()
    
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(self.maxepoch == epoch))
    
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)
        return num, meanps, mss

    def trainMMNet_ReadCorrect(self):
        if IS_TF_1:
            print_every = 1
        else:
            print_every = self.args.print_every
        self.scheduler.step()
        self.loss.step()
        self.epoch = epoch = self.scheduler.last_epoch
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
    
        timer_data, timer_model = utility.timer(), utility.timer()
    
        for batch, (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr) in enumerate(self.loader_train):
            # Preprocessing
            if self.args.cpu:
                hrrgb = torch.clamp(((raw_rgb) * self.a), 0.0, self.args.rgb_range)
                lrrgb = torch.clamp(((raw_rgblr) * self.a), 0.0, self.args.rgb_range)
                lrnir = torch.clamp(((raw_nirlr) * self.a), 0.0, self.args.rgb_range)  # [0, 1]
                hrnir = torch.clamp(((raw_nir) * self.a), 0.0, self.args.rgb_range)  # [0, 1]
            else:
                hrrgb = torch.clamp(((raw_rgb) * self.a).cuda(), 0.0, self.args.rgb_range)
                lrrgb = torch.clamp(((raw_rgblr) * self.a).cuda(), 0.0, self.args.rgb_range)
                lrnir = torch.clamp(((raw_nirlr) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]
                hrnir = torch.clamp(((raw_nir) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]
    
            bicnir = F.interpolate(lrnir, [hrrgb.shape[2], hrrgb.shape[3]], mode='bilinear', align_corners=False)
            timer_data.hold()
            timer_model.tic()
    
            self.optimizer.zero_grad()
            if 'HRnir' in self.args.modelname:
                sr = self.model((hrnir, lrrgb), 0)
            else:
                sr = self.model((bicnir, lrrgb), 0)
            loss = torch.sum(torch.abs(sr - hrrgb))
            # loss = torch.mean((sr - hrrgb).pow(2))
        
            loss.backward()
            if self.args.gclip > 0:
                print('self.args.gclip > 0', self.args.gclip > 0)
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
        
            timer_model.hold()
            
            if (batch + 1) % print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset), loss,  # self.loss.display_loss(batch),
                    timer_model.release(), timer_data.release()))
        
            timer_data.tic()
    
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()  # self.optimizer.schedule()
        self.ckp.save(self, epoch, is_best=False)

    ## VISNIR
    def trainMMNet_VISNIR(self):
        if IS_TF_1:
            printevery = 1
        else:
            printevery = self.args.print_every
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + self.args.resume
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
    
        for batch, (raw_rgb, raw_rgblr, raw_nir, raw_nirlr, key, dir) in enumerate(self.loader_train):
            # Preprocessing
            # print(raw_rgb.max(), raw_rgb.max())
            if not self.args.cpu:
                hrrgb = raw_rgb.cuda()
                lrrgb = raw_rgblr.cuda()
                hrnir = raw_nir.cuda()
                lrnir = raw_nirlr.cuda()
            else:
                hrrgb = raw_rgb
                lrrgb = raw_rgblr
                lrnir = raw_nirlr
                hrnir = raw_nir
            # print('hrrgb.max(), min()', hrrgb.max(), hrrgb.min())
            # exit()

            bicnir = F.interpolate(lrnir, [hrrgb.shape[2], hrrgb.shape[3]], mode='bilinear', align_corners=False)
            timer_data.hold()
            timer_model.tic()
    
            self.optimizer.zero_grad()
            if 'HRnir' in self.args.modelname:
                sr = self.model((hrnir, lrrgb), 0)
            else:
                sr = self.model((bicnir, lrrgb), 0)
            loss = torch.mean((sr - hrrgb).pow(2))
            
            loss.backward()
            if self.args.gclip > 0:
                print('self.args.gclip > 0', self.args.gclip > 0)
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            timer_model.hold()
            
            if (batch + 1) % printevery == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    loss,  # self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()  # self.optimizer.schedule()
        self.ckp.save(self, epoch, is_best=False)

    def testMMNet_VISNIR(self):
        torch.set_grad_enabled(False)
        epoch = self.scheduler.last_epoch
        print('Test epoch', epoch)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
    
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        num = 0
        meanps = 0
        meanpsb = 0
        mssb = 0
        mss = 0
        for idx_data, d in enumerate(self.loader_test):
            (raw_rgb, raw_rgblr, raw_nir, raw_nirlr, key, dir) = d
            num += 1
            
            if not self.args.cpu:
                hrrgb = raw_rgb.cuda()
                lrrgb = raw_rgblr.cuda()
                hrnir = raw_nir.cuda()
                lrnir = raw_nirlr.cuda()
            else:
                hrrgb = raw_rgb
                lrrgb = raw_rgblr
                lrnir = raw_nirlr
                hrnir = raw_nir

            bicnir = F.interpolate(lrnir, [hrrgb.shape[2], hrrgb.shape[3]], mode='bilinear', align_corners=False)

            self.optimizer.zero_grad()
            if 'HRnir' in self.args.modelname:
                sr = self.model((hrnir, lrrgb), 0)
            else:
                sr = self.model((bicnir, lrrgb), 0)
            # sr = F.interpolate(lrrgb, [hrrgb.shape[2], hrrgb.shape[3]], mode='bilinear', align_corners=False)
            # print(sr.max(), lrrgb.max())

            sr = utility.quantize(sr, self.args.rgb_range)
            # print(sr.max())
            if self.args.cpu:
                im2 = np.squeeze(hrrgb.numpy()).transpose(1, 2, 0) * (255 / self.args.rgb_range)
                im1 = np.uint8(np.squeeze(sr.numpy()).transpose(1, 2, 0) * (255 / self.args.rgb_range))
            else:
                im2 = np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0) * (255 / self.args.rgb_range)
                im1 = np.uint8(np.squeeze(sr.cpu().numpy()).transpose(1, 2, 0) * (255 / self.args.rgb_range))
            im2 = np.uint8(np.clip(im2, 0, 255))  # [0,255]
            psnr, _ = utility.psnr(im1, im2)
            self.ckp.log[-1, idx_data, 0] += psnr
            if self.args.save_results:
                self.savedir = os.path.join(self.savepath, dir[0])
                os.makedirs(self.savedir, exist_ok=True)
                print('save to', self.savedir + '/{}.png'.format(key[0]))
                imageio.imwrite(self.savedir + '/{}.png'.format(key[0]), im1)
        
            size1 = tuple(((np.array(
                Image.fromarray(im2).size)).astype(
                int)).astype(int))
        
            if self.args.cpu:
                lrnorm = np.clip(np.squeeze(lrrgb.numpy()).transpose(1, 2, 0), 0, self.args.rgb_range) * (
                            255 / self.args.rgb_range)
            else:
                lrnorm = np.clip(np.squeeze(lrrgb.cpu().numpy()).transpose(1, 2, 0), 0,
                                 self.args.rgb_range) * (255 / self.args.rgb_range)
            bc = np.uint8(np.array(Image.fromarray(np.uint8(lrnorm)).resize(size1, Image.BICUBIC)))  # [0, 49]
            psnrb, _ = utility.psnr(bc, im2)
            meanps += psnr
            meanpsb += psnrb
            try:
                ss = compare_ssim(im1, im2, multichannel=True)
                ssb = compare_ssim(bc, im2, multichannel=True)  # [0,49] [0,56]
            except:
                ss = compare_ssim(im1, im2, channel_axis=3)
                ssb = compare_ssim(bc, im2, channel_axis=3)
        
            mss += ss
            mssb += ssb
            print(key, psnr, ss, 'Bicubic', psnrb, ssb)
        
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.scale,
                    self.ckp.log[-1].sum() / num,
                    best[0].sum() / num,
                    best[1][idx_data, 0] + 1))
    
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        print('Mean PSNR for %d image = ' % num, meanps / num, mss / num, 'Bicubic', meanpsb / num, mssb / num)
        if meanps / num > self.maxpsnr:
            self.maxpsnr = meanps / num
            self.maxepoch = epoch
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=True)
        self.ckp.write_log(
            'All Image [x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.scale,
                self.ckp.log[-1].sum() / num,
                self.maxpsnr, self.maxepoch))
        if self.args.save_results:
            self.ckp.end_background()
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(self.maxepoch == epoch))
        torch.set_grad_enabled(True)
        return num, meanps, mss

    ## VISNIR
    def testVISNIR(self):
        torch.set_grad_enabled(False)
        epoch = self.scheduler.last_epoch
        # epoch = self.optimizer.get_last_epoch()
        print('Test epoch', epoch)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        num = 0
        meanps = 0
        meanpsb = 0
        mssb = 0
        mss = 0
        for idx_data, d in enumerate(self.loader_test):
            (raw_rgb, raw_rgblr, raw_nir, raw_nirlr, key, dir) = d
            num += 1
            if not self.args.cpu:
                hrrgb = raw_rgb.cuda()
                lrrgb = raw_rgblr.cuda()
                lrnir = raw_nirlr.cuda()
                hrnir = raw_nir.cuda()
            else:
                hrrgb = raw_rgb
                lrrgb = raw_rgblr
                lrnir = raw_nirlr
                hrnir = raw_nir
        
            sr, nsr = self.model((lrrgb, lrnir), 0)
            sr = utility.quantize(sr, self.args.rgb_range)
            if self.args.cpu:
                im2 = np.squeeze(hrrgb.numpy()).transpose(1, 2, 0)*(255/self.args.rgb_range)
                im1 = np.uint8(np.squeeze(sr.numpy()).transpose(1, 2, 0)*(255/self.args.rgb_range))
            else:
                im2 = np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0)*(255/self.args.rgb_range)
                im1 = np.uint8(np.squeeze(sr.cpu().numpy()).transpose(1, 2, 0)*(255/self.args.rgb_range))
            im2 = np.uint8(np.clip(im2, 0, 255))  # [0,255]
            psnr, _ = utility.psnr(im1, im2)
            self.ckp.log[-1, idx_data, 0] += psnr
            if self.args.save_results:
                # self.ckp.save_results(d, key[0], save_list, self.scale)
                self.savedir = os.path.join(self.savepath, dir[0])
                os.makedirs(self.savedir, exist_ok=True)
                print('save to', self.savedir + '/{}.png'.format(key[0]))
                imageio.imwrite(self.savedir + '/{}.png'.format(key[0]), im1)
                # imageio.imwrite(self.savedir + '/HR-{}.png'.format(key[0]), im2)
                
            size1 = tuple(((np.array(
                Image.fromarray(im2).size)).astype(
                int)).astype(int))
            
            if self.args.cpu:
                lrnorm = np.clip(np.squeeze(lrrgb.numpy()).transpose(1, 2, 0), 0, self.args.rgb_range)*(255/self.args.rgb_range)
            else:
                lrnorm = np.clip(np.squeeze(lrrgb.cpu().numpy()).transpose(1, 2, 0), 0,
                                     self.args.rgb_range)*(255/self.args.rgb_range)
            bc = np.uint8(np.array(Image.fromarray(np.uint8(lrnorm)).resize(size1, Image.BICUBIC)))  # [0, 49]
            psnrb, _ = utility.psnr(bc, im2)
            meanps += psnr
            meanpsb += psnrb
            try:
                ss = compare_ssim(im1, im2, multichannel=True)
                ssb = compare_ssim(bc, im2, multichannel=True)  # [0,49] [0,56]
            except:
                ss = compare_ssim(im1, im2, channel_axis=3)
                ssb = compare_ssim(bc, im2, channel_axis=3)

            mss += ss
            mssb += ssb
            print(key, psnr, ss, 'Bicubic', psnrb, ssb)
            
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.scale,
                    self.ckp.log[-1].sum() / num,
                    best[0].sum() / num,
                    best[1][idx_data, 0] + 1))
    
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        print('Mean PSNR for %d image = ' % num, meanps / num, mss / num, 'Bicubic', meanpsb / num, mssb / num)
        if meanps / num > self.maxpsnr:
            self.maxpsnr = meanps / num
            self.maxepoch = epoch
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=True)
        self.ckp.write_log(
            'All Image [x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.scale,
                self.ckp.log[-1].sum() / num,
                self.maxpsnr, self.maxepoch))
        if self.args.save_results:
            self.ckp.end_background()
            
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(self.maxepoch == epoch))
    
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)
        return num, meanps, mss

    def testBicPSNR_VISNIR(self):
        num = 0
        meanpsb = 0
        mssb = 0
        for idx_data, d in enumerate(self.loader_test):
            (hrrgb, lrrgb, raw_nir, raw_nirlr, key, dir) = d
            num += 1  # float32 [0,1]
            im2 = np.squeeze(hrrgb.numpy()).transpose(1, 2, 0) * (255 / self.args.rgb_range)
            im2 = np.uint8(np.clip(im2, 0, 255))
            size1 = tuple(((np.array(Image.fromarray(im2).size)).astype(int)).astype(int))
            lrnorm = np.clip(np.squeeze(lrrgb.numpy()).transpose(1, 2, 0), 0, self.args.rgb_range) * (
                            255 / self.args.rgb_range)
            bc = np.uint8(np.array(Image.fromarray(np.uint8(lrnorm)).resize(size1, Image.BICUBIC)))  # [0, 49]
            Image.fromarray(bc).save('E:/file/python_project/RGBNIRStereo/result/bictrain.png')
            psnrb, _ = utility.psnr(bc, im2)
            meanpsb += psnrb
            ssb = compare_ssim(bc, im2, multichannel=True)  # [0,49] [0,56]
            mssb += ssb
            print(key, 'Bicubic', psnrb, ssb)
        print('Mean PSNR for %d image = ' % num, 'Bicubic', meanpsb / num, mssb / num)
        
    def trainVISNIR(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + self.args.resume
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
    
        for batch, (raw_rgb, raw_rgblr, raw_nir, raw_nirlr, key, dir) in enumerate(self.loader_train):
            # Preprocessing
            if not self.args.cpu:
                hrrgb = raw_rgb.cuda()
                lrrgb = raw_rgblr.cuda()
                lrnir = raw_nirlr.cuda()
                hrnir = raw_nir.cuda()
            else:
                hrrgb = raw_rgb
                lrrgb = raw_rgblr
                lrnir = raw_nirlr
                hrnir = raw_nir
        
            timer_data.hold()
            timer_model.tic()
        
            self.optimizer.zero_grad()
            sr, nsr = self.model((lrrgb, lrnir), 0)
            if epoch >= 50:
                loss = self.loss(sr, hrrgb) + 0.001 * self.loss(nsr, hrnir)
            else:
                loss = self.loss(sr, hrrgb) + 0.1 * self.loss(nsr, hrnir)
            
            loss.backward()
            if self.args.gclip > 0:
                print('self.args.gclip > 0', self.args.gclip > 0)
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            timer_model.hold()
        
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()  # self.optimizer.schedule()

    def testFLOP(self, device='cpu'):
        import torch.backends.cudnn as cudnn
        from torchvision.transforms import ToTensor
        from torch.autograd import Variable
        scale = 4
        
        # ## ============================= Two ============================ ##
        net = self.model  # PASSRnet(scale).to('cuda:0')
        cudnn.benchmark = True
        # pretrained_dict = torch.load('./log/x' + str(scale) + '/PASSRnet_x' + str(scale) + '.pth')
        # net.load_state_dict(pretrained_dict)
        
        with torch.no_grad():
            h, w = 582, 428  # 128, 128  #
            h = h // (16 * scale) * (16 * scale)
            w = w // (16 * scale) * (16 * scale)
            
            LRl = ToTensor()(np.zeros([h // (scale), w // (scale), 3], dtype=np.float32)).view(1, 3, h // (scale),
                                                                                                   w // (scale))
            LRr = ToTensor()(np.zeros([h // (scale), w // (scale), 1], dtype=np.float32)).view(1, 1, h // (scale),
                                                                                                   w // (scale))
            LR_left, LR_right = Variable(LRl).to(device), Variable(LRr).to(device)
        total = sum([param.nelement() for param in net.parameters()])
    
        print("Number of parameter: %.2fM" % (total / 1e6))
        from thop import profile
        flops, params = profile(net, inputs=((LR_left, LR_right), True))
        print(flops / 1e9, params / 1e6)

    def testY(self):
        savepath = './result/traindata/'
        os.makedirs(savepath, exist_ok=True)
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        if self.args.save_results: self.ckp.begin_background()
        num = 0
        meanps = 0
        meanpsb = 0
        mssb = 0
        mss = 0
        for idx_data, d in enumerate(self.loader_test):
            (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, raw_bicrgb, raw_bicnir, raw_ycbcr, raw_cbcrbc,
             raw_cbcrlr, raw_y, raw_ybc, raw_ylr) = d
            
            num += 1
            ycbcr, cbcrbc, cbcrlr = F.relu((raw_ycbcr - 2.0) / (255.0 - 2.0)).cuda(), F.relu(
                (raw_cbcrbc - 2.0) / (255.0 - 2.0)).cuda(), F.relu((raw_cbcrlr - 2.0) / (255.0 - 2.0)).cuda()
            y, ybc, ylr = F.relu((raw_y - 2.0) / (255.0 - 2.0)).cuda(), F.relu(
                (raw_ybc - 2.0) / (255.0 - 2.0)).cuda(), F.relu((raw_ylr - 2.0) / (255.0 - 2.0)).cuda()
            nirbc = F.relu((raw_bicnir - 2.0) / (255.0 - 2.0)).cuda()
            rgb = F.relu((raw_rgb - 2.0) / (255.0 - 2.0)).cuda()
            rgblr = F.relu((raw_rgblr - 2.0) / (255.0 - 2.0)).cuda()

            if 'StereoSR' in self.args.modelname:
                sr = self.model((ybc, nirbc, cbcrbc), 0)
        
            sr = utility.quantize(sr, self.args.rgb_range)
            save_list = [sr]
            # Preprocessing
            im2 = np.uint8(np.clip(np.squeeze(rgb.cpu().numpy()).transpose(1, 2, 0) * 255, 0, 255))
            im1 = cv2.cvtColor(np.uint8(
                np.squeeze(sr.cpu().numpy()).transpose(1, 2, 0) * 255), cv2.COLOR_YUV2RGB)
            im1bc = cv2.cvtColor(
                np.uint8(np.clip(np.squeeze(torch.cat([ybc, cbcrbc], 1).cpu().numpy()).transpose(1, 2, 0) * 255, 0, 255))
                , cv2.COLOR_YUV2RGB)
            imycbcr = cv2.cvtColor(
                np.uint8(np.clip(np.squeeze(ycbcr.cpu().numpy()).transpose(1, 2, 0) * 255, 0, 255))
                , cv2.COLOR_YUV2RGB)
            
            psnr, _ = utility.psnr(im1, im2)
            psnrbc, _ = utility.psnr(im1bc, im2)  # 26.177
            psnrhrycbcr, _ = utility.psnr(imycbcr, im2)  # 40.88
            self.ckp.log[-1, idx_data, 0] += psnr
        
            size1 = tuple(((np.array(Image.fromarray(im2).size)).astype(int)).astype(int))
            lrnorm = 255 * np.clip(np.squeeze(rgblr.cpu().numpy()).transpose(1, 2, 0) / 5 * 5, 0, 1)
            bc = np.uint8(np.array(Image.fromarray(np.uint8(lrnorm)).resize(size1, Image.BICUBIC)))
            # Image.fromarray(bc).save(savepath + key[0] + 'bic.png')
            # Image.fromarray(im1bc).save(savepath + key[0] + 'bicycbcr.png')
            # Image.fromarray(imycbcr).save(savepath + key[0] + 'HRycbcr.png')
            # Image.fromarray(im2).save(savepath + key[0] + 'HR.png')

            psnrb, _ = utility.psnr(bc, im2)  # 25.105
            meanps += psnr
            meanpsb += psnrb
            ss = compare_ssim(im1, im2, multichannel=True)
            ssb = compare_ssim(bc, im2, multichannel=True)  # [0,49] [0,56]
            
            mss += ss
            mssb += ssb
            print(key, psnr, ss, 'Bicubic', psnrb, ssb, 'Bicubic(YUV2RGB+light)/HR', psnrbc, psnrhrycbcr)
            # Bicubic 25.162 0.567 Bicubic(YUV2RGB+light)/HR 25.848 40.3482
            if self.args.save_results:
                self.ckp.save_results(d, key[0], save_list, self.scale, yuv=True)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.scale,
                    self.ckp.log[-1].sum() / num,
                    best[0].sum() / num,  # best[0][idx_data, 0] / num,  #
                    best[1][idx_data, 0] + 1))

        print('Mean PSNR for %d image = ' % num, meanps / num, mss / num, 'Bicubic', meanpsb / num, mssb / num)
    
        if self.args.save_results:
            self.ckp.end_background()
    
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
    
        torch.set_grad_enabled(True)
        return num, meanps / num, mss / num

    def trainTWY(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
    
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()
    
        timer_data, timer_model = utility.timer(), utility.timer()
    
        for batch, \
            (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, raw_bicrgb, raw_bicnir, raw_ycbcr, raw_cbcrbc,
             raw_cbcrlr, raw_y, raw_ybc, raw_ylr) in enumerate(
                self.loader_train):
            timer_data.hold()
            timer_model.tic()

            ycbcr, cbcrbc, cbcrlr = F.relu((raw_ycbcr - 2.0) / (255.0 - 2.0)).cuda(), F.relu(
                (raw_cbcrbc - 2.0) / (255.0 - 2.0)).cuda(), F.relu((raw_cbcrlr - 2.0) / (255.0 - 2.0)).cuda()
            y, ybc, ylr = F.relu((raw_y - 2.0) / (255.0 - 2.0)).cuda(), F.relu(
                (raw_ybc - 2.0) / (255.0 - 2.0)).cuda(), F.relu((raw_ylr - 2.0) / (255.0 - 2.0)).cuda()
            nirbc = F.relu((raw_bicnir - 2.0) / (255.0 - 2.0)).cuda()
            rgb = F.relu((raw_rgb - 2.0) / (255.0 - 2.0)).cuda()
            
            self.optimizer.zero_grad()
            if 'StereoSR' in self.args.modelname:
                sr = self.model((ybc, nirbc, cbcrbc), 0)  # rgb_ybic, nir_bic, cbcr_images
                loss = self.loss(sr, ycbcr)

            # im2 = np.uint8(np.clip(np.squeeze(rgb[0].cpu().numpy()).transpose(1, 2, 0) * 255, 0, 255))
            # im1bc = cv2.cvtColor(np.uint8(
            #     np.clip(np.squeeze(torch.cat([ybc[:1], cbcrbc[:1]], 1).cpu().numpy()).transpose(1, 2, 0) * 255, 0, 255)),
            #                      cv2.COLOR_YUV2RGB)
            # imycbcr = cv2.cvtColor(np.uint8(np.clip(np.squeeze(ycbcr[0].cpu().numpy()).transpose(1, 2, 0) * 255, 0, 255)),
            #                        cv2.COLOR_YUV2RGB)
            # psnrbc, _ = utility.psnr(im1bc, im2)
            # psnrhrycbcr, _ = utility.psnr(imycbcr, im2)
            # psnrbc_colorcvt, _ = utility.psnr(imycbcr, im1bc)
            # print('psnrbc, psnrhrycbcr, psnrbc_colorcvt', psnrbc, psnrhrycbcr, psnrbc_colorcvt)
            # 32.945441447768346 40.57845870942707 33.66663143887875
            
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
        
            timer_model.hold()
        
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
        
            timer_data.tic()
    
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    ##------------ enhance+SR --------------
    def testdark(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()
        
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        num = 0
        meanps = 0
        for idx_data, d in enumerate(self.loader_test):
            # d.dataset.set_scale(0)
            # for collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, rgb_exp, nir_exp in tqdm(d, ncols=80):
            (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, rgb_exp, nir_exp) = d
            num += 1
            # Preprocessing
            rgbd = rgb = F.relu((raw_rgb - 2.0) / (255.0 - 2.0)).cuda()
            rgbdlr = rgblr = F.relu((raw_rgblr - 2.0) / (255.0 - 2.0)).cuda()
            nird = nir = F.relu((raw_nir - 2.0) / (255.0 - 2.0)).cuda()
            nirdlr = nirlr = F.relu((raw_nirlr - 2.0) / (255.0 - 2.0)).cuda()
            rgb_ratio = 0.5 / (rgb.mean(1).mean(1).mean(1) + 1e-3)
            nir_ratio = 0.5 / (nir.mean(1).mean(1).mean(1) + 1e-3)
            hrrgb = torch.clamp(rgb * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            lrrgb = torch.clamp(rgblr * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            hrnir = torch.clamp(nir * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            lrnir = torch.clamp(nirlr * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)  # [0, 1]
            
            # EDSR
            if self.args.modelname == 'EDSR_RGB':
                sr = self.model(rgbdlr, 0)
            # # EDSR  dark + SR
            elif self.args.modelname == 'EDSR_EnhanceSR_RGB':
                lightlr, sr = self.model(rgbdlr, 0)
            elif self.args.modelname == 'EDSR_SREnhance_RGB':
                darksr, sr = self.model(rgbdlr, 0)
            elif self.args.modelname == 'EDSR_RGBNIR' or self.args.modelname == 'EDSR_RGBNIRw':
                sr = self.model(torch.cat([lrrgb, lrnir], 1), 0)
            elif self.args.modelname == 'EDSR_RGBNIR1' or self.args.modelname == 'EDSR_RGBNIRw1':
                sr = self.model(torch.cat([lrrgb, nirdlr], 1), 0)
            elif self.args.modelname == 'EDSR_EnhanceSR_RGBNIRw':
                lightlr, sr = self.model(torch.cat([lrrgb, lrnir], 1), 0)
            elif self.args.modelname == 'EDSR_RGBNIRwEn_2SR':
                lightlr, sr, nsr = self.model((lrrgb, lrnir), 0)
            elif self.args.modelname == 'EDSR_SREnhance_RGBNIRw_2stage':
                darksr, sr = self.model(torch.cat([lrrgb, nirdlr], 1), 0)
            
            sr = utility.quantize(sr, self.args.rgb_range)
            save_list = [sr]
            
            im1 = np.squeeze(sr.cpu().numpy()).transpose(1, 2, 0) * 255
            im2 = np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0) * 255
            psnr, _ = utility.psnr(im1, im2)
            # psnr = utility.calc_psnr(
            #     sr * 255, hrrgb * 255, self.scale[0], self.args.rgb_range, dataset=d
            # )
            meanps += psnr
            self.ckp.log[-1, idx_data, 0] += psnr
            print(psnr)
            # self.ckp.log[-1, idx_data, 1] += psnr1
            # if self.args.save_gt:
            #     save_list.extend([lr, hr])
            if self.args.save_results:
                self.ckp.save_results(d, key[0], save_list, self.scale)
            
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.scale,
                    self.ckp.log[-1].sum() / num,
                    # self.ckp.log[-1, idx_data, 0],  # self.ckp.log[-1, idx_data, 1] / num,
                    best[0].sum() / num,  # best[0][idx_data, 0] / num,  #
                    best[1][idx_data, 0] + 1))
        
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        print('Mean PSNR for %d image = ' % num, meanps)
        
        if self.args.save_results:
            self.ckp.end_background()
        
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
        
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        
        torch.set_grad_enabled(True)
        return num, meanps, meanps
    
    def trainTWdark(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        
        timer_data, timer_model = utility.timer(), utility.timer()
        i = 0
        for batch, (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, rgb_exp, nir_exp) in enumerate(
                self.loader_train):
            i += 1
            # Preprocessing
            rgbd = rgb = F.relu((raw_rgb - 2.0) / (255.0 - 2.0)).cuda()
            rgbdlr = rgblr = F.relu((raw_rgblr - 2.0) / (255.0 - 2.0)).cuda()
            nird = nir = F.relu((raw_nir - 2.0) / (255.0 - 2.0)).cuda()
            nirdlr = nirlr = F.relu((raw_nirlr - 2.0) / (255.0 - 2.0)).cuda()
            
            rgb_ratio = 0.5 / (rgb.mean(1).mean(1).mean(1) + 1e-3)
            nir_ratio = 0.5 / (nir.mean(1).mean(1).mean(1) + 1e-3)
            hrrgb = torch.clamp(rgb * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            lrrgb = torch.clamp(rgblr * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            hrnir = torch.clamp(nir * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            lrnir = torch.clamp(nirlr * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)  # [0, 1]
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            
            if self.args.modelname == 'EDSR_SREnhance_RGBNIRw_2stage':
                darksr, sr = self.model(torch.cat([lrrgb, nirdlr], 1), 0)
                # loss1 = self.loss(darksr, rgbd)  # 100 epoch
                # loss1.backward()
                loss2 = self.loss(sr, hrrgb)  # 200 epoch
                loss2.backward()
            
            else:
                # EDSR
                if self.args.modelname == 'EDSR_RGB':
                    sr = self.model(rgbdlr, 0)
                    loss = self.loss(sr, hrrgb)
                # # EDSR  dark + SR
                elif self.args.modelname == 'EDSR_EnhanceSR_RGB':
                    lightlr, sr = self.model(rgbdlr, 0)
                    loss = self.loss(sr, hrrgb) + self.loss(lightlr, lrrgb)
                elif self.args.modelname == 'EDSR_SREnhance_RGB':
                    darksr, sr = self.model(rgbdlr, 0)
                    loss = self.loss(sr, hrrgb) + self.loss(darksr, rgbd)
                elif self.args.modelname == 'EDSR_RGBNIR' or self.args.modelname == 'EDSR_RGBNIRw':
                    sr = self.model(torch.cat([lrrgb, lrnir], 1), 0)
                    loss = self.loss(sr, hrrgb)
                elif self.args.modelname == 'EDSR_RGBNIR1' or self.args.modelname == 'EDSR_RGBNIRw1':
                    sr = self.model(torch.cat([lrrgb, nirdlr], 1), 0)
                    loss = self.loss(sr, hrrgb)
                elif self.args.modelname == 'EDSR_EnhanceSR_RGBNIRw':
                    lightlr, sr = self.model(torch.cat([lrrgb, nirdlr], 1), 0)
                    loss = self.loss(sr, hrrgb) + self.loss(lightlr, lrrgb)
                # EDSR2branch
                elif self.args.modelname == 'EDSR_RGBNIRwEn_2SR':
                    lightlr, sr, nsr = self.model((lrrgb, nirdlr), 0)
                    loss = self.loss(sr, hrrgb) + self.loss(lightlr, lrrgb) + self.loss(nsr, hrnir)
                
                loss.backward()
            
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            
            timer_model.hold()
            
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            
            timer_data.tic()
        
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
    
    ## dark SR
    def testdarkSR(self, server=0):
        torch.set_grad_enabled(False)
        epoch = self.scheduler.last_epoch
        # epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        num = 0
        meanps = 0
        meanpsb = 0
        mss = 0
        mssb = 0
        for idx_data, d in enumerate(self.loader_test):
            (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, rgb_exp, nir_exp) = d
            num += 1
            # Preprocessing  raw_rgb [0, 58]
            rgb = F.relu((raw_rgb - 2.0) / (255.0 - 2.0)).cuda()  # [0, 0.22]
            rgblr = F.relu((raw_rgblr - 2.0) / (255.0 - 2.0)).cuda()  # [0, 0.3122]
            nir = F.relu((raw_nir - 2.0) / (255.0 - 2.0)).cuda()
            nirlr = F.relu((raw_nirlr - 2.0) / (255.0 - 2.0)).cuda()
            
            hrrgb = rgb  # [0, 1]
            lrrgb = rgblr  # [0, 1]
            # hrrgb = torch.clamp(rgb, 0.0, 1.0)  # [0, 1]
            # lrrgb = torch.clamp(rgblr, 0.0, 1.0)  # [0, 1]
            hrnir = torch.clamp(nir, 0.0, 1.0)
            lrnir = torch.clamp(nirlr, 0.0, 1.0)
            
            # EDSR
            if self.args.modelname == 'EDSR_RGB':
                sr = self.model(rgblr, 0)
            elif self.args.modelname == 'EDSR_RGBNIRw' or self.args.modelname == 'EDSR_RGBNIR':
                sr = self.model(torch.cat([lrrgb, lrnir], 1), 0)
            elif self.args.modelname == 'EDSR_RGBNIRw2branch':
                sr, srnir = self.model((lrrgb, lrnir), 0)
            
            sr = utility.quantize(sr, self.args.rgb_range)
            save_list = [sr]
            
            im1 = np.squeeze(sr.cpu().numpy()).transpose(1, 2, 0) * 255  # [0, 43]
            im2 = np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0) * 255  # [0, 56]
            psnr, _ = utility.psnr(im1, im2)
            # psnr = utility.calc_psnr(sr * 255, hrrgb * 255, self.scale[0], self.args.rgb_range, dataset=d)
            
            size1 = tuple(((np.array(
                Image.fromarray(np.uint8(np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0) * 255)).size)).astype(
                int)).astype(int))
            lrdnorm = np.squeeze(lrrgb.cpu().numpy()).transpose(1, 2, 0) * 255  # [0, 79.62]
            bcd = np.uint8(np.array(
                Image.fromarray(np.uint8(lrdnorm)).resize(size1, Image.BICUBIC)))  # [0, 49]
            psnrd, _ = utility.psnr(bcd, im2)
            meanps += psnr
            meanpsb += psnrd
            ss = compare_ssim(np.uint8(im1), np.uint8(im2), multichannel=True)
            ssb = compare_ssim(bcd, np.uint8(im2), multichannel=True)  # [0,49] [0,56]
            
            mss += ss
            mssb += ssb
            
            self.ckp.log[-1, idx_data, 0] += psnr
            print('Im%s' % key, psnr, ss, 'Bicubic', psnrd, ssb)
            if self.args.save_results:
                self.ckp.save_results(d, key[0], save_list, self.scale)
            
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[x{}]\tMean PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    self.scale,
                    self.ckp.log[-1].sum() / num,
                    best[0].sum() / num,  # best[0][idx_data, 0] / num,  #
                    best[1][idx_data, 0] + 1))
        
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        print('Mean PSNR for %d image = ' % num, meanps / num, mss / num, '\t Bicubic:', meanpsb/num, mssb/num)
        
        if self.args.save_results:
            self.ckp.end_background()
        
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
        
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        
        torch.set_grad_enabled(True)
        return num, meanps, mss
    
    def traindarkSR(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch
        # epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()
        
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        
        timer_data, timer_model = utility.timer(), utility.timer()
        i = 0
        for batch, (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, rgb_exp, nir_exp) in enumerate(
                self.loader_train):
            i += 1
            # Preprocessing
            rgb = F.relu((raw_rgb - 2.0) / (255.0 - 2.0)).cuda()
            rgblr = F.relu((raw_rgblr - 2.0) / (255.0 - 2.0)).cuda()
            nir = F.relu((raw_nir - 2.0) / (255.0 - 2.0)).cuda()
            nirlr = F.relu((raw_nirlr - 2.0) / (255.0 - 2.0)).cuda()
            
            hrrgb = torch.clamp(rgb, 0.0, 1.0)
            lrrgb = torch.clamp(rgblr, 0.0, 1.0)
            hrnir = torch.clamp(nir, 0.0, 1.0)
            lrnir = torch.clamp(nirlr, 0.0, 1.0)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()
            
            if self.args.modelname == 'EDSR_RGB':
                sr = self.model(lrrgb, 0)
            elif self.args.modelname == 'EDSR_RGBNIRw' or self.args.modelname == 'EDSR_RGBNIR':
                sr = self.model(torch.cat([lrrgb, lrnir], 1), 0)
            elif self.args.modelname == 'EDSR_RGBNIRw2branch':
                sr, srnir = self.model((lrrgb, lrnir), 0)
            
            loss = self.loss(sr, hrrgb)
            loss.backward()
            
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            
            timer_model.hold()
            
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            
            timer_data.tic()
        
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
    
    def testdata(self):
        savepathd = savepath = './result/traindata/'
        # # Generate LR data
        # savepath = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/20170223_1639100/s%d/' % self.args.scale[0]
        # savepathd = 'F:/SRdata/train_data/stereo/rgbnir/rgbnir_stereo/data/20170223_1639100/Darks%d/' % self.args.scale[0]
        os.makedirs(savepath, exist_ok=True)
        os.makedirs(savepathd, exist_ok=True)
        
        # for batch, (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, rgb_exp, nir_exp) in enumerate(
        #         self.loader_test):
        for batch, (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr) in enumerate(self.loader_test):
    
            # Preprocessing
            # rgbd = rgb = F.relu((raw_rgb - 2.0) / (255.0 - 2.0)).cuda()
            # rgbdlr = rgblr = F.relu((raw_rgblr - 2.0) / (255.0 - 2.0)).cuda()
            # nird = nir = F.relu((raw_nir - 2.0) / (255.0 - 2.0)).cuda()
            # nirdlr = nirlr = F.relu((raw_nirlr - 2.0) / (255.0 - 2.0)).cuda()
            #
            # rgb_ratio = 0.5 / (rgb.mean(1).mean(1).mean(1) + 1e-3)
            # nir_ratio = 0.5 / (nir.mean(1).mean(1).mean(1) + 1e-3)
            # hrrgb = torch.clamp(rgb * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            # lrrgb = torch.clamp(rgblr * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            # hrnir = torch.clamp(nir * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            # lrnir = torch.clamp(nirlr * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)  # [0, 1]
            #
            # # rgbre = to_image(hrrgb[0]).transpose([1, 2, 0])
            # # rgbre = cv2.cvtColor(rgbre, cv2.COLOR_RGB2BGR)
            # # cv2.imwrite(savepath + key[0] + 'rgb.png', rgbre)
            # # nirre = to_image(hrnir[0, 0])
            # # cv2.imwrite(savepath + key[0] + '.png', nirre)

            hrrgb = torch.clamp(((raw_rgb) * self.a).cuda(), 0.0, self.args.rgb_range)
            lrrgb = torch.clamp(((raw_rgblr) * self.a).cuda(), 0.0, self.args.rgb_range)
            lrnir = torch.clamp(((raw_nirlr) * self.a).cuda(), 0.0, self.args.rgb_range)  # [0, 1]
            hrnir = torch.clamp(((raw_nir) * self.a).cuda(), 0.0, self.args.rgb_range)

            rgbrelr = to_image(lrrgb[0]).transpose([1, 2, 0])
            cv2.imwrite(savepath + key[0][:-4] + 'rgblr.png', cv2.cvtColor(rgbrelr, cv2.COLOR_RGB2BGR))
            rgbrehr = to_image(hrrgb[0]).transpose([1, 2, 0])
            cv2.imwrite(savepath + key[0][:-4] + 'rgbhr.png', cv2.cvtColor(rgbrehr, cv2.COLOR_RGB2BGR))
            cv2.imwrite(savepath + key[0][:-4] + 'depthlr.png', to_image(lrnir[0,:,:,:]).transpose([1, 2, 0]))
            cv2.imwrite(savepath + key[0][:-4] + 'depthhr.png', to_image(hrnir[0,:,:,:]).transpose([1, 2, 0]))

            # rgbrelr = to_image(lrrgb[0]).transpose([1, 2, 0])
            # rgbrelr = cv2.cvtColor(rgbrelr, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(savepath + key[0] + 'rgblr.png', rgbrelr)
            # rgbredlr = to_image(rgbdlr[0]).transpose([1, 2, 0])
            # rgbredlr = cv2.cvtColor(rgbredlr, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(savepathd + key[0] + 'rgbdarklr.png', rgbredlr)
            # # nirrelr = to_image(lrnir[0, 0])
            # # cv2.imwrite(savepath + key[0] + 'lr.png', nirrelr)
    
    def terminate(self, server=1):
        self.server = server
        if self.args.test_only:
            if 'MMNet' in self.args.modelname:
                if (self.epoch % 100 ==0) and (self.epoch > 2):
                    self.testMMNet_ReadCorrect()
            if 'depthrgb' in self.args.modelname or 'cocrtin' in self.args.modelname:
                self.testReadCorrect()
            elif 'StereoSR' in self.args.modelname:
                self.testY()
            elif 'cocrt' in self.args.modelname:
                self.test()
            elif 'Enhance' in self.args.modelname:
                self.testdark()
            else:
                self.testdarkSR(server=server)
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            if 'depthrgb' in self.args.modelname:
                # self.testReadCorrect()
                if epoch < self.args.epochs:
                    if epoch % 200 == 0 and epoch != 1:
                        print('***Test**Epoch', epoch)
                        self.testReadCorrect()
                    return False
                elif epoch >= self.args.epochs:
                    return True
            else:
                return epoch >= self.args.epochs
    
    def terminateVISNIR(self):
        if self.args.test_only:
            self.testMMNet_VISNIR()
            # self.testVISNIR()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
            
    def testBic(self, server=0):
        savepath = 'F:/SRdata/train_data\stereo/rgbnir/rgbnir_stereo\data/20170224_0742100/Bics%d' % self.args.scale[0] + '/'
        # ./result/traindata
        os.makedirs(savepath, exist_ok=True)
        # savepathd = 'F:/SRdata/train_data\stereo/rgbnir/rgbnir_stereo\data/20170224_0742100/Darks%d' % self.args.scale[
        #     0] + '/'
        # os.makedirs(savepathd, exist_ok=True)
        num = 0
        ps = 0
        ss = 0
        ssd = 0
        psd = 0

        for idx_data, d in enumerate(self.loader_test):
            (collection, key, raw_rgb, raw_nir, raw_rgblr, raw_nirlr, rgb_exp, nir_exp) = d
            num += 1
            # Preprocessing # raw_rgb[0,58]
            rgbd = rgb = F.relu((raw_rgb - 2.0) / (255.0 - 2.0)).cuda()  # [0,0.2]
            rgbdlr = rgblr = F.relu((raw_rgblr - 2.0) / (255.0 - 2.0)).cuda()  # [0,0.2]
            nird = nir = F.relu((raw_nir - 2.0) / (255.0 - 2.0)).cuda()
            nirdlr = nirlr = F.relu((raw_nirlr - 2.0) / (255.0 - 2.0)).cuda()
            
            rgb_ratio = 0.5 / (rgb.mean(1).mean(1).mean(1) + 1e-3)  # 7.4781
            nir_ratio = 0.5 / (nir.mean(1).mean(1).mean(1) + 1e-3)
            hrrgb = torch.clamp(rgb * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)  # [0, 3.74]
            lrrgb = torch.clamp(rgblr * rgb_ratio.view(-1, 1, 1, 1), 0.0, 5.0)  # [0, 3.01]
            hrnir = torch.clamp(nir * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            lrnir = torch.clamp(nirlr * nir_ratio.view(-1, 1, 1, 1), 0.0, 5.0)
            
            size1 = tuple(((np.array(
                Image.fromarray(np.uint8(np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0) * 255)).size)).astype(
                int)).astype(int))
            
            im1 = np.squeeze(hrrgb.cpu().numpy()).transpose(1, 2, 0) * 51 * 5  # [0, 190.6] *5
            im1 = np.clip(im1, 0, 255)  # [0,255]
            im1d = np.squeeze(rgbd.cpu().numpy()).transpose(1, 2, 0) * 255  # [0, 56]
            lrrgbnorm = np.clip(np.squeeze(lrrgb.cpu().numpy()).transpose(1, 2, 0) / 5 * 5, 0, 1)  # [0, 0.6026] * 5  [0,1]
            lr = cv2.cvtColor(lrrgbnorm * 255, cv2.COLOR_RGB2BGR)
            
            bc = np.array(Image.fromarray(np.uint8(lr)).resize(size1, Image.BICUBIC))  # [0, 156]
            lrrgbdnorm = np.squeeze(rgbdlr.cpu().numpy()).transpose(1, 2, 0) * 255  # [0,47]
            bcd = np.array(Image.fromarray(np.uint8(lrrgbdnorm)).resize(size1, Image.BICUBIC))  # [0, 81]
            # ih, iw = im1.shape[:2]
            ## !!!!No CV.resize()!!!!
            # bc = cv2.resize(lrrgbnorm, (iw, ih), interpolation=cv2.INTER_CUBIC) * 255
            # bcd = cv2.resize(np.squeeze(rgbdlr.cpu().numpy()).transpose(1, 2, 0)/5, (iw, ih), interpolation=cv2.INTER_CUBIC) * 255
            
            # lrd = cv2.cvtColor(lrrgbdnorm, cv2.COLOR_RGB2BGR)
            # Image.fromarray(lr).save(savepath + key[0] + 'lr.png')
            # Image.fromarray(lrd).save(savepathd + key[0] + 'lrd.png')
            # cv2.imwrite(savepathd + key[0] + 'lrd.png', lrd)
            # cv2.imwrite(savepath + key[0] + 'lr.png', lr)
            # hr = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(savepath + key[0] + 'hr.png', hr)
            
            ps += utility.psnr(bc, im1)[0]
            psd += utility.psnr(bcd, im1d)[0]

            ss += compare_ssim(bc, im1, multichannel=True)
            ssd += compare_ssim(np.uint8(bcd), np.uint8(im1d), multichannel=True)  # [0,49], [0,56]

            print(key, ps, '/', ss, psd, '/', ssd)
            bc = cv2.cvtColor(bc, cv2.COLOR_RGB2BGR)
            Image.fromarray(bc).save(savepath + key[0] + 'bic.png')
            # cv2.imwrite(savepath + key[0] + 'bic.png', bc)
            # bcd = cv2.cvtColor(bcd, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(savepath + key[0] + 'bicdark.png', bcd)
            
            print(ps / num, ss / num, psd / num, ssd / num)
    
    
def cpu_np(tensor):
    return tensor.cpu().detach().numpy()
    
    
def to_image(matrix):
    image = cpu_np(torch.clamp(matrix, 0, 1) * 255).astype(np.uint8)
    if matrix.size()[0] == 1:
        image = np.concatenate((image, image, image), 0)
    return image


def to_color(arr, colors=((0, 1, 0), (1, 0, 1), (0, 1, 1))):
    """Converts a 2D or 3D stack to a colored image (maximal 3 channels).

    Parameters
    ----------
    arr : numpy.ndarray
        2D or 3D input data
    pmin : float
        lower percentile, pass -1 if no lower normalization is required
    pmax : float
        upper percentile, pass -1 if no upper normalization is required
    gamma : float
        gamma correction
    colors : list
        list of colors (r,g,b) for each channel of the input

    Returns
    -------
    numpy.ndarray
        colored image
    """
    if not arr.ndim in (2, 3):
        raise ValueError("only 2d or 3d arrays supported")
    
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    
    ind_min = np.argmin(arr.shape)
    arr = np.moveaxis(arr, ind_min, 0).astype(np.float32)
    out = arr
    return np.clip(out, 0, 1)


import matplotlib.pyplot as plt


def savecolorim(save, im, **imshow_kwargs):
    imshow_kwargs['cmap'] = 'magma'
    imshow_kwargs['vmin'] = 0
    imshow_kwargs['vmax'] = 255
    
    # im = np.asarray(im)
    # im = np.stack(map(to_color, im)) if 1 < im.shape[-1] <= 3 else im
    # ndim_allowed = 2 + int(1 <= im.shape[-1] <= 3)
    # proj_axis = tuple(range(1, 1 + max(0, im[0].ndim - ndim_allowed)))
    # im = np.max(im, axis=proj_axis)
    
    plt.imsave(save, im, **imshow_kwargs)
