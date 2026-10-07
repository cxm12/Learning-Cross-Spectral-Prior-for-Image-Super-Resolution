from model import common
import torch
import torch.nn as nn
from model.PASSR import PASSRnet
from model.MMNet import MMSR_net
from model.HAT import HAT_Our
import os

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


def make_model(args, parent=False):
    if 'HAT' in args.modelname:
        return HAT_Our(upscale=args.scale[0],depthsr='depth' in args.modelname)
    if 'MMNet' in args.modelname:
        return MMSR_net(weights_regularizer=[0.0005, 0, 0, 0])
    if 'StereoSR_cocrt_RGBNIR' == args.modelname:
        return StereoSR(args)
    elif 'PASSR_cocrt_RGBNIR' == args.modelname:
        return PASSRnet(args, upscale_factor=args.scale[0])
    elif 'depthrgb' in args.modelname:
        if 'depthrgb_uncertainty_elu' == args.modelname:
            return EDSR2branchuncertainty(args)
        elif 'depthrgbdepthSR_w2branch1' == args.modelname:
            return EDSR2branch1_depthsr(args)
        elif 'depthrgbleftSR_w2branch1' == args.modelname:
            return EDSR2branch1(args)
    elif 'cocrt' in args.modelname:
        if 'EDSRcocrtinNIRw_UDL_stage1' == args.modelname:
            return EDSRUDL(args)
        elif 'EDSRcocrtinNIRw_UDL_twostage' == args.modelname:
            return EDSRUDL_two(args)
        elif 'EDSRcocrtinNIRw_uncertainty_twostage_f' == args.modelname:
            return EDSR2branchuncertainty_twostage_f(args)
        elif 'EDSRcocrtinNIRw_uncertainty_twostage' == args.modelname:
            return EDSR2branchuncertainty_twostage(args)
        elif 'EDSRcocrtinNIRw_uncertainty_stage1' == args.modelname:
            return EDSR2branchuncertainty_stage1(args)
        elif args.modelname == 'EDSRcocrtinNIRw_uncertainty_elu':
            return EDSR2branchuncertainty(args)
        elif args.modelname == 'EDSR_cocrt_RGB':
            return EDSR(args)
        elif 'RGBNIR' in args.modelname:
            if 'EDSR_cocrt_RGBNIRw2branch' == args.modelname:
                return EDSR2branch(args)
            elif '2branch1' in args.modelname or 'EDSR_cocrt_RGBNIRw2branch3' == args.modelname or \
                    'EDSR_cocrt_RGBNIRw2branch4' == args.modelname or 'EDSR_cocrt_RGBNIRw2branch2' == args.modelname\
                    or args.modelname == 'EDSR_cocrtin_RGBNIRw2branch2':
                return EDSR2branch1(args)
            elif 'EDSR_cocrt_RGBNIRw2branch0' == args.modelname:
                return EDSR2branch0(args)
            elif 'EDSR_cocrt_RGBNIRw2HRbranch' in args.modelname:
                return 0
            elif 'hr' in args.modelname:
                return EDSRcathr(args)
            elif args.modelname == 'EDSR_cocrt_RGBNIR' or args.modelname == 'EDSR_cocrt_RGBNIRw':
                return EDSRcat(args)
        else:
            return EDSR(args)
    else:
        if args.modelname == 'EDSR_RGB':
            return EDSR(args)
        elif args.modelname == 'EDSR_EnhanceSR_RGB':
            return EDSRs1_SR(args)
        elif args.modelname == 'EDSR_SREnhance_RGB':
            return EDSRSR_s1(args)
        if 'RGBNIR' in args.modelname:
            if args.modelname == 'EDSR_RGBNIRw2branch':
                return EDSR2branch(args)
            if args.modelname == 'EDSR_RGBNIR' or args.modelname == 'EDSR_RGBNIRw' or args.modelname == 'EDSR_RGBNIR1' or args.modelname == 'EDSR_RGBNIRw1':
                return EDSRcat(args)
            if args.modelname == 'EDSR_EnhanceSR_RGBNIRw':
                return EDSRs1_SR(args, sub=False)
            if args.modelname == 'EDSR_RGBNIRwEn_2SR':
                return EDSR2branch_EnhanceSR(args)
            if args.modelname == 'EDSR_SREnhance_RGBNIRw_2stage':
                return EDSRSR_s1_2Stage(args, fixed=True)


class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        
        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        
        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
    
    def forward(self, x):
        x = self.sub_mean(x)
        
        x = self.head(x)
        
        res = self.body(x)
        res += x
        
        x = self.tail(res)
        x = self.add_mean(x)
        
        return x
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


## ------------- Dedark + SR: ------------------
class EDSRs1_SR(nn.Module):
    def __init__(self, args, conv=common.default_conv, sub=True):
        super(EDSRs1_SR, self).__init__()
        self.sub = sub
        
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        # define head module
        if sub:
            m_head = [conv(args.n_colors, n_feats, kernel_size)]
        else:
            m_head = [conv(args.n_colors + 1, n_feats, kernel_size)]
        
        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks // 2)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_bodysr = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks // 2)
        ]
        m_bodysr.append(conv(n_feats, n_feats, kernel_size))
        
        # define tail module
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]
        m_tailsr = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.bodysr = nn.Sequential(*m_bodysr)
        self.tail = nn.Sequential(*m_tail)
        self.tailsr = nn.Sequential(*m_tailsr)
    
    def forward(self, x):
        if self.sub:
            x = self.sub_mean(x)
        
        x = self.head(x)
        
        x1 = self.body(x)
        res = x1 + x
        xd = self.tail(res)
        
        xsr = self.bodysr(x1)
        xsr1 = self.tailsr(xsr)
        if self.sub:
            sr = self.add_mean(xsr1)
        else:
            sr = xsr1
        
        return xd, sr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSRSR_s1(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSRSR_s1, self).__init__()
        
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        
        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks // 2)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_bodysr = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks // 2)
        ]
        m_bodysr.append(conv(n_feats, n_feats, kernel_size))
        
        # define tail module
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]
        m_tailsrup = [
            common.Upsampler(conv, scale, n_feats, act=False),
        ]
        m_tailsr = [
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.bodysr = nn.Sequential(*m_bodysr)
        self.tail = nn.Sequential(*m_tail)
        self.tailsr = nn.Sequential(*m_tailsr)
        self.tailsrup = nn.Sequential(*m_tailsrup)
    
    def forward(self, x):
        x = self.sub_mean(x)
        
        x = self.head(x)
        
        x1 = self.bodysr(x)
        res = x1 + x
        xsrf = self.tailsrup(res)
        darksr = self.tailsr(xsrf)
        
        x2 = self.body(xsrf)
        x2 = self.tail(x2)
        sr = self.add_mean(x2)
        
        return darksr, sr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSRSR_s1_2Stage(nn.Module):
    def __init__(self, args, conv=common.default_conv, fixed=False):
        super(EDSRSR_s1_2Stage, self).__init__()
        
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # define SR module
        m_head = [conv(args.n_colors+1, n_feats, kernel_size)]
        m_bodysr = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks // 2)
        ]
        m_bodysr.append(conv(n_feats, n_feats, kernel_size))
        m_tailsrup = [
            common.Upsampler(conv, scale, n_feats, act=False),
        ]
        m_tailsr = [
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.head = nn.Sequential(*m_head)
        self.bodysr = nn.Sequential(*m_bodysr)

        self.tailsr = nn.Sequential(*m_tailsr)
        self.tailsrup = nn.Sequential(*m_tailsrup)
        
        if fixed:
            for p in self.parameters():
                p.requires_grad = False
            
        # define Enhance module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks // 2)]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_tail = [conv(n_feats, args.n_colors, kernel_size)]
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        # self.sub_mean = common.MeanShift(args.rgb_range)
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)

    def forward(self, x):
        # x = self.sub_mean(x)
        
        x = self.head(x)
        x1 = self.bodysr(x)
        res = x1 + x
        xsrf = self.tailsrup(res)
        darksr = self.tailsr(xsrf)
        
        x2 = self.body(xsrf)
        sr = self.tail(x2)
        
        return darksr, sr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSR2branch_EnhanceSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR2branch_EnhanceSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        # define head module
        self.head = nn.Sequential(conv(args.n_colors, n_feats, kernel_size))
        self.headn = nn.Sequential(conv(1, n_feats, kernel_size))

        # define body module
        m_body1 = [common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
                                  ) for _ in range(n_resblocks // 4)]
        m_body1.append(conv(n_feats, n_feats, kernel_size))
        m_bodyn1 = [common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
                                  ) for _ in range(n_resblocks // 4)]
        m_bodyn1.append(conv(n_feats, n_feats, kernel_size))
        m_body2 = [common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
                                  ) for _ in range(n_resblocks // 4)]
        m_body2.append(conv(n_feats, n_feats, kernel_size))
        m_bodynsr = [common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
                                  ) for _ in range(n_resblocks // 2)]
        m_bodynsr.append(conv(n_feats, n_feats, kernel_size))
        m_bodysr = [common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks // 2)]
        m_bodysr.append(conv(n_feats, n_feats, kernel_size))
        self.body1 = nn.Sequential(*m_body1)
        self.bodyn1 = nn.Sequential(*m_bodyn1)
        self.body2 = nn.Sequential(*m_body2)
        self.bodynsr = nn.Sequential(*m_bodynsr)
        self.conv1 = nn.Sequential(*[conv(n_feats*2, n_feats, kernel_size)])
        self.bodysr = nn.Sequential(*m_bodysr)
        self.conv = nn.Sequential(
            conv(n_feats, n_feats, kernel_size))
        
        # define tail module
        self.tailenhance = nn.Sequential(*[conv(n_feats, args.n_colors, kernel_size)])
        self.tailsr = nn.Sequential(
            *[common.Upsampler(conv, scale, n_feats, act=False), conv(n_feats, args.n_colors, kernel_size)])
        self.tailnsr = nn.Sequential(
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 1, kernel_size))
        
        self.RN1 = common.NIRRGBBlock(conv, n_feats, kernel_size)
    
    def forward(self, inp):
        (xrgb, xnir) = inp
        xr = self.head(xrgb)
        xn = self.headn(xnir)
        
        ##  RGB Enhance LR
        x1 = self.body1(xr)
        xn1 = self.bodyn1(xn)
        xc1 = self.RN1(x1, xn1)
        x2 = self.body2(xc1)
        xenhance = self.conv(x2)
        xenhanceLR = self.tailenhance(xenhance)
        
        ## NIR SR
        xnsr1 = self.bodynsr(xn)
        resn = xnsr1 + xn
        xnsr = self.tailnsr(resn)
        
        ## RGB SR
        xsr1 = self.bodysr(self.conv1(torch.cat([xnsr1, xenhance], 1)))
        resr = x1 + xsr1
        xrsr = self.tailsr(resr)
        
        return xenhanceLR, xrsr, xnsr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


## -------------- Only SR ----------------------
class EDSRcat(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSRcat, self).__init__()
        
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        # define head module
        m_head = [conv(args.n_colors + 1, n_feats, kernel_size)]
        
        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        
        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
    
    def forward(self, inputx):
        # [x, xnir] = inputx
        x = inputx
        
        x = self.head(x)
        
        res = self.body(x)
        res += x
        
        x = self.tail(res)
        
        return x
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSRcathr(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSRcathr, self).__init__()
        
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]
        
        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define NIR head module
        m_headn = [conv(1, n_feats, kernel_size)]

        # define NIR body module
        m_bodyn = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(1)
        ]
        m_bodyn.append(conv(n_feats, n_feats, kernel_size))
        
        # define tail module
        m_tailr = [
            common.Upsampler(conv, scale, n_feats, act=False),
        ]
        m_tailr1 = [
            conv(n_feats*2, args.n_colors, kernel_size)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.headn = nn.Sequential(*m_headn)
        self.body = nn.Sequential(*m_body)
        self.bodyn = nn.Sequential(*m_bodyn)
        self.tailr = nn.Sequential(*m_tailr)
        self.tailr1 = nn.Sequential(*m_tailr1)
    
    def forward(self, inp):
        x, xnir = inp
        
        x = self.head(x)
        res = self.body(x)
        res += x
        xr = self.tailr(res)  # upsample

        xnir1 = self.headn(xnir)
        xnir2 = self.bodyn(xnir1)
        
        x = self.tailr1(torch.cat([xr, xnir2], 1))
        
        return x
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


# double branch SR
class EDSR2branch(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR2branch, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        ## RGB branch
        self.head = nn.Sequential(*[conv(args.n_colors, n_feats, kernel_size)])
        self.conv = nn.Sequential(conv(n_feats, n_feats, kernel_size))

        self.body4 = self.body3 = self.body2 = self.body1 = nn.Sequential(*[
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks // 4)])
        self.body = nn.Sequential(*[
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)])
        self.tail = nn.Sequential(*[common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)])
            
        # # NIR branch
        self.headn = nn.Sequential(conv(1, n_feats, kernel_size))
        self.bodyn4 = self.bodyn3 = self.bodyn2 = self.bodyn1 = nn.Sequential(common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        self.tailn1 = nn.Sequential(
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 1, kernel_size))

        self.RN1 = common.NIRRGBBlock(conv, n_feats, kernel_size)
        self.RN2 = common.NIRRGBBlock(conv, n_feats, kernel_size)
        self.RN3 = common.NIRRGBBlock(conv, n_feats, kernel_size)
        self.RN4 = common.NIRRGBBlock(conv, n_feats, kernel_size)
        
    def forward(self, inp):
        (xrgb, xnir) = inp
        
        xr = self.head(xrgb)
        xn = self.headn(xnir)
        
        xrgb1 = self.body(xr)
        x1 = self.body1(xr)
        xn1 = self.bodyn1(xn)
        xc1 = self.RN1(x1, xn1)

        x2 = self.body2(xc1)
        xn2 = self.bodyn2(xn1)
        xc2 = self.RN1(x2, xn2)

        x3 = self.body3(xc2)
        xn3 = self.bodyn3(xn2)
        xc3 = self.RN1(x3, xn3)

        x4 = self.body4(xc3)
        xn4 = self.bodyn4(xn3)
        xc4 = self.RN1(x4, xn4)
        xc5 = self.conv(xc4)

        resr = xrgb1 + xc5 + xr
        resn = xn4 + xn

        xrsr = self.tail(resr)
        xnsr = self.tailn1(resn)
        
        return xrsr, xnsr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSR2branch1(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR2branch1, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        ## RGB branch
        self.head = nn.Sequential(conv(args.n_colors, n_feats, kernel_size), conv(n_feats, n_feats, kernel_size))
        # self.conv = nn.Sequential(conv(n_feats, n_feats, kernel_size))
        self.convcat = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size))
        self.body4 = self.body3 = self.body2 = self.body1 = nn.Sequential(*[
            common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks // 4)])
        # self.body = nn.Sequential(*[
        #     common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
        #     ) for _ in range(n_resblocks // 2)])
        self.tail = nn.Sequential(common.Upsampler(conv, scale, n_feats, act=False),
                                    conv(n_feats, args.n_colors, kernel_size))
    
        ## NIR branch
        self.headn = nn.Sequential(conv(1, n_feats, kernel_size),
                                   conv(n_feats, n_feats, kernel_size))
        self.bodyn4 = self.bodyn3 = self.bodyn2 = self.bodyn1 = nn.Sequential(common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        self.tailn = nn.Sequential(
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 1, kernel_size))
        
        # CMFT
        self.RN3 = self.RN2 = self.RN1 = common.NIRRGBBlock1(conv, n_feats, kernel_size)
    
    def forward(self, inp):
        (xrgb, xnir) = inp
        
        xr = self.head(xrgb)
        # xrgb1 = self.body(xr)
        xn = self.headn(xnir)
        
        x1 = self.body1(xr)
        xn1 = self.bodyn1(xn)
        xc1 = self.RN1(x1, xn1)
        
        x2 = self.body2(self.convcat(torch.cat([xc1, x1], 1)))
        xn2 = self.bodyn2(xn1)
        xc2 = self.RN2(x2, xn2)
        
        x3 = self.body3(self.convcat(torch.cat([xc2, x2], 1)))
        xn3 = self.bodyn3(xn2)
        xc3 = self.RN3(x3, xn3)
        
        x4 = self.body4(self.convcat(torch.cat([xc3, x3], 1)))
        resr = x4 + xr  # xrgb1 +
        xrsr = self.tail(resr)
        
        xn4 = self.bodyn4(xn3)
        resn = xn4 + xn
        xnsr = self.tailn(resn)
        
        # return x1, xn1
        return xrsr, xnsr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSR2branch1HR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR2branch1HR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]
        act = nn.ReLU(True)
        
        ## RGB branch
        self.head = nn.Sequential(conv(args.n_colors, n_feats, kernel_size), conv(n_feats, n_feats, kernel_size))
        # self.conv = nn.Sequential(conv(n_feats, n_feats, kernel_size))
        self.convcat = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size))
        self.body4 = self.body3 = self.body2 = self.body1 = nn.Sequential(*[
            common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
                            ) for _ in range(n_resblocks // 4)])
        self.tail = nn.Sequential(common.Upsampler(conv, self.scale, n_feats, act=False),
                                  conv(n_feats, args.n_colors, kernel_size))
        
        ## NIR branch
        self.headn = nn.Sequential(conv(1, n_feats, kernel_size),
                                   conv(n_feats, n_feats, kernel_size))
        self.bodyn4 = self.bodyn3 = self.bodyn2 = self.bodyn1 = nn.Sequential(common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        self.tailn = nn.Sequential(
            common.Upsampler(conv, self.scale, n_feats, act=False),
            conv(n_feats, 1, kernel_size))
        # CMFT
        self.RN3 = self.RN2 = self.RN1 = common.NIRRGBBlock1(conv, n_feats, kernel_size)
    
    def forward(self, inp):
        (xrgb, xnir) = inp
        xnir = torch.nn.functional.pixel_unshuffle(xnir, self.scale)
        xr = self.head(xrgb)  # xrgb1 = self.body(xr)
        xn = self.headn(xnir)
        
        x1 = self.body1(xr)
        xn1 = self.bodyn1(xn)
        xc1 = self.RN1(x1, xn1)
        
        x2 = self.body2(self.convcat(torch.cat([xc1, x1], 1)))
        xn2 = self.bodyn2(xn1)
        xc2 = self.RN2(x2, xn2)
        
        x3 = self.body3(self.convcat(torch.cat([xc2, x2], 1)))
        xn3 = self.bodyn3(xn2)
        xc3 = self.RN3(x3, xn3)
        
        x4 = self.body4(self.convcat(torch.cat([xc3, x3], 1)))
        resr = x4 + xr  # xrgb1 +
        xrsr = self.tail(resr)
        
        xn4 = self.bodyn4(xn3)
        resn = xn4 + xn
        xnsr = self.tailn(resn)
        
        # return x1, xn1
        return xrsr, xnsr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


## w/o CMFT
class EDSR2branch0(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR2branch0, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        ## RGB branch
        self.head = nn.Sequential(conv(args.n_colors, n_feats, kernel_size),
                                  conv(n_feats, n_feats, kernel_size))
        self.convcat3 = self.convcat2 = self.convcat1 = self.convcat = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size))
        self.body4 = self.body3 = self.body2 = self.body1 = nn.Sequential(*[
            common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
                            ) for _ in range(n_resblocks // 4)])
        self.tail = nn.Sequential(common.Upsampler(conv, scale, n_feats, act=False),
                                  conv(n_feats, args.n_colors, kernel_size))
        
        ## NIR branch
        self.headn = nn.Sequential(conv(1, n_feats, kernel_size),
                                   conv(n_feats, n_feats, kernel_size))
        self.bodyn4 = self.bodyn3 = self.bodyn2 = self.bodyn1 = nn.Sequential(common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        self.tailn = nn.Sequential(
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 1, kernel_size))
        
        # CMFT
        self.RN3 = self.RN2 = self.RN1 = common.NIRRGBBlock1(conv, n_feats, kernel_size)
    
    def forward(self, inp):
        (xrgb, xnir) = inp
        
        xr = self.head(xrgb)
        xn = self.headn(xnir)
        
        x1 = self.body1(xr)
        xn1 = self.bodyn1(xn)
        xc1 = self.convcat1(torch.cat([x1, xn1], 1))
        
        x2 = self.body2(self.convcat(torch.cat([xc1, x1], 1)))
        xn2 = self.bodyn2(xn1)
        xc2 = self.convcat2(torch.cat([x2, xn2], 1))
        
        x3 = self.body3(self.convcat(torch.cat([xc2, x2], 1)))
        xn3 = self.bodyn3(xn2)
        xc3 = self.convcat3(torch.cat([x3, xn3], 1))
        
        x4 = self.body4(self.convcat(torch.cat([xc3, x3], 1)))
        resr = x4 + xr  # xrgb1 +
        xrsr = self.tail(resr)
        
        xn4 = self.bodyn4(xn3)
        resn = xn4 + xn
        xnsr = self.tailn(resn)
        
        return xrsr, xnsr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


## CVPR 18
class StereoSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(StereoSR, self).__init__()
        act = nn.ReLU(True)
        self.conv1 = nn.Sequential(conv(64+1, 64, 3), act)
        self.conv_node = nn.Sequential(*[conv(64, 64, 3) for _ in range(16 - 2)])
        self.conv_last = nn.Sequential(conv(64, 1, 3))
        self.color_conv1 = nn.Sequential(conv(3, 64, 3), act)
        self.color_conv = nn.Sequential(*[conv(64, 64, 3) for _ in range(14)])
        self.color_conv16 = nn.Sequential(conv(64, 3, 3), act)
        self.color_residual = nn.Sequential(conv(6, 3, 3), act)

    def forward(self, inp):
        (left_images, right_images, cbcr_images) = inp
        images = torch.cat([left_images, right_images], 1)

        conv1 = self.conv1(images)

        conv_node = conv1
        conv_node = self.conv_node(conv_node)

        matching = self.conv_last(conv_node)

        logits = matching + left_images
        luminance_image = logits

        # Color Recon
        images = torch.cat([luminance_image, cbcr_images], 1)
        conv1 = self.color_conv1(images)
        conv_node = conv1
        conv_node = self.color_conv(conv_node)

        residual = self.color_conv16(conv_node)
        merged = torch.cat([residual, images], 1)
        logits = self.color_residual(merged)

        return logits
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSR2branch1_depthsr(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR2branch1_depthsr, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        
        ## RGB branch
        self.head = nn.Sequential(conv(args.n_colors, n_feats, kernel_size), conv(n_feats, n_feats, kernel_size))
        self.convcat = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size))
        self.body4 = self.body3 = self.body2 = self.body1 = nn.Sequential(common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        self.tail = nn.Sequential(common.Upsampler(conv, scale, n_feats, act=False),
                                  conv(n_feats, args.n_colors, kernel_size))
        
        ## NIR branch
        self.headn = nn.Sequential(conv(1, n_feats, kernel_size),
                                   conv(n_feats, n_feats, kernel_size))
        self.bodyn4 = self.bodyn3 = self.bodyn2 = self.bodyn1 = nn.Sequential(*[
            common.ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
                            ) for _ in range(n_resblocks // 4)])
        self.tailn = nn.Sequential(
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 1, kernel_size))
        
        # CMFT
        self.RN3 = self.RN2 = self.RN1 = common.NIRRGBBlock1(conv, n_feats, kernel_size)
    
    def forward(self, inp):
        (xrgb, xnir) = inp
        
        xr = self.head(xrgb)
        xn = self.headn(xnir)
        
        x1 = self.body1(xr)
        xn1 = self.bodyn1(xn)
        xc1 = self.RN1(xn1, x1)
        
        x2 = self.body2(self.convcat(torch.cat([xc1, x1], 1)))
        xn2 = self.bodyn2(xn1)
        xc2 = self.RN2(xn2, x2)
        
        x3 = self.body3(self.convcat(torch.cat([xc2, x2], 1)))
        xn3 = self.bodyn3(xn2)
        xc3 = self.RN3(xn3, x3)
        
        x4 = self.body4(self.convcat(torch.cat([xc3, x3], 1)))
        resr = x4 + xr
        xrsr = self.tail(resr)
        
        xn4 = self.bodyn4(xn3)
        resn = xn4 + xn
        xnsr = self.tailn(resn)
        
        return xrsr, xnsr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


## ========================== Uncertainty ===========================
class EDSR2branchuncertainty(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR2branchuncertainty, self).__init__()
        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]
        act = nn.ReLU(True)
        
        ## RGB branch
        self.head = nn.Sequential(conv(args.n_colors, n_feats, kernel_size), conv(n_feats, n_feats, kernel_size))
        self.headn = nn.Sequential(conv(1, n_feats, kernel_size),
                                   conv(n_feats, n_feats, kernel_size))
        self.convcat = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size))
        self.body1 = self.bodyn1 = nn.Sequential(common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale), nn.InstanceNorm2d(n_feats))
        self.tailn = nn.Sequential(
            common.Upsampler(conv, self.scale, n_feats, act=False),
            conv(n_feats, 1, kernel_size))
        self.tail = nn.Sequential(common.Upsampler(conv, self.scale, n_feats, act=False),
                                  conv(n_feats, args.n_colors, kernel_size))

        self.B1 = nn.Sequential(conv(2*n_feats, n_feats, kernel_size), common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale), common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        self.convf1 = self.convf2 = self.convf3 = nn.Sequential(conv(n_feats, n_feats, kernel_size))
        self.convnf1 = self.convnf2 = self.convnf3 = nn.Sequential(conv(n_feats, n_feats, kernel_size))
        self.convim1 = self.convim2 = self.convim3 = nn.Sequential(common.Upsampler(conv, self.scale, n_feats, act=False), conv(n_feats, args.n_colors, kernel_size))
        
        self.convun1 = self.convun2 = self.convun3 = nn.Sequential(conv(n_feats, n_feats, kernel_size), nn.ELU())
        self.convunup1 = self.convunup2 = self.convunup3 = nn.Sequential(common.Upsampler(conv, self.scale, 1, act=False), nn.ELU())
        self.B2 = self.B3 = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size), common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale), common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        self.var_conv = nn.Sequential(*[conv(n_feats, n_feats, kernel_size), nn.ELU(),
                                        conv(n_feats, n_feats, kernel_size), nn.ELU(),
                                        conv(n_feats, args.n_colors, kernel_size), nn.ELU()])
    
    def forward_f(self, inp):  # feature 扰动
        (xrgb, xnir) = inp
        
        xr = self.head(xrgb)
        xn = self.headn(xnir)
        
        x1 = self.body1(xr)
        xn1 = self.bodyn1(xn)
        
        xc1 = self.B1(torch.cat([x1, xn1], 1))
        xf1 = self.convf1(xc1)
        xnf1 = self.convnf1(xn1)
        sr1 = self.convim1(xf1 + xr)
        thetaf1 = self.convun1(xc1)
        normt1 = xf1 + thetaf1 * torch.normal(0, 1, xf1.size(), generator=None, out=None).cuda()
        theta1 = self.convunup1(thetaf1)

        xc2 = self.B2(torch.cat([x1, normt1, xnf1], 1))
        xf2 = self.convf2(xc2)
        xnf2 = self.convnf2(xnf1)
        sr2 = self.convim2(xf2 + xr)
        thetaf2 = self.convun1(xc2)
        normt2 = xf2 + thetaf2 * torch.normal(0, 1, xf2.size(), generator=None, out=None).cuda()
        theta2 = self.convunup1(thetaf2)

        xc3 = self.B3(torch.cat([x1, normt2, xnf2], 1))
        xf3 = self.convf3(xc3)
        xnf3 = self.convnf3(xnf2)
        sr3 = self.convim3(xf3 + xr)
        thetaf3 = self.convun1(xc3)
        theta3 = self.convunup1(thetaf3)
       
        resn = xnf3 + xn
        xnsr = self.tailn(resn)
        
        return sr1, theta1, sr2, theta2, sr3, theta3, xnsr

    def forward(self, inp):
        (xrgb, xnir) = inp
    
        xr = self.head(xrgb)
        xn = self.headn(xnir)
    
        x1 = self.body1(xr)
        xn1 = self.bodyn1(xn)
    
        xc1 = self.B1(torch.cat([x1, xn1], 1))
        xf1 = self.convf1(xc1)
        xnf1 = self.convnf1(xn1)
        sr1 = self.convim1(xf1 + xr)
        theta1 = self.var_conv(
            nn.functional.interpolate(xf1, scale_factor=self.scale, mode='nearest'))
    
        xc2 = self.B2(torch.cat([x1,  xnf1], 1))
        xf2 = self.convf2(xc2)
        xnf2 = self.convnf2(xnf1)
        sr2 = self.convim2(xf2 + xr)
        theta2 = self.var_conv(
            nn.functional.interpolate(xf2, scale_factor=self.scale, mode='nearest'))
    
        xc3 = self.B3(torch.cat([x1, xnf2], 1))
        xf3 = self.convf3(xc3)
        xnf3 = self.convnf3(xnf2)
        sr3 = self.convim3(xf3 + xr)
        theta3 = self.var_conv(
            nn.functional.interpolate(xf3, scale_factor=self.scale, mode='nearest'))
        
        resn = xnf3 + xn
        xnsr = self.tailn(resn)
    
        return sr1, theta1, sr2, theta2, sr3, theta3, xnsr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSR2branchuncertainty_stage1(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR2branchuncertainty_stage1, self).__init__()
        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]
        act = nn.ReLU(True)
        
        ## RGB branch
        self.head = nn.Sequential(conv(args.n_colors, n_feats, kernel_size), conv(n_feats, n_feats, kernel_size))
        self.headn = nn.Sequential(conv(1, n_feats, kernel_size),
                                   conv(n_feats, n_feats, kernel_size))
        self.convcat = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size))
        self.body1 = self.bodyn1 = nn.Sequential(common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale), nn.InstanceNorm2d(n_feats))
        self.tailn = nn.Sequential(
            common.Upsampler(conv, self.scale, n_feats, act=False),
            conv(n_feats, 1, kernel_size))
        self.tail = nn.Sequential(common.Upsampler(conv, self.scale, n_feats, act=False),
                                  conv(n_feats, args.n_colors, kernel_size))
        
        self.B1 = self.B2 = self.B3 = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size), common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale), common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        self.convf1 = self.convf2 = self.convf3 = nn.Sequential(conv(n_feats, n_feats, kernel_size))
        self.convim1 = self.convim2 = self.convim3 = nn.Sequential(common.Upsampler(conv, self.scale, n_feats, act=False),
                                                                   conv(n_feats, args.n_colors, kernel_size))
        self.var_conv = nn.Sequential(*[conv(3 * n_feats, n_feats, kernel_size), nn.ELU(),
                                        conv(n_feats, n_feats, kernel_size), nn.ELU(),
                                        # common.Upsampler(conv, self.scale, n_feats, act=False),
                                        conv(n_feats, args.n_colors, kernel_size), nn.ELU()])
        
    def forward(self, inp):
        (xrgb, xnir) = inp
        
        xr = self.head(xrgb)
        xn = self.headn(xnir)
        
        x1 = self.body1(xr)
        xn1 = self.bodyn1(xn)
        
        xc1 = self.B1(torch.cat([x1, xn1], 1))
        xf1 = self.convf1(xc1)
        sr1 = self.convim1(xf1 + xr)
        
        xc2 = self.B2(torch.cat([x1, xf1], 1))
        xf2 = self.convf2(xc2)
        sr2 = self.convim2(xf2 + xr)
        
        xc3 = self.B3(torch.cat([x1, xf2], 1))
        xf3 = self.convf3(xc3)
        sr3 = self.convim3(xf3 + xr)

        theta = self.var_conv(nn.functional.interpolate(torch.cat([xf1, xf2, xf3], 1), scale_factor=self.scale, mode='nearest'))
        # theta = self.var_conv(torch.cat([xf1, xf2, xf3], 1))  # [2, 3, 256, 256]
        resn = xf3 + xn
        xnsr = self.tailn(resn)
        
        return sr1, sr2, sr3, theta, xnsr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSR2branchuncertainty_twostage(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR2branchuncertainty_twostage, self).__init__()
        self.UnBranch = EDSR2branchuncertainty_stage1(args, conv)
        self.SRBranch = EDSR2branchuncertainty_stage1(args, conv)
        prepath = './experiment/test/%s/model_%s/model_best.pt' % (
        'EDSRcocrtinNIRw_uncertainty_stage1', 'EDSRcocrtinNIRw_uncertainty_stage1')
        if os.path.exists(prepath):
            if args.cpu:
                self.UnBranch.load_state_dict(torch.load(prepath, map_location=torch.device('cpu')), strict=True)
                self.SRBranch.load_state_dict(torch.load(prepath, map_location=torch.device('cpu')), strict=True)
            else:
                self.UnBranch.load_state_dict(torch.load(prepath), strict=True)
                self.SRBranch.load_state_dict(torch.load(prepath), strict=True)
            print('Load Stage 1 Model from', prepath)
        else:
            print('!!! No Stage 1 Model !!!')
            exit()
    
    def forward(self, inp):
        with torch.no_grad():
            _, _, _, theta, _ = self.UnBranch(inp)
        sr1, sr2, sr3, _, xnsr = self.SRBranch(inp)

        return sr1, sr2, sr3, theta, xnsr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSR2branchuncertainty_stage2(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR2branchuncertainty_stage2, self).__init__()
        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]
        self.cpu = args.cpu
        act = nn.ReLU(True)
        
        ## RGB branch
        self.head = nn.Sequential(conv(args.n_colors, n_feats, kernel_size), conv(n_feats, n_feats, kernel_size))
        self.headn = nn.Sequential(conv(1, n_feats, kernel_size),
                                   conv(n_feats, n_feats, kernel_size))
        self.body1 = self.bodyn1 = nn.Sequential(common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale), nn.InstanceNorm2d(n_feats))
        self.tailn = nn.Sequential(
            common.Upsampler(conv, self.scale, n_feats, act=False),
            conv(n_feats, 1, kernel_size))
        self.tail = nn.Sequential(common.Upsampler(conv, self.scale, n_feats, act=False),
                                  conv(n_feats, args.n_colors, kernel_size))
        
        self.B1 = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size), common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale), common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
        self.convf1 = self.convf2 = self.convf3 = nn.Sequential(conv(n_feats, n_feats, kernel_size))
        self.convnf1 = self.convnf2 = self.convnf3 = nn.Sequential(conv(n_feats, n_feats, kernel_size))
        self.convupim1 = self.convupim2 = self.convupim3 = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size),
            common.Upsampler(conv, self.scale, n_feats, act=False))
        self.convim1 = self.convim2 = self.convim3 = nn.Sequential(conv(n_feats, args.n_colors, kernel_size))
        
        self.B2 = self.B3 = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size), common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale), common.ResBlock(
            conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
            
    def forward(self, inp, theta):  # feature 扰动
        (xrgb, xnir) = inp
        
        xr = self.head(xrgb)
        xn = self.headn(xnir)
        xrup = nn.functional.interpolate(xr, scale_factor=self.scale, mode='nearest')

        x1 = self.body1(xr)
        xn1 = self.bodyn1(xn)
        
        xc1 = self.B1(torch.cat([x1, xn1], 1))
        xf1 = self.convf1(xc1)
        xnf1 = self.convnf1(xn1)
        xfup1 = self.convupim1(torch.cat([xf1, xnf1], 1))
        if self.cpu:
            normt1 = xfup1 + theta * torch.normal(0, 1, xfup1.size(), generator=None, out=None)
        else:
            normt1 = xfup1 + theta * torch.normal(0, 1, xfup1.size(), generator=None, out=None).cuda()
        sr1 = self.convim1(normt1 + xrup)
        
        xc2 = self.B2(torch.cat([x1, xnf1], 1))
        xf2 = self.convf2(xc2)
        xnf2 = self.convnf2(xnf1)
        xfup2 = self.convupim2(torch.cat([xf2, xnf2], 1))
        if self.cpu:
            normt2 = xfup2 + theta * torch.normal(0, 1, xfup2.size(), generator=None, out=None)
        else:
            normt2 = xfup2 + theta * torch.normal(0, 1, xfup2.size(), generator=None, out=None).cuda()
        sr2 = self.convim2(normt2 + xrup)

        xc3 = self.B3(torch.cat([x1, xnf2], 1))
        xf3 = self.convf3(xc3)
        xnf3 = self.convnf3(xnf2)
        xfup3 = self.convupim3(torch.cat([xf3, xnf3], 1))
        if self.cpu:
            normt3 = xfup3 + theta * torch.normal(0, 1, xfup3.size(), generator=None, out=None)
        else:
            normt3 = xfup3 + theta * torch.normal(0, 1, xfup3.size(), generator=None, out=None).cuda()
        sr3 = self.convim3(normt3 + xrup)
        
        resn = xnf3 + xn
        xnsr = self.tailn(resn)
        return sr1, sr2, sr3, xnsr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSR2branchuncertainty_twostage_f(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR2branchuncertainty_twostage_f, self).__init__()
        self.UnBranch = EDSR2branchuncertainty_stage1(args, conv)
        self.SRBranch = EDSR2branchuncertainty_stage2(args, conv)
        prepath = './experiment/test/%s/model_%s/model_best.pt' % ('EDSRcocrtinNIRw_uncertainty_stage1', 'EDSRcocrtinNIRw_uncertainty_stage1')
        if os.path.exists(prepath):
            if args.cpu:
                self.UnBranch.load_state_dict(torch.load(prepath, map_location=torch.device('cpu')), strict=True)
            else:
                self.UnBranch.load_state_dict(torch.load(prepath), strict=True)
            print('Load Stage 1 Model from', prepath)
        else:
            print('!!! No Stage 1 Model !!!')
            exit()
    
    def forward(self, inp):
        with torch.no_grad():
            _, _, _, theta, _ = self.UnBranch(inp)
            theta1 = torch.mean(theta, 1)
            theta1 = theta1.unsqueeze(1)
        sr1, sr2, sr3, xnsr = self.SRBranch(inp, theta1)
        
        return sr1, sr2, sr3, theta, xnsr
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


#### ------------------------------- NIPS 2021  UDL  --------------------------------######
class EDSRUDL(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSRUDL, self).__init__()

        n_resblock = 16
        n_feats = 64
        self.args = args
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.up_factor = args.scale[0]

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        self.var_conv = nn.Sequential(*[conv(n_feats, n_feats, kernel_size), nn.ELU(),
                                        conv(n_feats, n_feats, kernel_size), nn.ELU(),
                                        conv(n_feats, args.n_colors, kernel_size), nn.ELU()])

    def forward(self, inp):
        (xrgb, xnir) = inp

        x = self.head(xrgb)
        res = self.body(x)
        res += x
        x = self.tail(res)
        var = self.var_conv(nn.functional.interpolate(res, scale_factor=self.up_factor, mode='nearest'))
        return x, var

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


class EDSRUDL_two(nn.Module):
    def __init__(self, args):
        super(EDSRUDL_two, self).__init__()
        self.EDSR_var = EDSR(args=args)
        self.EDSR_U = EDSR(args=args)
        # if args.pre_train_step1 != '.':
        self.EDSR_var.load_state_dict(torch.load(args.pre_train_step1), strict=True)
        self.EDSR_U.load_state_dict(torch.load(args.pre_train_step1), strict=True)
    
    def forward(self, x):
        with torch.no_grad():  # 当前计算不需要反向传播，使用之后，强制后边的内容不进行计算图的构建
            var = self.EDSR_var(x)
        x = self.EDSR_U(x)
        x = x[0]
        return x, var[1]
    
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
