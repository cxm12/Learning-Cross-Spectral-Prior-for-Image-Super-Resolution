import torch
import torch.nn as nn
import torch.nn.functional as F


class attention1d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention1d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv1d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv1d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention1d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height = x.size()
        x = x.view(1, -1, height, )# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size,)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-1))
        return output


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)  # 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


# 2 feature input, learn weight to conv 1 feature
class Dynamic_conv2d_2f(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, K=4, temperature=34, init_weight=True):
        super(Dynamic_conv2d_2f, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(2*in_planes, ratio, K, temperature)
        
        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        # TODO 初始化
    
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])
    
    def update_temperature(self):
        self.attention.updata_temperature()
    
    def forward(self, x, y):  # 将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        f = torch.concat([x, y], dim=1)
        softmax_attention = self.attention(f)
        batch_size, in_planes, height, width = x.size()
        weight = self.weight.view(self.K, -1)
        
        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size * self.out_planes,
                                                                    self.in_planes // self.groups, self.kernel_size,
                                                                    self.kernel_size)
        
        x = x.view(1, -1, height, width)  # 变化成一个维度进行组卷积
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class attention3d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature):
        super(attention3d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv3d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv3d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)


class Dynamic_conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34):
        super(Dynamic_conv3d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention3d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None


        #TODO 初始化
        # nn.init.kaiming_uniform_(self.weight, )

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, depth, height, width = x.size()
        x = x.view(1, -1, depth, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv3d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv3d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-3), output.size(-2), output.size(-1))
        return output


#############################  SRModel  ################################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Dynamic_conv2d_2f(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return Dynamic_conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


from model import common


class EDSR2branch1Dynamic(nn.Module):
    def __init__(self, args, conv=common.default_conv, name=''):
        super(EDSR2branch1Dynamic, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.scale = args.scale[0]
        act = nn.ReLU(True)
        
        ## RGB branch
        self.head = nn.Sequential(conv(args.n_colors, n_feats, kernel_size), conv(n_feats, n_feats, kernel_size))
        self.conv = nn.Sequential(conv(n_feats, n_feats, kernel_size))
        self.dyconv1 = self.dyconv2 = self.dyconv3 = conv3x3(n_feats, n_feats, stride=1)
        self.convcat1 = self.convcat2 = self.convcat3 = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size))
        
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
        if not name == 'v0':
            self.convcatn1 = self.convcatn2 = self.convcatn3 = nn.Sequential(conv(2 * n_feats, n_feats, kernel_size))
        # CMFT
        self.RN3 = self.RN2 = self.RN1 = common.NIRRGBBlock1(conv, n_feats, kernel_size)
    
    def forward(self, inp):
        (rgb, nir) = inp
        xr = self.head(rgb)
        xn = self.headn(nir)
        
        xr1 = self.conv(xr)
        x1 = self.body1(xr1)
        xnd1 = self.dyconv1(xn, x1)
        xn1 = self.bodyn1(self.convcatn1(torch.cat([xnd1, xn], 1)))
        xc1 = self.RN1(x1, xn1)
        
        xr2 = self.convcat1(torch.cat([xc1, x1], 1))
        x2 = self.body2(xr2)
        xnd2 = self.dyconv2(xnd1, x2)
        xn2 = self.bodyn2(self.convcatn2(torch.cat([xnd2, xn1], 1)))
        xc2 = self.RN2(x2, xn2)
        
        xr3 = self.convcat2(torch.cat([xc2, x2], 1))
        x3 = self.body3(xr3)
        xnd3 = self.dyconv3(xnd2, x3)
        xn3 = self.bodyn3(self.convcatn3(torch.cat([xnd3, xn2], 1)))
        xc3 = self.RN3(x3, xn3)
        
        x4 = self.body4(self.convcat3(torch.cat([xc3, x3], 1)))
        resr = x4 + xr
        bic = torch.nn.Upsample(scale_factor=self.scale, mode='bicubic')(rgb)
        xrsr = self.tail(resr) + bic
        
        xn4 = self.bodyn4(xn3)
        resn = xn4 + xn
        bicn = torch.nn.Upsample(scale_factor=self.scale, mode='bicubic')(nir)
        xnsr = self.tailn(resn) + bicn
        
        return xrsr, xnsr

    def forward0(self, inp):
        (rgb, nir) = inp
        xr = self.head(rgb)
        xn = self.headn(nir)
    
        xr1 = self.conv(xr)
        x1 = self.body1(xr1)
        xnd1 = self.dyconv1(xn, x1)
        xn1 = self.bodyn1(xnd1)
        xc1 = self.RN1(x1, xn1)
    
        xr2 = self.convcat1(torch.cat([xc1, x1], 1))
        x2 = self.body2(xr2)
        xnd2 = self.dyconv2(xnd1, x2)
        xn2 = self.bodyn2(xnd2)
        xc2 = self.RN2(x2, xn2)
    
        xr3 = self.convcat2(torch.cat([xc2, x2], 1))
        x3 = self.body3(xr3)
        xnd3 = self.dyconv3(xnd2, x3)
        xn3 = self.bodyn3(xnd3)
        xc3 = self.RN3(x3, xn3)
    
        x4 = self.body4(self.convcat3(torch.cat([xc3, x3], 1)))
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
