import os
import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        self.args = args
        self.resume = args.resume
        if 'SISR' in args.modelname:
            if 'HAT' in args.modelname:
                import model.HAT as SISR
                self.model = SISR.HAT(upscale=args.scale[0]).to(self.device)
            else:
                import model.enlcn.enlcn as SISR
                self.model = SISR.ENLCN(args, cat='cat' in args.modelname).to(self.device)
        elif 'Dynamic_nirscene' in args.modelname:
            import model.dynamicConv as module
            name = args.modelname[-2:]
            self.model = module.EDSR2branch1Dynamic(args, name=name).to(self.device)
        else:  # module = import_module('model.' + args.model.lower())
            import model.edsr as module
            self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()

        # # self.load(ckp.get_path('model'),
        # #             pre_train=args.pre_train,
        # #             resume=args.resume,
        # #             cpu=args.cpu)
        # self.loadAuto(ckp.get_path('model'), cpu=args.cpu)
        
        # SISR test set
        kwargs = {}
        load_from = torch.load(self.args.pre_train, **kwargs)
        self.model.load_state_dict(load_from, strict=False)
        # SISR test set
        
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        apath = apath + '%s/' % self.args.modelname
        
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            print('~~~~~~~~~~~~~~~~~~~ save to ', s)
            # torch.save(self.model.state_dict(), s, _use_new_zipfile_serialization=False)
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def loadAuto(self, apath, cpu=False):
        kwargs = {}
        
        apath = apath + '_%s/' % self.args.modelname
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        # print('----------------- apath ---------------------------', apath)
        if self.resume > 0 and os.path.exists(os.path.join(apath, 'model_%d.pt' % self.resume)):
            print('Load From ', os.path.join(apath, 'model_%d.pt' % self.resume))
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(self.resume)),
                **kwargs
            )
        elif self.resume == -2 and os.path.exists(os.path.join(apath, 'model_best.pt')):
            load_from = torch.load(
                os.path.join(apath, 'model_best.pt'),
                **kwargs
            )
            print('Load From ', os.path.join(apath, 'model_best.pt'))
        elif self.resume == -1 and os.path.exists(os.path.join(apath, 'model_latest.pt')):
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
            print('Load From ', os.path.join(apath, 'model_latest.pt'))
        elif self.resume == 0 and self.args.pre_train != '.':
            print('Load the model from {}'.format(self.args.pre_train))
            load_from = torch.load(self.args.pre_train, **kwargs)
        else:
            load_from = None
            print('None Load the model')

        if load_from:
            self.model.load_state_dict(load_from, strict=False)
            # torch.save(self.model.state_dict(), os.path.join(apath, 'model_best1.pt'), _use_new_zipfile_serialization=False)
            # exit()

    def forward_chop(self, *args, shave=10, min_size=160000):
        # scale = 1 if self.input_large else self.scale[self.idx_scale]
        scale = self.scale[self.idx_scale]
        # height, width
        h, w = args[0][0].size()[-2:]

        top = slice(0, (h//2 + shave) // 16 * 16)
        bottom = slice(h - ((h//2 + shave) // 16 * 16), h)
        left = slice(0, (w//2 + shave) // 16 * 16)
        right = slice(w - ((w//2 + shave) // 16 * 16), w)
        x_chops = [torch.cat([
            a[0][..., top, left],
            a[0][..., top, right],
            a[0][..., bottom, left],
            a[0][..., bottom, right]
        ]) for a in args]
        xnir_chops = [torch.cat([
            a[1][..., top, left],
            a[1][..., top, right],
            a[1][..., bottom, left],
            a[1][..., bottom, right]
        ]) for a in args]

        y_chops = []
        ynir_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4):
                x = [x_chop[i:(i + 1)] for x_chop in x_chops]
                xnir = [xnir_chop[i:(i + 1)] for xnir_chop in xnir_chops]
                y, ynir = self.model.forward((x[0], xnir[0]))
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(1, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(1, dim=0))
                if not ynir_chops:
                    ynir_chops = [[c for c in _y.chunk(1, dim=0)] for _y in ynir]
                else:
                    for yn_chop, _yn in zip(ynir_chops, ynir):
                        yn_chop.extend(_yn.chunk(1, dim=0))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0]

        return y

    def forward_chopor(self, *args, shave=10, min_size=160000):
        # scale = 1 if self.input_large else self.scale[self.idx_scale]
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        # height, width
        h, w = args[0].size()[-2:]
        
        top = slice(0, h//2 + shave)
        bottom = slice(h - h//2 - shave, h)
        left = slice(0, w//2 + shave)
        right = slice(w - w//2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]
        
        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                y = P.data_parallel(self.model, *x, range(n_GPUs))
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h//2)
        bottom = slice(h - h//2, h)
        bottom_r = slice(h//2 - h, None)
        left = slice(0, w//2)
        right = slice(w - w//2, w)
        right_r = slice(w//2 - w, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0]

        return y

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y
