import os
from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale  # 超分辨路重建的 scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision  # 预测结果的浮点数精度
        self.cpu = args.cpu  # 如果传入cpu的外部参数，手动声明使用cpu，就用cpu，否则用gpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs  # gpu的个数，默认1个
        self.save_models = args.save_models  # 是否选择保存模型

        # 导入需要使用的预测模型
        # ! 但是我觉得这里有个问题，应该在 option 把 model 的默认值从 BSR 改成 MW
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)

        # 浮点格式的精度
        if args.precision == 'half': self.model.half()

        # 多GPU
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        # 导入外部的预训练模型
        self.load(ckp.dir,
                  pre_train=args.pre_train,
                  resume=args.resume,
                  name=args.model,
                  cpu=args.cpu)

        # 输出模型
        if args.print_model:
            print(self.model)

    # ? 没看懂
    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)

        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward
            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x)
            # return self.model(x)
        else:
            return self.model(x)

    # 根据 GPU 个数导入训练模型
    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    # 对模型进行参数映射
    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    # 保存模型
    def save(self, apath, epoch, name, is_best=False):
        target = self.get_model()
        torch.save(target.state_dict(),
                   os.path.join(apath, 'model', name + 'model_latest.pt'))
        if is_best:
            torch.save(target.state_dict(),
                       os.path.join(apath, 'model', name + 'model_best.pt'))
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model',
                             name + 'model_{}.pt'.format(epoch)))

    # 导入模型
    def load(self, apath, pre_train='.', resume=-1, name='', cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            self.get_model().load_state_dict(torch.load(
                os.path.join(pre_train, name + 'model_latest.pt'), **kwargs),
                                             strict=False)

            # self.get_model().load_state_dict(
            #     torch.load(
            #         os.path.join(apath, 'model', name + 'model_latest.pt'),
            #         **kwargs
            #     ),
            #     strict=False
            # )

        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(torch.load(
                    pre_train, **kwargs),
                                                 strict=False)
        else:
            self.get_model().load_state_dict(torch.load(
                os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                **kwargs),
                                             strict=False)

    def forward_chop(self, x, shave=10, min_size=160000):
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size], x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]
        ]
        # lr_list = [
        #     x[:, :, 0:h_size, 0:w_size],
        #     x[:, :, 0:h_size, (w - w_size):w],
        #     x[:, :, (h - h_size):h, 0:w_size],
        #     x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
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

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output
