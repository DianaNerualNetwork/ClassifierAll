import torch
import torch.nn as nn 


class VGG(nn.Moudle):
    
    # Parameters to build layers. Each element specifies the number of conv in
    # each stage. For example, VGG11 contains 11 layers with learnable
    # parameters. 11 is computed as 11 = (1 + 1 + 2 + 2 + 2) + 3,
    # where 3 indicates the last three fully-connected layers.
    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }

    def __init__(self,
                 layers,
                 num_classes=-1,
                 num_stages=5,
                 dilations=(1, 1, 1, 1, 1),
                 out_indices=None,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 ceil_mode=False,
                 with_last_pool=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(type='Constant', val=1., layer=['_BatchNorm']),
                     dict(type='Normal', std=0.01, layer=['Linear'])
                 ]):
        super(VGG, self).__init__(init_cfg)
        if layers not in self.arch_settings:
            raise KeyError(f'invalid depth {layers} for vgg')
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[layers]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages

        self.num_classes = num_classes
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        with_norm = norm_cfg is not None

        if out_indices is None:
            out_indices = (5, ) if num_classes > 0 else (4, )
        assert max(out_indices) <= num_stages
        self.out_indices = out_indices

        self.in_channels = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            out_channels = 64 * 2**i if i < 4 else 512
            vgg_layer = make_vgg_layer(
                self.in_channels,
                out_channels,
                num_blocks,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                dilation=dilation,
                with_norm=with_norm,
                ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            self.in_channels = out_channels
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        if not with_last_pool:
            vgg_layers.pop(-1)
            self.range_sub_modules[-1][1] -= 1
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))

        # self.adpool=
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(512 , 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i in range(len(self.stage_blocks)):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outs.append(x)

        return tuple(outs)