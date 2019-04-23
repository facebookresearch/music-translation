# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from pathlib import Path
import tempfile


def column_major(x):
    """
    PyTorch Tensors are row major, so this just returns a contiguous transpose
    """
    assert (x.is_contiguous)
    if len(x.size()) == 1:
        return x

    if len(x.size()) == 3:
        assert (x.size(2) == 1)
        x = torch.squeeze(x)

    if len(x.size()) == 2:
        return torch.t(x).contiguous()

    if len(x.size()) == 4:
        return x.permute(3, 2, 1, 0).contiguous()


def enum(**enums):
    return type('Enum', (), enums)


Impl = enum(AUTO=0, SINGLE_BLOCK=1, DUAL_BLOCK=2, PERSISTENT=3)


class NVWavenetGenerator:
    def __init__(self, model, sample_count, batch_size, implementation, cond_repeat=800):
        self.model = model
        self.cond_repeat = cond_repeat

        fname = Path(__file__)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            postfix = os.environ["CUDA_VISIBLE_DEVICES"].replace(',', '_')
        else:
            postfix = ""

        try:
            import nv_wavenet_ext
            self.wavenet_cu = nv_wavenet_ext
        except ImportError as e:
            print("Failed loading nv_wavenet_ext, building dynamically")
            build_name = "wavenet_cu_" + next(tempfile._get_candidate_names()) + postfix
            if not os.path.exists(".build_dir"):
                os.mkdir(".build_dir")

            if not os.path.exists('.build_dir/' + build_name):
                os.mkdir('.build_dir/' + build_name)

            self.wavenet_cu = load(name=build_name,
                                   sources=[fname.parent / "nv-wavenet/wavenet_infer.cu",
                                            fname.parent / "nv-wavenet/wavenet_infer_wrapper.cpp",
                                            fname.parent / "nv-wavenet/matrix.cpp"],
                                   build_directory='.build_dir/' + build_name,
                                   verbose=False,
                                   extra_cuda_cflags=["-arch=sm_70", "-std=c++14", "--use_fast_math",
                                                      "-maxrregcount 128", "--ptxas-options=-v",
                                                      "--expt-relaxed-constexpr", "-D__GNUC__=6"])

        embedding_prev, embedding_curr = model.export_embed_weights()
        conv_init_weight, conv_init_bias, conv_out_weight, \
        conv_out_bias, conv_end_weight, conv_end_bias = model.export_final_weights()
        dilate_weights, dilate_biases, res_weights, \
        res_biases, skip_weights, skip_biases = model.export_layer_weights()
        use_embed_tanh = False
        layers = len(model.layers)
        blocks = model.blocks
        max_dilation = 2 ** (layers // blocks - 1)

        self.R = self.wavenet_cu.num_res_channels()
        self.S = self.wavenet_cu.num_skip_channels()
        self.A = self.wavenet_cu.num_out_channels()

        self.max_dilation = max_dilation
        self.use_embed_tanh = use_embed_tanh
        assert embedding_prev.size() == (self.A, self.R), \
            ("embedding_prev: {} doesn't match compiled"
             " nv-wavenet size: {}").format(embedding_prev.size(),
                                            (self.A, self.R))
        self.embedding_prev = column_major(torch.t(embedding_prev))

        assert embedding_curr.size() == (self.A, self.R), \
            ("embedding_curr: {} doesn't match compiled"
             " nv-wavenet size: {}").format(embedding_curr.size(),
                                            (self.A, self.R))
        self.embedding_curr = column_major(torch.t(embedding_curr))

        assert conv_init_weight.size()[:2] == (self.S, self.R), \
            ("conv_init_weight: {} doesn't match compiled"
             " nv-wavenet size: {}").format(conv_init_weight.size()[:2],
                                            (self.S, self.R))
        self.conv_init = column_major(conv_init_weight)
        self.conv_init_bias = column_major(conv_init_bias)

        assert conv_out_weight.size()[:2] == (self.S, self.S), \
            ("conv_out_weight: {} doesn't match compiled"
             " nv-wavenet size: {}").format(conv_out_weight.size()[:2],
                                            (self.S, self.S))
        self.conv_out = column_major(conv_out_weight)
        self.conv_out_bias = column_major(conv_out_bias)

        assert conv_end_weight.size()[:2] == (self.A, self.S), \
            ("conv_end_weight: {} doesn't match compiled"
             " nv-wavenet size: {}").format(conv_end_weight.size()[:2],
                                            (self.A, self.S))
        self.conv_end = column_major(conv_end_weight)
        self.conv_end_bias = column_major(conv_end_bias)

        self.dilate_weights_prev = []
        self.dilate_weights_curr = []
        for weight in dilate_weights:
            assert weight.size(2) == 2, \
                "nv-wavenet only supports kernel_size 2"
            assert weight.size()[:2] == (2 * self.R, self.R), \
                ("dilated weight: {} doesn't match compiled"
                 " nv-wavenet size: {}").format(weight.size()[:2],
                                                (2 * self.R, self.R))
            Wprev = column_major(weight[:, :, 0])
            Wcurr = column_major(weight[:, :, 1])
            self.dilate_weights_prev.append(Wprev)
            self.dilate_weights_curr.append(Wcurr)

        for bias in dilate_biases:
            assert (bias.size(0) == 2 * self.R)
        for weight in res_weights:
            assert weight.size()[:2] == (self.R, self.R), \
                ("residual weight: {} doesn't match compiled"
                 " nv-wavenet size: {}").format(weight.size()[:2],
                                                (self.R, self.R))
        for bias in res_biases:
            assert (bias.size(0) == self.R), \
                ("residual bias: {} doesn't match compiled"
                 " nv-wavenet size: {}").format(bias.size(0), self.R)
        for weight in skip_weights:
            assert weight.size()[:2] == (self.S, self.R), \
                ("skip weight: {} doesn't match compiled"
                 " nv-wavenet size: {}").format(weight.size()[:2],
                                                (self.S, self.R))
        for bias in skip_biases:
            assert (bias.size(0) == self.S), \
                ("skip bias: {} doesn't match compiled"
                 " nv-wavenet size: {}").format(bias.size(0), self.S)

        self.dilate_biases = [column_major(bias) for bias in dilate_biases]
        self.res_weights = [column_major(weight) for weight in res_weights]
        self.res_biases = [column_major(bias) for bias in res_biases]
        self.skip_weights = [column_major(weight) for weight in skip_weights]
        self.skip_biases = [column_major(bias) for bias in skip_biases]

        # There's an extra residual layer that's not used
        # self.res_weights.append(torch.zeros(self.R,self.R))
        # self.res_biases.append(torch.zeros(self.R))

        assert (len(self.res_biases) == len(self.skip_biases) and
                len(self.res_biases) == len(self.dilate_biases) and
                len(self.res_weights) == len(self.skip_weights) and
                len(self.res_weights) == len(dilate_weights)), \
            """Number of layers is inconsistent for different parameter types.
        The list sizes should be the same for skip weights/biases and 
        dilate weights/biases.  Additionally the residual weights/biases
        lists should be one shorter.  But their sizes are:
        len(dilate_weights) = {}
        len(dilale_biases) = {}
        len(skip_weights) = {}
        len(skip_biases) = {}
        len(res_weights) = {}
        len(res_biases) = {}""".format(len(dilate_weights),
                                       len(self.dilate_biases),
                                       len(self.skip_weights),
                                       len(self.skip_biases),
                                       len(self.res_weights) - 1,
                                       len(self.res_biases) - 1)

        self.num_layers = len(res_biases)
        self.num_samples = sample_count
        self.max_num_samples = sample_count
        self.batch_size = batch_size
        self.nv_wavenet = self.wavenet_cu.construct(self.max_num_samples,
                                                    self.batch_size,
                                                    self.embedding_prev,
                                                    self.embedding_curr,
                                                    self.conv_init,
                                                    self.conv_init_bias,
                                                    self.conv_out,
                                                    self.conv_out_bias,
                                                    self.conv_end,
                                                    self.conv_end_bias,
                                                    self.dilate_weights_prev,
                                                    self.dilate_weights_curr,
                                                    self.dilate_biases,
                                                    self.res_weights,
                                                    self.res_biases,
                                                    self.skip_weights,
                                                    self.skip_biases,
                                                    self.num_layers,
                                                    self.use_embed_tanh,
                                                    self.max_dilation,
                                                    implementation)
        # build weights channel-major and not layer-major
        all_condition_weights = torch.stack([l.condition.weight for l in model.layers], dim=1)
        all_condition_biases = torch.stack([l.condition.bias for l in model.layers], dim=1)
        self.all_condition_weights = all_condition_weights.view(
            all_condition_weights.size(0) * all_condition_weights.size(1),
            all_condition_weights.size(2),
            all_condition_weights.size(3)
        )

        self.all_condition_biases = all_condition_biases.view(
            all_condition_biases.size(0) * all_condition_biases.size(1),
        )

    def prepare_cond(self, c):
        Lh = F.conv1d(c, self.all_condition_weights, self.all_condition_biases)
        Lh = Lh.view(c.size(0), 2 * self.R, self.num_layers, self.num_samples // self.cond_repeat)
        Lh = Lh.transpose(0, 1).contiguous()

        # Final condition
        LhFinal = self.model.condition(c)
        LhFinal = LhFinal.transpose(0, 1)[:, :, None].contiguous()

        return Lh, LhFinal

    def generate(self, cond_input, output_selectors=None):
        self.num_samples = cond_input.size(2) * self.cond_repeat
        assert self.num_samples <= self.max_num_samples
        samples = torch.cuda.IntTensor(self.batch_size, self.num_samples)
        cond_input, cond_final = self.prepare_cond(cond_input)
        if output_selectors is None:
            output_selectors = torch.rand(self.num_samples * self.batch_size, device=cond_input.device)

        # cond_input is channels x batch x num_layers x samples
        assert (cond_input.size()[0:3:2] == (2 * self.R, self.num_layers)), \
            """Inputs are channels x batch x num_layers x samples.
        Channels and num_layers should be sizes: {}
        But input is: {}""".format((2 * self.R, self.num_layers),
                                   cond_input.size()[0:3:2])

        # cond_final is channels x batch x samples
        assert (cond_final.size()[0:3:2] == (self.S, 1)), \
            """Inputs are channels x batch x samples.
            Channels and num_layers should be sizes: {}
            But input is: {}""".format((self.S, 1),
                                       cond_final.size()[0:3:2])

        cond_input = column_major(cond_input)
        cond_final = column_major(cond_final)
        self.wavenet_cu.infer(self.nv_wavenet,
                              samples,
                              cond_input,
                              cond_final,
                              output_selectors,
                              self.num_samples,
                              self.batch_size)
        return samples

    def reset(self):
        self.wavenet_cu.reset(self.nv_wavenet)

    def __del__(self):
        if self.wavenet_cu is not None:
            self.wavenet_cu.destruct(self.nv_wavenet)
