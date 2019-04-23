/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#include <torch/torch.h>
#include "wavenet_infer.h"
#include <cstdint>

uint64_t construct(int sample_count,
                   int batch_size,
                   at::Tensor embed_prev_tensor,
                   at::Tensor embed_curr_tensor,
                   at::Tensor conv_init_tensor,
                   at::Tensor conv_init_bias_tensor,
                   at::Tensor conv_out_tensor,
                   at::Tensor conv_out_bias_tensor,
                   at::Tensor conv_end_tensor,
                   at::Tensor conv_end_bias_tensor,
                   std::vector<at::Tensor>& Wprev, std::vector<at::Tensor>& Wcur,std::vector<at::Tensor>& Bh,
                   std::vector<at::Tensor>& Wres, std::vector<at::Tensor>& Bres,
                   std::vector<at::Tensor>& Wskip, std::vector<at::Tensor>& Bskip,
                   int num_layers,
                   int use_embed_tanh,
                   int max_dilation,
                   int implementation) {
    float* embedding_prev = embed_prev_tensor.data<float>();
    float* embedding_curr = embed_curr_tensor.data<float>();
    float* conv_init = conv_init_tensor.data<float>();
    float* conv_init_bias = conv_init_bias_tensor.data<float>();
    float* conv_out = conv_out_tensor.data<float>();
    float* conv_out_bias = conv_out_bias_tensor.data<float>();
    float* conv_end = conv_end_tensor.data<float>();
    float* conv_end_bias = conv_end_bias_tensor.data<float>();

    float** in_layer_weights_prev = (float**) malloc(num_layers*sizeof(float*));
    float** in_layer_weights_curr = (float**) malloc(num_layers*sizeof(float*));
    float** in_layer_biases = (float**) malloc(num_layers*sizeof(float*));
    float** res_layer_weights = (float**) malloc(num_layers*sizeof(float*));
    float** res_layer_biases = (float**) malloc(num_layers*sizeof(float*));
    float** skip_layer_weights = (float**) malloc(num_layers*sizeof(float*));
    float** skip_layer_biases = (float**) malloc(num_layers*sizeof(float*));
    for (int i=0; i < num_layers; i++) {
        in_layer_weights_prev[i] = Wprev[i].data<float>();
        in_layer_weights_curr[i] = Wcur[i].data<float>();
        in_layer_biases[i] = Bh[i].data<float>();
        res_layer_weights[i] = Wres[i].data<float>();
        res_layer_biases[i] = Bres[i].data<float>();
        skip_layer_weights[i] = Wskip[i].data<float>();
        skip_layer_biases[i] = Bskip[i].data<float>();
    }

    void* wavenet = wavenet_construct(sample_count,
                                      batch_size,
                                      embedding_prev,
                                      embedding_curr,
                                      num_layers,
                                      max_dilation,
                                      in_layer_weights_prev,
                                      in_layer_weights_curr,
                                      in_layer_biases,
                                      res_layer_weights,
                                      res_layer_biases,
                                      skip_layer_weights,
                                      skip_layer_biases,
                                      conv_init,
                                      conv_init_bias,
                                      conv_out,
                                      conv_out_bias,
                                      conv_end,
                                      conv_end_bias,
                                      use_embed_tanh,
                                      implementation
                                     );
    
    free(in_layer_weights_prev);
    free(in_layer_weights_curr);
    free(in_layer_biases);
    free(res_layer_weights);
    free(res_layer_biases);
    free(skip_layer_weights);
    free(skip_layer_biases);
    
    return reinterpret_cast<uint64_t>(wavenet);
}

int infer(uint64_t wavenet,
          at::Tensor samples_tensor,
          at::Tensor cond_input_tensor,
          at::Tensor cond_final_tensor,
          at::Tensor output_selectors_tensor,
          int sample_count,
          int batch_size) {
    py::gil_scoped_release release;
    int* samples = samples_tensor.data<int>();
    float* cond_input = cond_input_tensor.data<float>();
    float* cond_final = cond_final_tensor.data<float>();
    float* output_selectors = output_selectors_tensor.data<float>();
    wavenet_infer((void*) wavenet, samples, cond_input, cond_final, output_selectors, sample_count, batch_size);
    return 1;
}

int getZa(uint64_t wavenet, at::Tensor Za_tensor) {
    float* za_tensor = Za_tensor.data<float>();
    wavenet_getZa((void*) wavenet, za_tensor);
    return 1;
}

int reset(uint64_t wavenet) {
    wavenet_reset((void*) wavenet);
    return 1;
}

int destruct(uint64_t wavenet) {
    wavenet_destruct((void*) wavenet);
    return 1;
}

int num_res_channels(void){return get_R();}
int num_skip_channels(void){return get_S();}
int num_out_channels(void){return get_A();}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("construct", &construct, "Constructs wavenet.");
    m.def("infer", &infer, "Runs wavenet inference.");
    m.def("getZa", &getZa, "Get Za activations.");
    m.def("reset", &reset, "Reset inference buffers.");
    m.def("destruct", &destruct, "Destructs wavenet.");
    m.def("num_res_channels", &num_res_channels, "Returns resodiual channel size.");
    m.def("num_skip_channels", &num_skip_channels, "Returns skip channel size.");
    m.def("num_out_channels", &num_out_channels, "Returns out channel size.");  
}
