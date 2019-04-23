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
#include "wavenet_infer.h"
#include <iostream>
#include <string>
#include <memory>
#include "nv_wavenet.cuh"
#include "matrix.h"

// Must match the wavenet channels
const int A = 256;
const int R = 128;
const int S = 128;
const int COND_REPEAT = 800; // must match encoder_pool
typedef nvWavenetInfer<half2,half, R, S, A, COND_REPEAT> MyWaveNet;
//typedef nvWavenetInfer<float,float, R, S, A, COND_REPEAT> MyWaveNet;


void* wavenet_construct(int sample_count,
                        int batch_size,
                        float* embedding_prev,
                        float* embedding_curr,
                        int num_layers,
                        int max_dilation,
                        float** in_layer_weights_prev,
                        float** in_layer_weights_curr,
                        float** in_layer_biases,
                        float** res_layer_weights,
                        float** res_layer_biases,
                        float** skip_layer_weights,
                        float** skip_layer_biases,
                        float* conv_init_weight,
                        float* conv_init_bias,
                        float* conv_out_weight,
                        float* conv_out_bias,
                        float* conv_end_weight,
                        float* conv_end_bias,
                        int use_embed_tanh,
                        int implementation
                        ) {
		MyWaveNet* wavenet = new MyWaveNet(num_layers, 
                                           max_dilation,
                                           batch_size, 
                                           sample_count,
                                           implementation,
                                           use_embed_tanh
                                           );
    
        wavenet->setEmbeddings(embedding_prev, embedding_curr);

        for (int l = 0; l < num_layers; l++) {
            wavenet->setLayerWeights(l, in_layer_weights_prev[l], 
                                        in_layer_weights_curr[l],
                                        in_layer_biases[l],
                                        res_layer_weights[l],
                                        res_layer_biases[l],
                                        skip_layer_weights[l],
                                        skip_layer_biases[l]);
        }

        wavenet->setOutWeights(conv_init_weight,
                               conv_init_bias,
                               conv_out_weight,
                               conv_out_bias,
                               conv_end_weight,
                               conv_end_bias);

        return (void*) wavenet;
}	

void wavenet_infer(void* wavenet,
                   int* samples,
                   float* cond_input,
                   float* cond_final,
                   float* output_selectors,
                   int sample_count,
                   int batch_size) {
    assert(samples);
    
    MyWaveNet* myWaveNet = (MyWaveNet*) wavenet;
    myWaveNet->setInputs(cond_input, cond_final, output_selectors, sample_count, batch_size);

    int batch_size_per_block = ((batch_size % 4) == 0) ? 4 : ((batch_size % 2) == 0) ? 2 : 1;
    assert(myWaveNet->run(sample_count, batch_size, samples, batch_size_per_block, false));
    gpuErrChk(cudaDeviceSynchronize());
    return;
}

void wavenet_getZa(void* wavenet, float* Za) {
    MyWaveNet* myWaveNet = (MyWaveNet*) wavenet;
    myWaveNet->getZa(Za);
    gpuErrChk(cudaDeviceSynchronize());
}

void wavenet_reset(void* wavenet) {
    MyWaveNet* myWaveNet = (MyWaveNet*) wavenet;
    myWaveNet->reset();
    gpuErrChk(cudaDeviceSynchronize());
}

void wavenet_destruct(void* wavenet) {
    MyWaveNet* myWaveNet = (MyWaveNet *) wavenet;
    delete myWaveNet;
}

int get_R() {return R;}	
int get_S() {return S;}	
int get_A() {return A;}	
