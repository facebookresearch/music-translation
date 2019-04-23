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

__device__ __forceinline__ bool isNegativeZero(float a) {
    int ret;
    asm volatile("{  set.eq.s32.b32 %0, %1, %2;}\n" : "=r"(ret) : "f"(a), "r"(0x80000000));
    return ret;
}

__device__ __forceinline__ bool isNegativeZero(half a){
    const __half_raw* a_raw_ptr = (reinterpret_cast<const __half_raw *>(&a) );
    int ret;
    asm volatile("{  set.eq.s32.b32 %0, %1, %2;}\n" : "=r"(ret) : "r"(0x0u + (*a_raw_ptr).x), "r"(0x00008000));
    return ret;
}

__device__ __forceinline__ float validate(float a) {
    return isNegativeZero(a) ? 0.f : a;
}

__device__ __forceinline__ half validate(half a) {
    return isNegativeZero(a) ? (half)0.f : a;
}

__device__ __forceinline__ void storeValidate(volatile half* y, int index, half val) {
    half* y_nv = (half*)y;
    y_nv[index] = validate(val);
}

__device__ __forceinline__ void storeValidate(volatile float* y, int index, float val) {
    y[index] = validate(val);
}

template <typename T_data, int R>
__global__ void initializeActivations(T_data* xt, T_data* h_out) {
    assert(blockDim.x == R);

    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    xt[offset] = -0.f;
    h_out[offset] = -0.f;
}

template <typename T_data>
__global__ void initializeActivationsGeneric(T_data* skipIn) {
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    skipIn[offset] = -0.f;
}

template <typename T_data>
__global__ void initializeAPrev(T_data* a_prev, T_data* B, T_data* L) {
    int offset = (blockIdx.x+blockIdx.y*gridDim.x)*blockDim.x + threadIdx.x;
    a_prev[offset] = L[offset] + B[blockIdx.x*blockDim.x + threadIdx.x];
}

// Make sure all necessary clears are completed before processing a new sample.  Lock is per batch index.
template <int BATCH_UNROLL> 
__device__ __inline__ void sampleLockAcquire(int batch_offset, int sample, volatile int* sampleLock){
    if (threadIdx.x == 0) {
        bool valid = false;
        while (!valid) {
            valid = true;
#pragma unroll
            for (int u=0; u<BATCH_UNROLL; u++) {  
                valid &= (sampleLock[batch_offset+u]>=sample);
            }
        }
    }
    __syncthreads();
}

template <typename T_weight, typename T_data, int R, int BATCH_UNROLL, int barrier, int COND_REPEAT>
__device__ void nv_wavenet_persistent_prev_block(int row, int start_sample, int num_samples, volatile int* ySample, volatile int* hSample, int start_layer, int max_layer, int num_layers, int batch_size, int maxDilation, T_weight* Wprev, T_data* B, T_data* L, T_data* a_prev, volatile T_data* xt) {
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[R/WV];
    T_data accum[BATCH_UNROLL];
    T_data xtmd_reg[BATCH_UNROLL];
    __shared__ T_data xtmd_sh[2][BATCH_UNROLL][R];
    int startDilation = 1;
    for (int i=1; i <= start_layer; i++) {
        startDilation = startDilation << 1;
        if (startDilation > maxDilation) startDilation = 1;
    }

    if (row < 2*R) {
        for (int sample=start_sample; sample < start_sample + num_samples; sample++) {
            int ping = (~sample) & 1; // a_prev pipelining - which buffer do we write to?
            for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
                sampleLockAcquire<BATCH_UNROLL>(batch_offset,sample,ySample);
                int dilation = startDilation;
                for (int layer = start_layer; layer < max_layer; layer++) {
                    T_data conditioning[BATCH_UNROLL];
#pragma unroll
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        // add conditioning and bias in pipelined calculation
                        conditioning[b] = L[((sample+1-start_sample)/COND_REPEAT)*num_layers*batch_size*2*R + layer*batch_size*2*R + (batch_offset+b)*2*R + row];
                        conditioning[b] += B[layer*2*R+row];
                    }
                    int sample_offset = (sample + 1 - dilation) % (maxDilation+1); // note: doing _prev calculation for next sample, pipelined with the rest
                    volatile T_data* xtmd = xt + sample_offset*(num_layers+1)*R*batch_size;
                    if (row < R) {
                        if (dilation == 1) { // we need to wait for current sample processing
                            bool valid = false;
                            while (!valid) {
                                valid = true;
#pragma unroll
                                for (int b=0; b<BATCH_UNROLL; b++) {
                                    xtmd_reg[b] = (dilation <= (sample+1)) ? loadVolatile(xtmd,layer*batch_size*R + (batch_offset+b)*R + row) : (T_data)0.f;
                                }
#pragma unroll
                                for (int b=0; b<BATCH_UNROLL; b++) {
                                    valid &= !isNegativeZero(xtmd_reg[b]);
                                }
                            }
                        }
                        else { // result already exists from previous sample
#pragma unroll
                            for (int b=0; b<BATCH_UNROLL; b++) {
                                xtmd_reg[b] = (dilation <= (sample+1)) ? loadVolatile(xtmd,layer*batch_size*R + (batch_offset+b)*R + row) : (T_data)0.f;
                            }
                        }
#pragma unroll
                        for (int b=0; b<BATCH_UNROLL; b++) {
                            xtmd_sh[layer&1][b][row] = xtmd_reg[b];
                        }
                    }
                    loadWeights<2*R,R>(weights,Wprev,layer,row);
                    namedBarrierSync(barrier, 2*R);
                    GEMM<R,2,BATCH_UNROLL>(weights, xtmd_sh[layer&1], accum);
#pragma unroll
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        accum[b] += conditioning[b];
                        a_prev[layer*batch_size*2*R + (batch_offset+b)*2*R + ping*num_layers*batch_size*2*R + row] = accum[b];
                    }
                    dilation = dilation << 1;
                    if (dilation > maxDilation) dilation = 1;
                }
                __threadfence();
                namedBarrierSync(barrier, 2*R);
                if (threadIdx.x == 0) {
#pragma unroll
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        hSample[batch_offset+b] = sample + 1;
                    }
                }
            }
        }
    }
}

// NOTE: it's a good idea that # of blocks doesn't divide number of blocks in wavenet
//       since that's going to have all the blocks wait for dilation-1 results
constexpr int NUM_PREV_BLOCKS = 20;
template <typename T_weight, typename T_data, int R, int BATCH_UNROLL, int barrier, int COND_REPEAT>
__device__ void nv_wavenet_persistent_prev_multilayer(int block_id, int row, int start_sample, int num_samples, volatile int* ySample, volatile int* hSample, int num_layers, int batch_size, int maxDilation, T_weight* Wprev, T_data* B, T_data* L, T_data* a_prev, volatile T_data* xt) {
    int start_layer = (block_id*num_layers + (NUM_PREV_BLOCKS/2)) / NUM_PREV_BLOCKS;
    int end_layer = block_id == (NUM_PREV_BLOCKS-1) ? num_layers : ((block_id+1)*num_layers + (NUM_PREV_BLOCKS/2)) / NUM_PREV_BLOCKS;
    //if (row == 0) printf("pm, sl: %d, el: %d\n", start_layer, end_layer);
    //else if (start_layer < 2) printf("pm sl %d, r %d\n", start_layer, row);
    nv_wavenet_persistent_prev_block<T_weight,T_data,R,BATCH_UNROLL,barrier, COND_REPEAT>(row, start_sample, num_samples, ySample, hSample, start_layer, end_layer, num_layers, batch_size, maxDilation, Wprev, B, L, a_prev, xt);
}

template <typename T_weight, typename T_data, int R, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_cur(int row, int start_sample, int num_samples, volatile int* ySample, int layer, int num_layers, int batch_size, int maxDilation, T_weight* Wcur, volatile T_data* a_prev, volatile T_data* xt, T_data xt_sh[BATCH_UNROLL][R], T_data a_sh[BATCH_UNROLL][2*R], int* yInPrev, int* yInCur, T_data* embedPrev, T_data* embedCur, bool tanhEmbed) {
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[R/WV];
    loadWeights<2*R,R>(weights,Wcur,layer,row);
    T_data accum[BATCH_UNROLL];
    for (int sample=start_sample; sample < start_sample + num_samples; sample++) {
        int ping_pong = sample & 1; // a_prev pipelining - which buffer do we read from?
        volatile T_data* Xt = xt + (sample%(maxDilation+1))*(num_layers+1)*R*batch_size;
        for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
            sampleLockAcquire<BATCH_UNROLL>(batch_offset, sample, ySample);
            T_data conditioning[BATCH_UNROLL];
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) {
                conditioning[b] = loadVolatile(a_prev,layer*batch_size*2*R + (batch_offset+b)*2*R + row + ping_pong*num_layers*batch_size*2*R);//a_prev_reg[b];
            }

            if (row < R) {
                if (layer == 0) {
                    // Embedding
                    int yPrev[BATCH_UNROLL];
                    int yCur[BATCH_UNROLL];
#pragma unroll
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        yPrev[b] = yInPrev[batch_offset+b];
                        yCur[b] = yInCur[batch_offset+b];
                        T_data embedded = embedPrev[yPrev[b]*R + row] + embedCur[yCur[b]*R + row];
                        if (tanhEmbed) embedded = _tanh(embedded);
                        xt_sh[b][row] = embedded;
                        storeValidate(Xt, layer*batch_size*R + (batch_offset+b)*R + row, embedded);
                    }
                    // Make Xt visible before we write h, so that clears don't race ahead
                    // This is only needed for the embedding write, since it's read by the same block -- 
                    //  all other Xt writes get read by different blocks before they write h.  Since
                    //  the clears depend on h, then we know that the Xt writes are globally visible.
                    __threadfence();
                } else {
                    T_data xt_in[BATCH_UNROLL];
                    bool valid = false;
                    int xt_offset = layer*batch_size*R + batch_offset*R + row;
                    while (!valid) {
                        valid = true;
#pragma unroll
                        for (int b=0; b<BATCH_UNROLL; b++) {
                            xt_in[b] = loadVolatile(Xt,xt_offset+b*R);
                        }
#pragma unroll
                        for (int b=0; b<BATCH_UNROLL; b++) {
                            valid &= !isNegativeZero(xt_in[b]);
                        }
                    }
#pragma unroll
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        xt_sh[b][row] = xt_in[b];
                    }
                }
            }
            namedBarrierSync(4,2*R); // xt_sh produced
            GEMM<R,4,BATCH_UNROLL>(weights,xt_sh,accum);
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) {
                accum[b] += conditioning[b];
                //T_data val = (row < R) ? _tanh(accum[b]) : sigmoid(accum[b]);
                T_data val = (row < R) ? sigmoid(accum[b]) : _tanh(accum[b]);
                a_sh[b][row] = val;
            }
            namedBarrierArrive(1,3*R); // a_sh produced
        }
    }
}

template <typename T_weight, typename T_data, int R, int S, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_res(int row, int start_sample, int num_samples, volatile int* ySample, int layer, int num_layers, int batch_size, int maxDilation, T_weight* Wres, T_data* Bres, T_data* xt, T_data xt_sh[BATCH_UNROLL][R], T_data a_sh[BATCH_UNROLL][2*R], T_data h_sh[BATCH_UNROLL][R], T_data* xtOut, bool dumpActivations) {
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[R/WV];
    T_data bias;
    T_data accum[BATCH_UNROLL];
    loadWeights<R,R>(weights,Wres,layer,row);
    bias = Bres[layer*R+row];
    for (int sample=start_sample; sample < start_sample + num_samples; sample++) {
        volatile T_data* Xt = xt + (sample%(maxDilation+1))*(num_layers+1)*R*batch_size;
        for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
            sampleLockAcquire<BATCH_UNROLL>(batch_offset, sample, ySample);
            namedBarrierSync(1, 3*R); // a_sh produced
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) {
                T_data val = a_sh[b][row] * a_sh[b][row+R];
                h_sh[b][row] = val;
            }
            namedBarrierSync(2,R+S); // h_sh produced, a_sh consumed
            GEMM<R,4,BATCH_UNROLL>(weights,h_sh,accum);
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) {
                //accum[b] += xt_reg[b];
                accum[b] += xt_sh[b][row];
                accum[b] += bias;
                storeVolatile(Xt,(layer+1)*batch_size*R + (batch_offset+b)*R + row, accum[b]);//validate?
                if (dumpActivations) xtOut[layer*batch_size*R + (batch_offset+b)*R + row] = accum[b];
            }
        }
    }
}

template <typename T_weight, typename T_data, int R, int S, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_skip(int row, int start_sample, int num_samples, volatile int* ySample, int layer, int num_layers, int batch_size, T_weight* Wskip, T_data* Bskip, T_data h_sh[BATCH_UNROLL][R], volatile T_data* accum_in) {
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[S/WV];
    loadWeights<S,R>(weights,Wskip,layer,row,S);
    T_data accum[BATCH_UNROLL];
    T_data bias = Bskip ? Bskip[layer*S+row] : (T_data)0.f;

    if (row < S) {
        for (int sample=start_sample; sample < start_sample + num_samples; sample++) {
            for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
                // sampleLockacquire has a __syncthreads in it
                sampleLockAcquire<BATCH_UNROLL>(batch_offset, sample, ySample);
                namedBarrierSync(2, S+R); // h_sh produced
                GEMM<R,2,BATCH_UNROLL>(weights,h_sh,accum);
                if (layer > 0) {
                    bool valid = false;
                    T_data accum_in_reg[BATCH_UNROLL];
                    while (!valid) {
                        valid = true;
#pragma unroll
                        for (int b=0; b<BATCH_UNROLL; b++) {
                            accum_in_reg[b] = loadVolatile(accum_in,(layer - 1)*batch_size*S + (batch_offset+b)*S + row);
                        }
#pragma unroll
                        for (int b=0; b<BATCH_UNROLL; b++) {
                            valid &= !isNegativeZero(accum_in_reg[b]);
                        }
                    }
#pragma unroll
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        accum[b] += accum_in_reg[b];
                    }
                }
#pragma unroll
                for (int b=0; b<BATCH_UNROLL; b++) {
                    accum[b] += bias;
                    storeValidate(accum_in,layer*batch_size*S + (batch_offset+b)*S + row,accum[b]);
                }
            }
        }
    }
}

template <typename T_weight, typename T_data, int R, int S, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_cur_res_skip(int thread_id, int start_sample, int num_samples, volatile int* ySample, int layer, int num_layers, int batch_size, int maxDilation, T_weight* Wcur, T_weight* Wres, T_data* Bres, T_weight* Wskip, T_data* Bskip, T_data* a_prev, T_data* xt, T_data* skip_out, T_data* xtOut, int* yInPrev, int* yInCur, T_data* embedPrev, T_data* embedCur, bool tanhEmbed, bool dumpActivations) {
    __shared__ T_data h_sh[BATCH_UNROLL][R];
    __shared__ T_data a_sh[BATCH_UNROLL][2*R];
    __shared__ T_data x_sh[BATCH_UNROLL][R];
    if (thread_id < 2*R) {
        nv_wavenet_persistent_cur<T_weight, T_data, R, BATCH_UNROLL>(thread_id, start_sample, num_samples, ySample, layer, num_layers, batch_size, maxDilation, Wcur, a_prev, xt, x_sh, a_sh, yInPrev, yInCur, embedPrev, embedCur, tanhEmbed);
    }
    else if (thread_id < 3*R) {
        nv_wavenet_persistent_res<T_weight, T_data, R, S, BATCH_UNROLL>(thread_id - 2*R, start_sample, num_samples, ySample, layer, num_layers, batch_size, maxDilation, Wres, Bres, xt, x_sh, a_sh, h_sh, xtOut, dumpActivations);
    }
    else {
        nv_wavenet_persistent_skip<T_weight, T_data, S, R, BATCH_UNROLL>(thread_id - 3*R, start_sample, num_samples, ySample, layer, num_layers, batch_size, Wskip, Bskip, h_sh, skip_out);
    }
}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_softmax(int block_id, int batch_size, int num_layers, int start_sample, int num_samples, int maxDilation, volatile T_data* outAccumulate, float* outputSelectors, T_data* p, int* yOut, int* yInPrev, int* yInCur, volatile int* ySample, bool dumpActivations) {
    for (int sample = start_sample; sample < start_sample + num_samples; sample++) {
        __shared__ T_data out_sh[BATCH_UNROLL][A];
        __shared__ T_data p_sh[BATCH_UNROLL][A];
        __shared__ int yOut_sh[BATCH_UNROLL];

        int col = block_id*BATCH_UNROLL;
        sampleLockAcquire<BATCH_UNROLL>(col, sample, ySample);
        const int NUM_THREADS=std::min(2*R,A);
        if (threadIdx.x < NUM_THREADS) {

            const int ROWS_PER_THREAD = A/NUM_THREADS;
            T_data out_reg[BATCH_UNROLL][ROWS_PER_THREAD];
            bool valid = false;
            while (!valid) {
                valid = true;
#pragma unroll
                for (int u=0; u<BATCH_UNROLL; u++) {
                    for (int r=0; r<ROWS_PER_THREAD; r++) {
                        int row = threadIdx.x*ROWS_PER_THREAD + r;
                        out_reg[u][r] = loadVolatile(outAccumulate,(S/R-1)*batch_size*A + (col+u)*A + row);
                    }
                }
#pragma unroll
                for (int u=0; u<BATCH_UNROLL; u++) {
                    for (int r=0; r<ROWS_PER_THREAD; r++) {
                        valid &= !isNegativeZero(out_reg[u][r]);
                    }
                }
            }
#pragma unroll
            for (int u=0; u<BATCH_UNROLL; u++) {
                for (int r=0; r<ROWS_PER_THREAD; r++) {
                    out_sh[u][threadIdx.x*ROWS_PER_THREAD+r] = out_reg[u][r];
                }
            }
        }

        namedBarrierSync(1, 4*R); // input is read, buffers can be cleaned out in cleanup_advance block

        if (threadIdx.x < NUM_THREADS) {
            softmax_select<T_data, std::min(NUM_THREADS, A), A,BATCH_UNROLL>(0,BATCH_UNROLL, (T_data*)out_sh, dumpActivations ? (T_data*)p_sh : NULL, outputSelectors + (sample - start_sample)*batch_size + col, yOut_sh, 1, NUM_THREADS);

            namedBarrierSync(2, 2*R);
#pragma unroll
            for (int u=0; u<BATCH_UNROLL; u++) {
                if (dumpActivations) {
                    for (int i=threadIdx.x; i<A; i += 2*R){
                        p[(col+u)*A + i] = p_sh[u][i];
                    }
                }

                if (threadIdx.x == 0) {
                    yOut[(col+u)*num_samples + (sample - start_sample)] = yOut_sh[u];
                    yInPrev[col+u] = yInCur[col+u];
                    yInCur[col+u] = yOut_sh[u];
                    __threadfence();
                }
            }
        }
        namedBarrierArrive(3, 4*R); // mark ready to advance to next sample
    }
}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_cleanup_advance(int block_id, int thread_id, int batch_size, int num_layers, int start_sample, int num_samples, int maxDilation, volatile T_data* outAccumulate, volatile int* ySample, volatile int* hSample, T_data* xt, T_data* skip_out, T_data* skipOutAccumulate) {
    for (int sample = start_sample; sample < start_sample + num_samples; sample++) {
        volatile T_data* Xt = xt + ((sample+1)%(maxDilation+1))*(num_layers+1)*R*batch_size;
        int col = block_id*BATCH_UNROLL;
        sampleLockAcquire<BATCH_UNROLL>(col, sample, ySample);
        namedBarrierSync(1, 4*R); // wait till output read
        if ((sample + 1) < start_sample + num_samples) {
            for (int l=0; l<num_layers; l++) {
                for (int u=0; u<BATCH_UNROLL; u++) {
                    if (thread_id < R) {
                        storeVolatile(Xt,l*batch_size*R + (col+u)*R + thread_id,-0.f);
                    } 
                    else {
                        for (int i=0;i<S/R;i++) {
                            skip_out[l*batch_size*S + (col+u)*S + i*R + thread_id - R] = -0.f;
                        }
                    } 
                }
            }
            if (thread_id < R) {
                // zero out extra skip layer too
                for (int u=0; u<BATCH_UNROLL; u++) {
                    for (int i=0;i<S/R;i++) {
                        skip_out[num_layers*batch_size*S + (col+u)*S + i*R + thread_id] = -0.f;
                    }
                }
                for (int l=0; l<S/R; l++) {
                    for (int i=0; i<S/R; i++) {
                        for (int u=0; u<BATCH_UNROLL; u++) {
                            skipOutAccumulate[l*batch_size*S + (col+u)*S + i*R + thread_id] = -0.f;
                        }
                    }
                }
            }
            else {
                for (int l=0; l<A/R; l++) {
                    for (int i=0; i<A/R; i++) {
                        for (int u=0; u<BATCH_UNROLL; u++) {
                            storeVolatile(outAccumulate,l*batch_size*A + (col+u)*A + i*R + thread_id - R,-0.f);        
                        }
                    }
                }
            }
        }

        // wait until selection was made in softmax block and clears are visible before advancing sample lock
        __threadfence();
        namedBarrierSync(3, 4*R);

        if (thread_id == 0) {
#pragma unroll
            for (int u=0; u<BATCH_UNROLL; u++) {
                while (hSample[col + u] <= sample);
            }
#pragma unroll
            for (int u=0; u<BATCH_UNROLL; u++) {
                ySample[col+u] = sample+1;
            }
        }
    }
}

template <typename T_weight, typename T_data, int R, int S, int N_UNROLL>
__device__ void skip_condition(int thread_id, int start_sample, int num_samples, volatile int* ySample, int num_layers, int batch_size, int maxDilation, T_weight* W, T_data* B, volatile T_data* act_in, T_data skip_sh[N_UNROLL][S],T_data* act_out, volatile T_data* accum_in) {
    int row = thread_id;
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[R/WV];
    loadWeights<S,R>(weights,W,0,row);
    T_data accum[N_UNROLL];
    T_data bias = B ? B[row] : (T_data)0.f;

    T_data act_in_reg[N_UNROLL];

    if (thread_id < S) {
        for (int sample=start_sample; sample < start_sample + num_samples; sample++) {
            int sample_offset = (sample % (maxDilation+1)) * (num_layers+1)*R*batch_size;
            for (int batch_offset = 0; batch_offset < batch_size; batch_offset += N_UNROLL) {
                // sampleLockacquire has a __syncthreads in it, so we don't need to worry about act_in_sh race
                sampleLockAcquire<N_UNROLL>(batch_offset, sample, ySample);
                if (row < R) {
                    bool valid = false;
                    while (!valid) {
                        valid = true;
#pragma unroll
                        for (int b=0; b<N_UNROLL; b++) {
                            act_in_reg[b] = loadVolatile(act_in, sample_offset + (batch_offset+b)*R + row);
                        }
#pragma unroll
                        for (int b=0; b<N_UNROLL; b++) {
                            valid &= !isNegativeZero(act_in_reg[b]);
                        }
                    }
#pragma unroll
                    for (int b=0; b<N_UNROLL; b++) {
                        skip_sh[b][row] = act_in_reg[b];
                    }
                }
                namedBarrierSync(1, S);
                GEMM<R,2,N_UNROLL>(weights,skip_sh,accum);

                bool valid = false;
                T_data accum_in_reg[N_UNROLL];
                while (!valid) {
                    valid = true;
#pragma unroll
                    for (int b=0; b<N_UNROLL; b++) {
                        accum_in_reg[b] = loadVolatile(accum_in,(batch_offset+b)*S + row);
                    }
#pragma unroll
                    for (int b=0; b<N_UNROLL; b++) {
                        valid &= !isNegativeZero(accum_in_reg[b]);
                    }
                }
#pragma unroll
                for (int b=0; b<N_UNROLL; b++) {
                    accum[b] += accum_in_reg[b];
                    accum[b] += bias;
                    accum[b] = relu(accum[b]);
                    skip_sh[b][row] = accum[b];
                    if (act_out) act_out[(batch_offset+b)*S + row] = accum[b];
                }
                namedBarrierArrive(2, 2*S);
            }
        }
    }
}

template <typename T_weight, typename T_data, int S, int A, int N_UNROLL, int COND_REPEAT>
__device__ void calculate_zs(int thread_id, int start_sample, int num_samples, volatile int* ySample, int batch_size, T_weight* W, T_data* B, T_data skip_sh[N_UNROLL][S], T_data* finalConditioning, T_data* act_out) {
    int row = thread_id;
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[S/WV];
    loadWeights<S,S>(weights,W,0,row);
    T_data accum[N_UNROLL];
    T_data bias = B ? B[row] : (T_data)0.f;

    T_data cond_reg[N_UNROLL];

    if (thread_id < S) {
        for (int sample=start_sample; sample < start_sample + num_samples; sample++) {
            for (int batch_offset = 0; batch_offset < batch_size; batch_offset += N_UNROLL) {
                // sampleLockacquire has a __syncthreads in it, so we don't need to worry about act_in_sh race
                sampleLockAcquire<N_UNROLL>(batch_offset, sample, ySample);
#pragma unroll
                for (int b=0; b<N_UNROLL; b++) {
                    cond_reg[b] = finalConditioning[((sample-start_sample)/COND_REPEAT)*batch_size*S + (batch_offset+b)*S + row];
                    cond_reg[b] += bias;
                }
                namedBarrierSync(2, 2*S); // wait till skip_condition will fill skip_sh with our phase of the calculation
                GEMM<S,2,N_UNROLL>(weights,skip_sh,accum);

                for (int b=0; b<N_UNROLL; b++) {
                    accum[b] += cond_reg[b];
                    accum[b] = relu(accum[b]);
                    skip_sh[b][row] = accum[b];
                    if (act_out) act_out[(batch_offset+b)*S + row] = accum[b];
                }
                namedBarrierArrive(3, S+A);
            }
            //if (thread_id == 0){printf("zs done\n");}
        }
    }
}

template <typename T_weight, typename T_data, int S, int A, int N_UNROLL>
__device__ void calculate_za(int thread_id, int start_sample, int num_samples, volatile int* ySample, int batch_size, T_weight* W, T_data* B, T_data skip_sh[N_UNROLL][S], T_data* act_out, volatile T_data* outAccumulate) {
    int row = thread_id;
    const int WV = sizeof(T_weight)/sizeof(T_data);
    T_weight weights[S/WV];
    loadWeights<A,S>(weights,W,0,row,A);
    T_data accum[N_UNROLL];
    T_data bias = B ? B[row] : (T_data)0.f;

    if (thread_id < A) {
        for (int sample=start_sample; sample < start_sample + num_samples; sample++) {
            for (int batch_offset = 0; batch_offset < batch_size; batch_offset += N_UNROLL) {
                // sampleLockacquire has a __syncthreads in it, so we don't need to worry about act_in_sh race
                sampleLockAcquire<N_UNROLL>(batch_offset, sample, ySample);
                namedBarrierSync(3, S+A); // wait till skip_condition will create 
                GEMM<S,4,N_UNROLL>(weights,skip_sh,accum);

#pragma unroll
                for (int b=0; b<N_UNROLL; b++) {
                    accum[b] += bias;
                    if (act_out) act_out[(batch_offset+b)*A + row] = accum[b];
                    storeValidate(outAccumulate, (batch_offset+b)*A + row, accum[b]);
                }
            }
            //if (thread_id == 0){printf("sample done\n");}
        }
    }
}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL, int COND_REPEAT>
__device__ void nv_wavenet_persistent_final_convs(int start_sample, int num_samples, volatile int* ySample, int num_layers, int batch_size, int maxDilation, T_weight* WskipInit, T_data* BskipInit, T_weight* WskipOut, T_data* BskipOut, T_weight* Wout, T_data* Bout, volatile T_data* xt, T_data* skip_out, volatile T_data* skipOutAccumulate, T_data* finalConditioning, T_data* skipOutFinal, T_data* out, volatile T_data* outAccumulate, bool dumpActivations) {
    __shared__ T_data skip_sh[BATCH_UNROLL][S];
    if (threadIdx.x < S) {
        skip_condition<T_weight, T_data, R, S, BATCH_UNROLL>(threadIdx.x, start_sample, num_samples, ySample, num_layers, batch_size, maxDilation, WskipInit, BskipInit, xt, skip_sh, dumpActivations? (T_data*)skipOutAccumulate : nullptr, skip_out + (num_layers - 1)*S*batch_size);
    }
    else if (threadIdx.x < 2*S) {
        calculate_zs<T_weight, T_data, S, A, BATCH_UNROLL, COND_REPEAT>(threadIdx.x - S, start_sample, num_samples, ySample, batch_size, WskipOut, BskipOut, skip_sh, finalConditioning, dumpActivations ? skipOutFinal : nullptr);
    }
    else if (threadIdx.x < 2*S + A) {
        calculate_za<T_weight, T_data, S, A, BATCH_UNROLL>(threadIdx.x - 2*S, start_sample, num_samples, ySample, batch_size, Wout, Bout, skip_sh, dumpActivations ? out : nullptr, outAccumulate);
    }
}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL, int COND_REPEAT>
__global__ void nv_wavenet_persistent(nv_wavenet_params<T_weight, T_data> params) {
    //if (threadIdx.x == 0) {printf("block start %d\n", blockIdx.x);}
    int prev_blocks = NUM_PREV_BLOCKS / 2;
    int res_blocks = params.num_layers;
    int skip_precondition_blocks = 1; //precondition skip connections
    int softmax_blocks = params.batch_size;
    int thread_id = threadIdx.x;
    if (blockIdx.x < prev_blocks) {
        // Prev
        // two prev layers per block
        if (thread_id < 2*R) {
            nv_wavenet_persistent_prev_multilayer<T_weight, T_data, R, BATCH_UNROLL, 1, COND_REPEAT>(blockIdx.x * 2, thread_id, params.start_sample, params.num_samples, params.ySample, params.hSample, params.num_layers, params.batch_size, params.maxDilation, params.Wprev, params.B, params.L, params.a_prev, params.xt);
        }
        else {
            nv_wavenet_persistent_prev_multilayer<T_weight, T_data, R, BATCH_UNROLL, 2, COND_REPEAT>((blockIdx.x * 2) + 1, thread_id - 2*R, params.start_sample, params.num_samples, params.ySample, params.hSample, params.num_layers, params.batch_size, params.maxDilation, params.Wprev, params.B, params.L, params.a_prev, params.xt);
        }
    }
    else if (blockIdx.x < prev_blocks + res_blocks) {
        // Cur + Res + skip
        int layer = blockIdx.x - prev_blocks;
        nv_wavenet_persistent_cur_res_skip<T_weight, T_data, R, S, BATCH_UNROLL>(thread_id, params.start_sample, params.num_samples, params.ySample, layer, params.num_layers, params.batch_size, params.maxDilation, params.Wcur, params.Wres, params.Bres, params.Wskip, params.Bskip, params.a_prev, params.xt, params.skip_out, params.xtOut, params.yInPrev, params.yInCur, params.embedPrev, params.embedCur, params.tanhEmbed, params.dumpActivations);
    }
    else if (blockIdx.x < prev_blocks + res_blocks + skip_precondition_blocks) {
        // Final blocks - skip final conditioning, Zs and Za calculations
        nv_wavenet_persistent_final_convs<T_weight, T_data, R, S, A, BATCH_UNROLL, COND_REPEAT>(params.start_sample, params.num_samples, params.ySample, params.num_layers, params.batch_size, params.maxDilation, params.WskipInit, params.BskipInit, params.WskipOut, params.BskipOut, params.Wout, params.Bout, params.xt, params.skip_out, params.skipOutAccumulate, params.LFinal, params.skipOutFinal, params.out, params.outAccumulate, params.dumpActivations);
    }
    else if (blockIdx.x < prev_blocks + res_blocks + skip_precondition_blocks + softmax_blocks) {
        // softmax + cleanup block
        int block_id = blockIdx.x - prev_blocks - res_blocks - skip_precondition_blocks;
        if (thread_id < 2*R) {
            nv_wavenet_persistent_softmax<T_weight, T_data, R, S, A, 1>(block_id, params.batch_size, params.num_layers, params.start_sample, params.num_samples, params.maxDilation, params.outAccumulate, params.outputSelectors, params.p, params.yOut, params.yInPrev, params.yInCur, params.ySample, params.dumpActivations);
        } 
        else {
            nv_wavenet_persistent_cleanup_advance<T_weight, T_data, R, S, A, 1>(block_id, thread_id - 2*R, params.batch_size, params.num_layers, params.start_sample, params.num_samples, params.maxDilation, params.outAccumulate, params.ySample, params.hSample, params.xt, params.skip_out, params.skipOutAccumulate);
        }
    }
    else {
        if (threadIdx.x == 0) {
            printf("unused block: %d\n", blockIdx.x);
        }
    }
   //if (threadIdx.x == 0) printf("block done: %d\n", blockIdx.x);
}

template <typename T_weight, typename T_data, int R, int S, int A, int BATCH_UNROLL, int COND_REPEAT>
bool launch_persistent(nv_wavenet_params<T_weight, T_data> params, cudaStream_t stream) {
    int prev_blocks = NUM_PREV_BLOCKS / 2;
    assert(NUM_PREV_BLOCKS % 2 == 0); // we always create pairs of prev blocks
    assert(params.num_layers >= NUM_PREV_BLOCKS);
    int res_blocks = params.num_layers;
    //if (S<2*R) assert (S%R==0); else assert(S%2*R==0);
    //assert(A>=4*R);
    assert(A<=4*R);
    assert(S<=R);
    int skip_precondition_blocks = 1;
    //printf("Zs: %d, Za: %d\n", Zs_blocks, Za_blocks);
    int softmax_blocks = params.batch_size;
    dim3 grid(prev_blocks + res_blocks + skip_precondition_blocks + softmax_blocks);
    dim3 block(4*R);
    int occ = getOccupancy(0, block.x*block.y*block.z,(void*)nv_wavenet_persistent<T_weight, T_data, R, S, A, BATCH_UNROLL, COND_REPEAT>);
    //printf("%d blocks, %d blocks per SM, %d threads per block\n", grid.x, occ, block.x*block.y*block.z);
    assert(occ>0);
    int sample_offset = (params.start_sample % (params.maxDilation + 1))*(params.num_layers + 1)*R*params.batch_size;
    //printf("start sample/offset: %d/%d\n", params.start_sample, sample_offset);
    initializeActivations<T_data,R><<<params.num_layers*params.batch_size,R,0,stream>>>(params.xt + sample_offset, params.h);
    if (params.start_sample == 0) {
        dim3 initGrid(params.num_layers, params.batch_size);
        // initialize a_prev to correct values because we skip calculating a_prev for the first sample due to pipelining
        initializeAPrev<T_data><<<initGrid,2*R,0,stream>>>(params.a_prev, params.B, params.L);
    }
    initializeActivationsGeneric<T_data><<<params.num_layers*params.batch_size,2*R,0,stream>>>(params.a);
    initializeActivationsGeneric<T_data><<<(params.num_layers+1)*params.batch_size,S,0,stream>>>(params.skip_out);
    initializeActivationsGeneric<T_data><<<(S/R)*params.batch_size,S,0,stream>>>(params.skipOutAccumulate);
    initializeActivationsGeneric<T_data><<<(S/R)*params.batch_size,A,0,stream>>>(params.outAccumulate);
    //printf("done initializing\n");
    void* p_params = {&params};
    cudaError_t code = cudaLaunchCooperativeKernel((void*)nv_wavenet_persistent<T_weight,T_data,R,S,A,BATCH_UNROLL, COND_REPEAT>, grid, block, &p_params, 0, stream);
    //printf("launched kernel ok\n");
    gpuAssert(code, __FILE__, __LINE__, false);
    return code == cudaSuccess;
}
