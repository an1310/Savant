//
// Created by bogoslovskiy_nn on 13.10.2021.
//

#ifndef LIBS_CUDACROP_H
#define LIBS_CUDACROP_H
#include "npp.h"

// Pitches are byte strides; the kernel converts them to RGBA pixel strides.
cudaError_t cudaCrop(
        const Npp8u *ref_frame,
        NppiSize inputSize,
        NppiRect crop_rect,
        Npp8u* output,
        NppiSize outputSize,
        unsigned int inputPitch,
        unsigned int outputPitch,
        int shiftX,
        int shiftY);

inline __device__ __host__ int iDivUp( unsigned int a, unsigned int b ) { return (a % b != 0) ? (a / b + 1) : (a / b); }

#endif //LIBS_CUDACROP_H
