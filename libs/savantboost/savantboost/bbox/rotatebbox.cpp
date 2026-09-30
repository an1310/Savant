/**
 * @file rotatebbox.cpp
 * @brief A class for working with rotated boxes
 * @version 0.1
 * @date 2022-02-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include "rotatebbox.h"
#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <npp.h>
#include <stdexcept>


static NppStreamContext currentNppStreamContext() {
    NppStreamContext context{};
    cudaDeviceProp properties{};
    if (cudaGetDevice(&context.nCudaDeviceId) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, context.nCudaDeviceId) != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device properties for NPP");
    }
    context.hStream = nullptr;
    context.nMultiProcessorCount = properties.multiProcessorCount;
    context.nMaxThreadsPerMultiProcessor = properties.maxThreadsPerMultiProcessor;
    context.nMaxThreadsPerBlock = properties.maxThreadsPerBlock;
    context.nSharedMemPerBlock = properties.sharedMemPerBlock;
    context.nCudaDevAttrComputeCapabilityMajor = properties.major;
    context.nCudaDevAttrComputeCapabilityMinor = properties.minor;
    context.nStreamFlags = cudaStreamDefault;
    return context;
}


RotateBBox::RotateBBox(float x_center, float y_center, float width, float height, float angle) {
    _x_center = x_center;
    _y_center = y_center;
    _width = width;
    _height = height;
    _angle = angle;
    _confidence = 0.f;
}

RotateBBox::RotateBBox(float x_center, float y_center, float width, float height, float angle, float confidence) {
    _x_center = x_center;
    _y_center = y_center;
    _width = width;
    _height = height;
    _angle = angle;
    _confidence = confidence;
}

savantboost::Image* RotateBBox::CutFromFrame(Npp8u* frame, NppiSize frame_size, int frame_step, float padding_width, float padding_height ) {
    double rotated_ex_rect_bbox[2][2];
    float max_side =std::fmax(_height, _width);


    NppiRect ex_rect_bbox = {
            .x = (int) std::ceil(_x_center - max_side/2 - padding_width*4),
            .y = (int) std::ceil(_y_center - max_side/2 - padding_height*4),
            .width = (int) std::ceil(max_side + 8 * padding_width),
            .height = (int) std::ceil(max_side + 8 * padding_height)};
    if (ex_rect_bbox.y < 0) {
        ex_rect_bbox.height += ex_rect_bbox.y;
        ex_rect_bbox.y = 0;
    }
    if (ex_rect_bbox.x < 0) {
        ex_rect_bbox.width += ex_rect_bbox.x;
        ex_rect_bbox.x = 0;
    }

    nppiGetRotateBound(ex_rect_bbox, rotated_ex_rect_bbox, (double) _angle, 0, 0 );
    int image_width = (int) _width + (int) std::ceil(padding_width * 2);
    int image_height = (int) _height + (int) std::ceil(padding_height * 2);
    auto object_image = std::make_unique<savantboost::Image>(image_width, image_height);

    NppiRect ex_rect_pencil_bbox = {
            .x = (int) 0,
            .y = (int) 0,
            .width = image_width,
            .height = image_height};

    float shift_x = -rotated_ex_rect_bbox[0][0] -
                    (rotated_ex_rect_bbox[1][0] - rotated_ex_rect_bbox[0][0]) / 2 + _width / 2 + padding_width;
    float shift_y = -rotated_ex_rect_bbox[0][1] -
                    (rotated_ex_rect_bbox[1][1] - rotated_ex_rect_bbox[0][1]) / 2 + _height / 2 + padding_height;

    const NppStatus npp_status = nppiRotate_8u_C4R_Ctx(
            frame,
            frame_size,
            frame_step,
            ex_rect_bbox,
            (Npp8u*) object_image->getDataPtr(),
            ex_rect_pencil_bbox.width * 4,
            ex_rect_pencil_bbox,
            _angle,
            shift_x,
            shift_y,
            NPPI_INTER_LINEAR,
            currentNppStreamContext()
    );
    if (npp_status != NPP_SUCCESS) {
        throw std::runtime_error("NPP rotation failed: " + std::to_string(npp_status));
    }
    const cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(cuda_status));
    }
    return object_image.release();
}
