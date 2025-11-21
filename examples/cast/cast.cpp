/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "kernel_operator.h"
#include "atvos.h"
#include "example_common.h"

static constexpr int32_t HEIGHT = 1;
static constexpr int32_t WIDTH = 32;


template<typename T, typename U>
struct CastOp : ATVOS::ExprTmpl::Maker {
    using shape = AscendC::Shape<Int<HEIGHT>, Int<WIDTH>>;
    using layout = AscendC::Std::tuple<shape>;
    using maxSizeType = std::conditional_t<sizeof(T) >= sizeof(U),T,U>;
    template <template <typename> class VectorType>
    __host_aicore__ constexpr auto Get() const
    {
        using namespace ATVOS::ExprTmpl;
        auto _1 = DefineParam<1, VectorType<T>, layout>();
        auto _2 = DefineParam<2, VectorType<U>, layout, ParamUsage::out>();
        return (_2 = Cast<ATVOS::CastMode::CAST_NONE>(_1));
    }
};

static constexpr ATVOS::Kernel::PolicyEleWise kernelPolicyWidthAssign{48, 1, 0, 1, ATVOS::Kernel::PolicySegment::UniformSegment};
static constexpr ATVOS::Block::PolicyEleWise blockPolicyWidthAssign{190 * 1024, WIDTH};

using BlockOp = ATVOS::Block::BlockEleWise<CastOp<half,float>, blockPolicyWidthAssign>;
using KernelOp = ATVOS::Kernel::KernelEleWise<BlockOp, kernelPolicyWidthAssign>;
using DeviceOp = ATVOS::Device::DeviceAdapter<KernelOp>;

int main() {
    std::vector<half> v1(32*32, 1.5F);
    std::vector<float> v2(32*32);

    std::vector<float> golden(32*32, 1.5F);
    std::vector<uint32_t> shape{32*32};
    DeviceOp deviceOp(shape);
    deviceOp.Run(v1, v2);
    if (!VerifyResults(golden, v2)) {
        return -1;
    }

    printf("Accuracy verification passed.\n");
    return 0;
}