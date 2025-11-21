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

template<typename T>
struct RmsNormOp : ATVOS::ExprTmpl::Maker {
    using shape = AscendC::Shape<Int<HEIGHT>, Int<WIDTH>>;
    using layout = AscendC::Std::tuple<shape>;
    using maxSizeType = T;
    template <template <typename> class VectorType>
    __host_aicore__ constexpr auto Get() const
    {
        using namespace ATVOS::ExprTmpl;
        auto _1 = DefineParam<1, VectorType<T>, layout>();
        auto _2 = DefineParam<2, VectorType<T>, layout>();
        auto _3 = DefineLocalVarLike<1>(_1);
        auto _4 = DefineParam<3, VectorType<T>, layout, ParamUsage::out>();

        return (_3 = _1 *_1,
                _4 = ReduceSum<ATVOS::Patterns::Pattern::AR>(_3),
                _4 = Broadcast<ATVOS::Patterns::Pattern::AB>(_4),
                _3 = Divs<WIDTH>(_4),
                _4 = Sqrt(_3),
                _3 = _1 / _4,
                _4 = _2 * _3);
    }
};

static constexpr ATVOS::Kernel::PolicyEleWise kernelPolicyWidthAssign{48, 1, WIDTH, 1, ATVOS::Kernel::PolicySegment::UniformSegment};
static constexpr ATVOS::Block::PolicyEleWise blockPolicyWidthAssign{190 * 1024, WIDTH};

using BlockOp = ATVOS::Block::BlockEleWise<RmsNormOp<float>, blockPolicyWidthAssign>;
using KernelOp = ATVOS::Kernel::KernelEleWise<BlockOp, kernelPolicyWidthAssign>;
using DeviceOp = ATVOS::Device::DeviceAdapter<KernelOp>;

int main() {
    std::vector<float> v1(32*32, 1.0F);
    std::vector<float> v2(32*32, 2.0F);
    std::vector<float> v3(32*32);
    std::vector<float> golden(32*32, 2.0f);
    std::vector<uint32_t> shape{32,32};
    DeviceOp deviceOp(shape);
    deviceOp.Run(v1, v2, v3);
    if (!VerifyResults(golden, v3)) {
        return -1;
    }

    printf("Accuracy verification passed.\n");
    return 0;
}