/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "kernel_operator.h"
#include "kernel_event.h"
#include "include/utils/expression/common.h"
#include "kernel/kernel_elewise.h"
#include "tile/tile_elewise.h"
#include "tile/tile_evaluator_common.h"
#include "tile/tile_evaluator.h"
#include "block/block_elewise.h"

namespace ATVOSS {
class AtvosParamTest : public testing::Test {
protected:
    void SetUp() override {
        std::cout << "[SetUp] Preparing test environment." << std::endl;
    }

    void TearDown() override {
        std::cout << "[TearDown] Cleaning up test environment." << std::endl;
    }
};
static constexpr int32_t HEIGHT = 1;
static constexpr int32_t WIDTH = 32;
template<typename T>
struct RmsNormOp : ATVOSS::ExprTmpl::Maker {
    using shape = AscendC::Shape<Int<HEIGHT>, Int<WIDTH>>;
    using layout = AscendC::Std::tuple<shape>;
    using maxSizeType = T;
    template <template <typename> class VectorType>
    __host_aicore__ constexpr auto Get() const
    {
        using namespace ATVOSS::ExprTmpl;
        auto _1 = PlaceHolder<1, VectorType<T>, layout>();
        auto _2 = PlaceHolder<2, VectorType<T>, layout, ParamUsage::out>();
        return (_2 = _1 * _1);
    }
};
// TEST_F(AtvosParamTest, AtvosParamTestSuccessCase) {
//     std::cout << "[TEST] Running AtvosParamTestCase - Just printing info." << std::endl;
//     static constexpr ATVOSS::Kernel::PolicyEleWise kernelPolicyWidthAssign{48, 1, 0, 1, ATVOSS::Kernel::PolicySegment::UniformSegment};
//     static constexpr ATVOSS::Block::PolicyEleWise blockPolicyWidthAssign{190 * 1024, WIDTH};
//     using BlockOp = ATVOSS::Block::BlockBuilder<RmsNormOp<float>, blockPolicyWidthAssign>;
//     using KernelOp = ATVOSS::Kernel::KernelBuilder<BlockOp, kernelPolicyWidthAssign>;
//     using KernelParamStruct = typename KernelOp::ParamStruct;
//     using BlockParamStruct = typename KernelOp::BlockTemplate::ParamStruct;
//     KernelParamStruct kernelParam;
//     BlockParamStruct blockParam;
//     std::vector<uint32_t> shape1{32*32};
//     auto ret = KernelOp::MakeKernelParam(shape1, kernelParam);
//     EXPECT_EQ(true, ret);  // 占位断言，保证测试通过
//     std::vector<uint32_t> shape4{1, 1024};
//     ret = KernelOp::MakeKernelParam(shape4, kernelParam);
//     EXPECT_EQ(true, ret);
//     ret = BlockOp::MakeBlockParam(blockParam);
//     EXPECT_EQ(true, ret);
// }


// TEST_F(AtvosParamTest, AtvosParamTestFailedCase) {
//     std::cout << "[TEST] Running AtvosParamTestCase - Just printing info." << std::endl;
//     static constexpr ATVOSS::Kernel::PolicyEleWise kernelPolicyWidthAssign{48, 1, 0, 0, ATVOSS::Kernel::PolicySegment::UniformSegment};
//     static constexpr ATVOSS::Block::PolicyEleWise blockPolicyWidthAssign{31, WIDTH};
//     using BlockOp = ATVOSS::Block::BlockBuilder<RmsNormOp<float>, blockPolicyWidthAssign>;
//     using KernelOp = ATVOSS::Kernel::KernelBuilder<BlockOp, kernelPolicyWidthAssign>;
//     using KernelParamStruct = typename KernelOp::ParamStruct;
//     using BlockParamStruct = typename KernelOp::BlockTemplate::ParamStruct;
//     KernelParamStruct kernelParam;
//     BlockParamStruct blockParam;
//     std::vector<uint32_t> shape1{};
//     auto ret = KernelOp::MakeKernelParam(shape1, kernelParam);
//     EXPECT_EQ(false, ret);
//     std::vector<uint32_t> shape2{0};
//     ret = KernelOp::MakeKernelParam(shape2, kernelParam);
//     EXPECT_EQ(false, ret);
//     ret = BlockOp::MakeBlockParam(blockParam);
//     EXPECT_EQ(false, ret);
// }

}  // namespace ATVOSS
