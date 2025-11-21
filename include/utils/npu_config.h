/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVOS_DEV_NPU_CONFIG_H
#define ATVOS_DEV_NPU_CONFIG_H

namespace ATVOS {
    enum class NPUModel {
        Ascend_910B=2201,
        Ascend_310B=3002,
        Ascend_310P=2002
    };

    template<size_t NPU_ARCH>
    struct NPUConfig {
        static constexpr uint32_t vectorCoreNum = 0;
        static constexpr uint64_t ubSize = 0;
    };
    // 特化910B
    template<>
    struct NPUConfig<2201> {
        static constexpr uint32_t vectorCoreNum = 48;
        static constexpr uint64_t ubSize = 194560;
    };
    // 特化310B
    template<>
    struct NPUConfig<3002> {
        static constexpr uint32_t vectorCoreNum = 1;
        static constexpr uint64_t ubSize = 262144;
    };
    // 特化310P
    template<>
    struct NPUConfig<2002> {
        static constexpr uint32_t vectorCoreNum = 8;
        static constexpr uint64_t ubSize = 262144;
    };
}

#endif //ATVOS_DEV_NPU_CONFIG_H
