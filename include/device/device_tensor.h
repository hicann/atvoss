/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "acl/acl.h"
namespace ATVOS::Device {

template <typename T>
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<T>& src)
    {
        SetSize(src.size());
        src_ = &src;
    }

    T& operator[](std::size_t pos)
    {
        return ptr_[pos];
    }
    const T& operator[](std::size_t pos) const
    {
        return ptr_[pos];
    }

    void Clear()
    {
        delete[] ptr_;
        ptr_ = nullptr;
        len_ = 0;
    }

    void SetSize(std::size_t size)
    {
        if (ptr_ != nullptr) {
            throw std::logic_error(
                "[ERROR]: [ATVOS][Device] MyTensor::SetSize can only be called on an empty object");
        }
        aclrtMalloc((void **)&ptr_, size * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
        len_ = size;
    }
    std::size_t GetSize() const
    {
        return len_;
    }
    uint8_t* GetPtr() const
    {
        return (uint8_t* )ptr_;
    }

    void CopyIn(const std::vector<T>& src)
    {
        if (src.size() > len_) {
            throw std::logic_error("[ERROR]: [ATVOS][Device] Source vector is too big");
        }
        aclrtMemcpy(ptr_, len_ * sizeof(T), src_->data(), len_ * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    }

    void CopyOut(std::vector<T>& dst)
    {
        if (dst.size() < len_) {
            throw std::logic_error("[ERROR]: [ATVOS][Device] Destination vector is too small");
        }
        aclrtMemcpy(src_->data(), len_ * sizeof(T), ptr_, len_ * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    }

private:
    T* ptr_{};
    std::size_t len_{};
    std::vector<T>* src_;
};
}