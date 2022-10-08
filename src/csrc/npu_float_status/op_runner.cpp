/*
 * Copyright (c) 2020, Huawei Technologies.All rights reserved.
 * 
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "op_runner.h"
#include <limits>
#include "common.h"
#include <torch/extension.h>
#include <c10/npu/NPUEvent.h>

using namespace std;

OpRunner::OpRunner(OperatorDesc *opDesc) : opDesc_(opDesc)
{
    numInputs_ = opDesc->inputDesc.size();
    numOutputs_ = opDesc->outputDesc.size();
}

OpRunner::~OpRunner()
{
    for (auto *inputBuf : inputBuffers_) {
        aclDestroyDataBuffer(inputBuf);
    }

    for (auto *devInput : devInputs_) {
        aclrtFree(devInput);
    }

    for (auto *hostInput : hostInputs_) {
        aclrtFreeHost(hostInput);
    }

    for (auto *outputBuf : outputBuffers_) {
        aclDestroyDataBuffer(outputBuf);
    }

    for (auto *devOutput : devOutputs_) {
        aclrtFree(devOutput);
    }

    for (auto *hostOutput : hostOutputs_) {
        aclrtFreeHost(hostOutput);
    }
}

int OpRunner::Init()
{
    for (size_t i = 0; i < numInputs_; ++i) {
        auto size = GetInputSize(i);
        void *devMem = nullptr;
        if (aclrtMalloc(&devMem, size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) {
            ERROR_LOG("Malloc device memory for input[%zu] failed", i);
            return RUN_FAILED;
        }

        if (aclrtMemset(devMem, size, 0, size) != ACL_ERROR_NONE) {
            ERROR_LOG("Set device memory for input[%zu] failed", i);
            return RUN_FAILED;
        }

        devInputs_.emplace_back(devMem);
        inputBuffers_.emplace_back(aclCreateDataBuffer(devMem, size));

        void *hostMem = nullptr;
        if (aclrtMallocHost(&hostMem, size) != ACL_ERROR_NONE) {
            ERROR_LOG("Malloc device memory for input[%zu] failed", i);
            return RUN_FAILED;
        }
        if (hostMem == nullptr) {
            ERROR_LOG("Malloc memory for input[%zu] failed", i);
            return RUN_FAILED;
        }
        hostInputs_.emplace_back(hostMem);
    }

    for (size_t i = 0; i < numOutputs_; ++i) {
        auto size = GetOutputSize(i);
        void *devMem = nullptr;
        if (aclrtMalloc(&devMem, size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) {
            ERROR_LOG("Malloc device memory for output[%zu] failed", i);
            return RUN_FAILED;
        }

        if (aclrtMemset(devMem, size, 0, size) != ACL_ERROR_NONE) {
            ERROR_LOG("Set device memory for output[%zu] failed", i);
            return RUN_FAILED;
        }

        devOutputs_.emplace_back(devMem);
        outputBuffers_.emplace_back(aclCreateDataBuffer(devMem, size));

        void *hostOutput = nullptr;
        if (aclrtMallocHost(&hostOutput, size) != ACL_ERROR_NONE) {
            ERROR_LOG("Malloc device memory for output[%zu] failed", i);
            return RUN_FAILED;
        }
        if (hostOutput == nullptr) {
            ERROR_LOG("Malloc host memory for output[%zu] failed", i);
            return RUN_FAILED;
        }
        hostOutputs_.emplace_back(hostOutput);
    }

    return RUN_SUCCESS;
}

size_t OpRunner::NumInputs()
{
    return numInputs_;
}

size_t OpRunner::NumOutputs()
{
    return numOutputs_;
}

size_t OpRunner::GetInputSize(size_t index)
{
    if (index >= opDesc_->inputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescSize(opDesc_->inputDesc[index]);
}

size_t OpRunner::GetOutputSize(size_t index)
{
    if (index >= opDesc_->outputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescSize(opDesc_->outputDesc[index]);
}

int OpRunner::RunOp()
{
    auto stream = at::npu::getCurrentNPUStream();
    int holdGIL = PyGILState_Check();
    aclError ret = ACL_ERROR_NONE;

    if (holdGIL) {
        Py_BEGIN_ALLOW_THREADS
        ret = aclopCompileAndExecute(opDesc_->opType.c_str(),
                                     numInputs_,
                                     opDesc_->inputDesc.data(),
                                     inputBuffers_.data(),
                                     numOutputs_,
                                     opDesc_->outputDesc.data(),
                                     outputBuffers_.data(),
                                     opDesc_->opAttr,
                                     ACL_ENGINE_SYS,
                                     ACL_COMPILE_SYS,
                                     nullptr,
                                     stream);
        Py_END_ALLOW_THREADS
    } else {
        ret = aclopCompileAndExecute(opDesc_->opType.c_str(),
                                     numInputs_,
                                     opDesc_->inputDesc.data(),
                                     inputBuffers_.data(),
                                     numOutputs_,
                                     opDesc_->outputDesc.data(),
                                     outputBuffers_.data(),
                                     opDesc_->opAttr,
                                     ACL_ENGINE_SYS,
                                     ACL_COMPILE_SYS,
                                     nullptr,
                                     stream);
    }

    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("Execute %s failed. ret = %d", opDesc_->opType.c_str(), ret);
        return RUN_FAILED;
    }

    if (opDesc_->opType == OP_TYPE_NPU_GET_FLOAT_STATUS) {
        if (aclrtSynchronizeStream(stream) != ACL_ERROR_NONE) {
            ERROR_LOG("Synchronize stream failed");
            return RUN_FAILED;
        }

        for (size_t i = 0; i < numInputs_; ++i) {
            auto size = GetInputSize(i);
            if (aclrtMemcpy(hostInputs_[i], size, devInputs_[i], size, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_ERROR_NONE) {
                ERROR_LOG("Copy input[%zu] failed", i);
                return RUN_FAILED;
            }
        }
    }

    return RUN_SUCCESS;
}
