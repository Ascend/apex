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

#ifndef OP_RUNNER_H
#define OP_RUNNER_H

#include "common.h"
#include "operator_desc.h"

/**
 * Op Runner
 */
class OpRunner {
public:
    /**
     * @brief Constructor
     * @param [in] opDesc: op description
     */
    explicit OpRunner(OperatorDesc *opDesc);

    /**
     * @brief Destructor
     */
    ~OpRunner();

    /**
    * @brief Init op runner
    */
    int Init();

    /**
     * @brief Get number of inputs
     * @return number of inputs
     */
    size_t NumInputs();

    /**
     * @brief Get number of outputs
     * @return number of outputs
     */
    size_t NumOutputs();

    /**
     * @brief Get input size by index
     * @param [in] index: input index
     * @return size of the input
     */
    size_t GetInputSize(size_t index);

    /**
     * @brief Get output size by index
     * @param [in] index: output index
     * @return size of the output
     */
    size_t GetOutputSize(size_t index);

    /**
     * @brief Get input buffer(host memory) by index
     * @tparam T: data type
     * @param [in] index: input index
     * @return host address of the input
     */
    template<typename T>
    T *GetInputBuffer(size_t index)
    {
        if (index >= numInputs_) {
            ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
            return nullptr;
        }
        return reinterpret_cast<T *>(hostInputs_[index]);
    }

    /**
     * @brief Get output buffer(host memory) by index
     * @tparam T: data type
     * @param [in] index: output index
     * @return host address of the output
     */
    template<typename T>
    const T *GetOutputBuffer(size_t index)
    {
        if (index >= numOutputs_) {
            ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
            return nullptr;
        }

        return reinterpret_cast<T *>(hostOutputs_[index]);
    }

    /**
     * @brief Run op
     * @return run result
     */
    int RunOp();

private:
    size_t numInputs_;
    size_t numOutputs_;

    std::vector<const aclDataBuffer *> inputBuffers_;
    std::vector<aclDataBuffer *> outputBuffers_;

    std::vector<void *> devInputs_;
    std::vector<void *> devOutputs_;

    std::vector<void *> hostInputs_;
    std::vector<void *> hostOutputs_;
    OperatorDesc *opDesc_;
};

#endif // OP_RUNNER_H
