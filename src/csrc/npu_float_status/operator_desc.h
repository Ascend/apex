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

#ifndef OPERATOR_DESC_H
#define OPERATOR_DESC_H

#include <string>
#include <vector>

#include <third_party/acl/inc/acl/acl.h>

/**
 * Op description
 */
struct OperatorDesc {
    /**
     * Constructor
     * @param [in] opType: op type
     */
    explicit OperatorDesc(std::string opType);

    /**
     * Destructor
     */
    ~OperatorDesc();

    /**
     * Add an input tensor description
     * @param [in] dataType: data type
     * @param [in] numDims: number of dims
     * @param [in] dims: dims
     * @param [in] format: format
     * @return OperatorDesc
     */
    OperatorDesc &AddInputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

    /**
     * Add an output tensor description
     * @param [in] dataType: data type
     * @param [in] numDims: number of dims
     * @param [in] dims: dims
     * @param [in] format: format
     * @return OperatorDesc
     */
    OperatorDesc &AddOutputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format);

    std::string opType;
    std::vector<const aclTensorDesc *> inputDesc;
    std::vector<const aclTensorDesc *> outputDesc;
    aclopAttr *opAttr;
};

#endif // OPERATOR_DESC_H
