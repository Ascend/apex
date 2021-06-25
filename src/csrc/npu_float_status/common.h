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

#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#include <third_party/acl/inc/acl/acl.h>
#include <third_party/acl/inc/acl/acl_op_compiler.h>

#define RUN_SUCCESS 0
#define RUN_FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

const std::string OP_TYPE_NPU_GET_FLOAT_STATUS = "NPUGetFloatStatus";
const std::string OP_TYPE_NPU_CLEAR_FLOAT_STATUS = "NPUClearFloatStatus";

const int FLOAT_STATUS_OP_TENSOR_DIMS_SIZE = 8;
const int FLOAT_STATUS_OVERFLOW = 1;

#endif // COMMON_H
