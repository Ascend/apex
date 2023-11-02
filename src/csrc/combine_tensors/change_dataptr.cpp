/*
 * Copyright (c) 2021, Huawei Technologies.All rights reserved.
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

#include <torch/extension.h>
#include <vector>
#include <iostream>


void change_data_ptr(at::Tensor dst, at::Tensor src, int offset)
{
  if (offset < 0) {
    throw std::runtime_error("Expect offset equal or greater than zero, got: " + std::to_string(offset));
  }

  const auto& src_scalar_type = src.scalar_type();
  const auto& dst_scalar_type = dst.scalar_type();

  if ((src_scalar_type != dst_scalar_type) ||
      ((src_scalar_type != at::ScalarType::Half) && (src_scalar_type != at::ScalarType::Float))) {

    throw std::runtime_error(
      "Expect src and dst tensors having the same dtype in float16 or float32.");
  }

  if (dst.numel() * dst.element_size() + offset > src.numel() * src.element_size()) {
    throw std::runtime_error("Offsets overflow in change_data_ptr.");
  }

  if (src.storage().data_ptr().get() == nullptr) {
    throw std::runtime_error("Null storage or data_ptr in change_data_ptr.");
  }

  if (src.scalar_type() == at::ScalarType::Half) {
    at::Half* data_ptr = static_cast<at::Half*>(src.storage().data_ptr().get()) + offset;
    at::DataPtr aim_data_ptr = at::DataPtr(data_ptr, dst.storage().device());
    dst.storage().set_data_ptr(std::move(aim_data_ptr));
  } else {
    float* data_ptr = static_cast<float*>(src.storage().data_ptr().get()) + offset;
    at::DataPtr aim_data_ptr = at::DataPtr(data_ptr, dst.storage().device());
    dst.storage().set_data_ptr(std::move(aim_data_ptr));
  }
}

PYBIND11_MODULE(change_data_ptr, m) {
  m.def("change_data_ptr", &change_data_ptr, "Change data ptr");
}
