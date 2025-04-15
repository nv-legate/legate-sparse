/* Copyright 2022-2024 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "legate/utilities/typedefs.h"
#include "realm/logging.h"
#include <string_view>

namespace sparse {

// Create 1D extents from lower and upper bounds
template <typename T, typename Q>
legate::Rect<1> create_1d_extents(const T& lo, const Q& hi)
{
  return legate::Rect<1>{legate::Point<1>{lo}, legate::Point<1>{hi}};
}

inline Realm::Logger& get_logger()
{
  static Realm::Logger logger("legate-sparse");
  return logger;
}

// Remove the path and use only the filename
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

// Macros for buffer allocation logging
#if ENABLE_BUFFER_LOGGING
#define CREATE_BUFFER(T, size, mem, desc)                                                       \
  [&]() {                                                                                       \
    auto buf = legate::create_buffer<T, 1>(size, mem);                                          \
    get_logger().print() << "Buffer allocation at " << __FILENAME__ << ":" << __LINE__          \
                         << " - Size: " << size << " Type: " << #T << " Description: " << desc; \
    return buf;                                                                                 \
  }()
#else
#define CREATE_BUFFER(T, size, mem, desc) legate::create_buffer<T, 1>(size, mem)
#endif

#if ENABLE_BUFFER_LOGGING
#define LOG_BUFFER(T, size, desc)                                                    \
  get_logger().print() << "Buffer allocation at " << __FILENAME__ << ":" << __LINE__ \
                       << " - Size: " << size << " Type: " << #T << " Description: " << desc
#else
#define LOG_BUFFER(T, size, desc)
#endif

}  // namespace sparse
