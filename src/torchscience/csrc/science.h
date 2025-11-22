#pragma once

#include <cstdint>

namespace science {
int64_t cuda_version();

namespace detail {
extern "C" inline auto _register_ops = &cuda_version;
}  // namespace detail
}  // namespace science
