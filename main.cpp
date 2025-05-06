#include <iostream>

#include "backend/metal_mgr.h"
#include "core/allocator.h"
#include "internal/array_view.h"
#include "internal/pattern.h"
#include "internal/view_pack.h"
#include "macros/log.h"
#include "utils/gguf/gguf_def.h"
#include "utils/gguf/gguf_file.h"

int main()
{
  auto& mgr = legrad::metal::Manager::instance();
  auto cpu_allocator = legrad::cpu::CPUAllocator();
  auto cpu_buffer = cpu_allocator.malloc(100);
  return 0;
}