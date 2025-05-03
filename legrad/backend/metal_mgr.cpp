#include <memory>
#include <stdexcept>

#include "Foundation/NSError.hpp"
#include "Metal/MTLCounters.hpp"
#include "Metal/MTLResource.hpp"
#include "backend/metal_allocator.h"
#include "macros/log.h"
#include "metal_mgr.h"

namespace legrad::metal
{
Manager::Manager()
{
  device_ = MTL::CreateSystemDefaultDevice();
  LEGRAD_CHECK_AND_THROW(device_ != nullptr, std::runtime_error,
                         "Cannot create default device", 0);

  cmd_queue_ = device_->newCommandQueue();
  LEGRAD_CHECK_AND_THROW(device_ != nullptr, std::runtime_error,
                         "Cannot create command queue", 0);

  // https://developer.apple.com/documentation/metal/creating-a-counter-sample-buffer-to-store-a-gpus-counter-data-during-a-pass?language=objc
  counter_set_ = get_counter_set(MTL::CommonCounterSetTimestamp, device_);
  LEGRAD_CHECK_AND_THROW(counter_set_ != nullptr, std::runtime_error,
                         "Cannot create counter set", 0);

  sample_desc_ = MTL::CounterSampleBufferDescriptor::alloc()->init();
  sample_desc_->setCounterSet(counter_set_);
  sample_desc_->setStorageMode(MTL::StorageModeShared);
  sample_desc_->setSampleCount(2);

  NS::Error* error = nullptr;
  counter_buffer_ = device_->newCounterSampleBuffer(sample_desc_, &error);

  if (error != nullptr) {
    const char* msg = error->localizedDescription()->utf8String();
    LEGRAD_THROW_ERROR(std::runtime_error,
                       "Cannot create counter buffer because {}", msg);
  }

  bucket_allocator_ = std::make_unique<BucketAllocator>(device_);
}

Manager::~Manager()
{
  LEGRAD_LOG_TRACE("MetalMgr destructor called", 0);
  bucket_allocator_ = nullptr;

  LEGRAD_LOG_TRACE("MetalMgr final delete (library, device, command queue)", 0);
  counter_buffer_->release();
  sample_desc_->release();
  counter_set_->release();
  cmd_queue_->release();
  device_->release();
}
}  // namespace legrad::metal