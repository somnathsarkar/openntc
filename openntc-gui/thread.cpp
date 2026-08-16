#include <openntc-gui/thread.h>

SharedContext::Access SharedContext::Acquire()
{
  return { std::unique_lock(mutex_), ctx_ };
}

std::optional<SharedContext::Access> SharedContext::TryAcquire()
{
  std::unique_lock<std::mutex> ulock(mutex_, std::try_to_lock);
  if (ulock.owns_lock())
  {
    return Access{ std::move(ulock), ctx_ };
  }
  return std::nullopt;
}