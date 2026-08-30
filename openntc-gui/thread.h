#include <mutex>
#include <optional>
#include <libopenntc/libopenntc.h>

struct SharedFields
{
  bool train_in_progress_;
  bool train_complete_;
  int train_steps_;
  int train_total_steps_;
  double eval_psnr_;
  double eval_mse_;
};

class SharedContext
{
  public:
    struct Access
    {
      std::unique_lock<std::mutex> guard_;
      openntc::Context& ctx_;
    };
    Access Acquire();
    std::optional<Access> TryAcquire();

  private:
    std::mutex mutex_;
    openntc::Context ctx_;
};