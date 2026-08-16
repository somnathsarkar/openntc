#include <mutex>
#include <optional>
#include <libopenntc/libopenntc.h>

struct SharedFields
{
  bool train_in_progress;
  bool train_complete;
  int train_steps;
  int train_total_steps;
  double eval_psnr;
  double eval_mse;
};

class SharedContext
{
  public:
    struct Access
    {
      std::unique_lock<std::mutex> guard_;
      OpenNTCContext& ctx_;
    };
    Access Acquire();
    std::optional<Access> TryAcquire();

  private:
    std::mutex mutex_;
    OpenNTCContext ctx_;
};