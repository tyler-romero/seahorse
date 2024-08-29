from contextlib import contextmanager

import torch
from torch.profiler import ProfilerActivity
from transformers.trainer_callback import TrainerCallback


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


@contextmanager
def maybe_profile(do_profile: bool):
    if not do_profile:
        yield None
        return

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name="results/profiling"),
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof:
        yield prof

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    prof.export_chrome_trace("trace.json")
