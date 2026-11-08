import gc
import torch
import numpy as np
from contextlib import suppress
from fvcore.nn import FlopCountAnalysis

def count_parameters(model):
    """Native PyTorch parameter counting is safer than fvcore's dict."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_stats_complete(
    model: torch.nn.Module, 
    input_res=(3, 112, 112), 
    batch_size_throughput=64, 
    num_warmup=100, 
    num_measure=200, 
    compile_model=True,
    amp_dtype=torch.bfloat16  # Use torch.float16 on Colab T4, or bfloat16/float16 on RTX 5090
):
    model.eval()
    
    # 1. Calculate FLOPs and Params on CPU (Safest for memory)
    cpu_dummy_input = torch.randn(1, *input_res)
    flops = FlopCountAnalysis(model, cpu_dummy_input)
    
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    
    flops_count = flops.total() / 1e9  # GFLOPs (GMACs)
    params_count = count_parameters(model) / 1e6

    if compile_model:
        model = torch.compile(model)

    # 2. Calculate Throughput (FPS) & Batched Peak Memory (MB)
    fps, peak_mem_batch_mb = get_model_stats(
        model, input_res, batch_size=batch_size_throughput, 
        is_throughput=True, num_warmup=num_warmup, num_measure=num_measure, amp_dtype=amp_dtype
    )
    
    # 3. Calculate Latency (ms) & Single-Image Peak Memory (MB)
    latency, peak_mem_single_mb = get_model_stats(
        model, input_res, batch_size=1, 
        is_throughput=False, num_warmup=num_warmup, num_measure=num_measure, amp_dtype=amp_dtype
    )

    # Clean up VRAM completely before the next model in the loop
    model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return params_count, flops_count, latency, fps, peak_mem_batch_mb, peak_mem_single_mb


def get_model_stats(
    model: torch.nn.Module, 
    input_res, 
    batch_size, 
    is_throughput=False, 
    num_warmup=200, 
    num_measure=400,
    amp_dtype=torch.bfloat16
):
    """
    Get the hardware statistics of a model (Latency/Throughput + Peak GPU Memory in MB).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Clear any cached blocks from previous runs before loading the model
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = model.to(device)
    model.eval()

    # Match dummy_input dtype with autocast dtype
    dummy_input = torch.randn(batch_size, *input_res, device=device, dtype=amp_dtype)
    autocast_ctx = torch.autocast('cuda', dtype=amp_dtype) if torch.cuda.is_available() else suppress()

    with torch.no_grad():
        with autocast_ctx:
            # 1. WARMUP PHASE
            for _ in range(num_warmup):
                _ = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Reset peak stats AFTER warmup so compilation overhead is excluded,
                # leaving only (Model Weights + Input + Peak Forward Activations)
                torch.cuda.reset_peak_memory_stats(device)

            # 2. MEASUREMENT PHASE
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_measure)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_measure)]
            
            for i in range(num_measure):
                start_events[i].record()
                _ = model(dummy_input)
                end_events[i].record()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Capture peak memory in Megabytes (MB)
                peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                peak_mem_mb = 0.0

            # 3. CALCULATE METRICS
            times_ms = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
            p5, p95 = np.percentile(times_ms, [5, 95])
            filtered_times = times_ms[(times_ms >= p5) & (times_ms <= p95)]
            
            avg_time_ms = np.mean(filtered_times)

    # Free dummy_input before exiting
    del dummy_input

    if is_throughput:
        fps = (batch_size * 1000.0) / avg_time_ms
        return fps, peak_mem_mb
    else:
        latency = avg_time_ms / batch_size
        return latency, peak_mem_mb