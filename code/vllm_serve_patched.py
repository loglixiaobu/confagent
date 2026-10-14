#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vLLM 启动包装器：绕过"某张 GPU 故障导致 vLLM 无法启动"的问题。

背景
--------------------------------------------------------------------
vLLM 的 `CudaPlatform.log_warnings()` 会这样写：

    device_ids = pynvml.nvmlDeviceGetCount()          # 返回物理卡总数（本机 8）
    device_names = [cls._get_physical_device_name(i)  # 遍历 0..7
                    for i in range(device_ids)]

注意它遍历的是**物理索引 0..N-1**，而不是 `CUDA_VISIBLE_DEVICES` 指定的可见卡。
因此只要物理卡 0 出现故障（NVML Unknown Error），即使 `CUDA_VISIBLE_DEVICES=1`，
`nvmlDeviceGetHandleByIndex(0)` 仍会抛异常，导致 vLLM 在打印日志阶段就崩溃。

本包装器把 `_get_physical_device_name` 换成容错版本：单张卡查不到就返回占位名，
不影响其余 GPU 正常启动。

用法（与 `vllm serve` 参数完全一致）：
    python vllm_serve_patched.py serve /path/to/model --port 8000 ...
"""

import sys


def _patch() -> None:
    try:
        from vllm.platforms.cuda import CudaPlatform

        orig_func = CudaPlatform._get_physical_device_name.__func__

        @classmethod
        def _safe_get_physical_device_name(cls, device_id: int = 0) -> str:
            try:
                return orig_func(cls, device_id)
            except Exception:
                return "GPU-%d(unavailable)" % device_id

        CudaPlatform._get_physical_device_name = _safe_get_physical_device_name
        print("[vllm-patch] 已启用 NVML 容错补丁（跳过不可用 GPU 的名称查询）",
              file=sys.stderr)
    except Exception as e:  # 补丁失败不应阻塞启动
        print("[vllm-patch] 打补丁失败（将按原样运行）: %s" % e, file=sys.stderr)


def main() -> int:
    _patch()
    from vllm.entrypoints.cli.main import main as vllm_main
    return vllm_main()


if __name__ == "__main__":
    sys.exit(main())
