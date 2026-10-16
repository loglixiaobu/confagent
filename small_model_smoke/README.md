# Qwen3-0.6B 独立小模型烟测

这个目录只验证主项目的端到端工程链路：本地模型生成动作 → API-Bank mock 环境执行 → 自主验证/纠错 → 轨迹落盘 → 汇总与可视化。它与正式 vLLM 实验隔离，不能把 3 条合成任务的结果当作论文指标。

## 目录约束

- 模型：`models/Qwen3-0.6B/`
- Hugging Face / PyTorch 缓存：`cache/`
- 临时文件：`tmp/`
- 轨迹和图表：`outputs/`

上述目录都位于本项目的 E 盘路径下，并由本目录 `.gitignore` 排除。代码与小型 fixture 可以正常提交，模型权重不会进入 Git。

## 运行

在项目根目录执行：

```powershell
$smoke = 'E:\projects\confagent\small_model_smoke'
$env:HF_HOME = "$smoke\cache\huggingface"
$env:HF_HUB_CACHE = "$smoke\cache\huggingface\hub"
$env:TRANSFORMERS_CACHE = "$smoke\cache\huggingface\transformers"
$env:TORCH_HOME = "$smoke\cache\torch"
$env:XDG_CACHE_HOME = "$smoke\cache"
$env:TEMP = "$smoke\tmp"
$env:TMP = "$smoke\tmp"
python small_model_smoke/run_smoke.py
python small_model_smoke/visualize_smoke.py
```

默认比较 `vanilla` 与 `always + critic`，使用同一批任务、同一随机种子。CPU 较慢时可先运行单条：

```powershell
python small_model_smoke/run_smoke.py --limit 1 --conditions vanilla,always --max-new-tokens 128
```

本机默认使用 `--dtype float32`，因为旧 CPU 上 BF16 推理反而明显更慢；若内存不足，可传入 `--dtype auto` 使用权重原始精度。

主要产物：`outputs/trajectories_*.jsonl`、`outputs/summary.json`、`outputs/RESULTS.md` 和 `outputs/smoke_summary.png`。
