# 基于 SAR 影像的自监督预训练与作物分类

## 项目定位

- 目标：围绕时序 SAR 影像开展自监督预训练，并将编码器迁移到地块级作物分类与推理任务。
- 当前研究主线：Physics-Guided MAE / 掩蔽策略改进 / 双分支极化融合 / 下游作物分类评估。
- ARIS 项目级 skills 已安装在 `.trae/skills/`，当前仓库可直接作为 Trae 的项目级技能工作区使用。

## 代码结构

- `src/train/pretrain_encoder.py`：自监督预训练入口。
- `src/train/finetune_classifier_processed.py`：基于 processed zarr 的地块分类微调入口。
- `src/train/finetune_monthly_classifier.py`：月度作物分类训练入口。
- `src/train/test_classifier.py`：分类测试与评估报告生成。
- `src/infer/parcel_infer.py`：地块级推理、矢量输出与混淆矩阵导出。
- `src/models/`：SAR 编码器、分类器、DoRA 模块。
- `src/data/`：像素级、地块级、月度数据集读取。
- `datasets/`：本地 zarr 数据。
- `outputs/`：训练日志、权重、测试结果。
- `refine-logs/`：ARIS 方案细化、实验规划、评审总结与跟踪记录。

## 运行环境

- 工作目录：`d:\CVRSG\BERT`
- 依赖文件：`requirements.txt`
- 安装方式：`pip install -r requirements.txt`
- 当前 Trae 默认 Python 可用，但未内置 `torch`；运行训练或评估前应先切换到已安装 PyTorch、rasterio、fiona、zarr 的 Python 环境。
- 如使用 GPU，请先确认 `python -c "import torch; print(torch.cuda.is_available())"` 返回 `True`。

## 常用命令模板

### 1. 自监督预训练

```bash
python -m src.train.pretrain_encoder --zarr_path "<zarr路径>" --epochs 40 --batch_size 128 --d_model 256 --nhead 8 --layers 6 --ff 512 --mask_ratio 0.15 --noise_scale 0.1 --samples 10000 --seed 42 --train_ratio 0.85 --weight_decay 1e-4 --patience 8 --clip_norm 1.0 --amp --log_path "<日志路径>" --save_best "<最佳权重>" --save_last "<最后权重>"
```

### 2. processed zarr 地块分类微调

```bash
python -m src.train.finetune_classifier_processed --zarr_path "<processed zarr>" --epochs 200 --batch_size 256 --d_model 256 --nhead 8 --layers 6 --ff 512 --lr_head 1e-3 --lr_enc 1e-4 --warmup_epochs 10 --patience 25 --train_ratio 0.4 --val_ratio 0.1 --log_path "outputs/<exp>/cls_train_log.csv" --save_best "outputs/<exp>/parcel_cls_best.pt" --save_last "outputs/<exp>/parcel_cls_last.pt" --use_dora --dora_rank 32 --amp
```

### 3. 分类测试

```bash
python -m src.train.test_classifier --zarr_path "<processed zarr>" --model_weights "outputs/<exp>/parcel_cls_best.pt" --batch_size 128 --d_model 256 --nhead 8 --layers 6 --ff 512 --train_ratio 0.4 --val_ratio 0.1 --output_dir "outputs/<exp>-test" --use_dora --dora_rank 32
```

### 4. 地块推理

```bash
python -m src.infer.parcel_infer --processed_zarr "<processed zarr>" --weights "outputs/<exp>/parcel_cls_best.pt" --vector_path "<shp路径>" --out_path "outputs/predicted_parcels.shp" --metrics_dir "outputs/<exp>-infer"
```

## 数据与结果约定

- 自监督预训练权重通常与原始影像数据放在同一路径或数据盘目录。
- 分类训练结果默认放在 `outputs/`。
- 推荐每次实验单独使用一个 `outputs/<exp_name>/` 目录，避免权重与日志覆盖。
- 关键研究文档位于 `refine-logs/`，适合作为后续实验、论文与自动评审的输入。

## ARIS 技能推荐用法

### 方案细化

- `research-refine-pipeline`：把模糊的 SAR SSL 想法收敛成问题锚定方案与实验路线图。
- `experiment-plan`：把 Physics-Guided MAE、双分支极化融合、DoRA 微调等思路拆成可执行对比实验。

### 文献与查新

- `research-lit`：搜集时序 SAR、自监督预训练、作物分类、Mamba/Transformer 相关论文。
- `novelty-check`：验证 Physics-Guided MAE 或其他掩蔽策略是否具备新颖性。

### 实验执行

- `run-experiment`：启动训练命令。
- `monitor-experiment`：检查长任务进度与日志。
- `analyze-results`：分析 `outputs/` 与测试结果目录中的 CSV/JSON。

### 结果打磨

- `research-review`：针对方案或结果做外部批评式审阅。
- `auto-review-loop`：在已有初始结果后做多轮审阅—修复—复跑闭环。
- `paper-plan` / `paper-write` / `paper-figure` / `paper-compile`：在结果稳定后生成论文草稿。

## 适合本项目的自然语言调用示例

- 帮我细化一个基于时序 SAR 的物理引导掩蔽预训练方案，并生成实验计划。
- 对 `refine-logs/FINAL_PROPOSAL.md` 做 novelty check，并指出最强 baseline。
- 运行 `src.train.pretrain_encoder` 的 Physics-Guided MAE 版本，并把日志写到 `outputs/pgmae/`。
- 比较 `outputs/` 下不同预训练权重对 Estonia 作物分类测试结果的提升。
- 对当前结果做自动评审循环，重点关注新颖性、消融充分性和可复现性。
