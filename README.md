# SAR-TS-BERT

SAR-TS-BERT is a PyTorch project for self-supervised pretraining on temporal SAR imagery and downstream parcel-level crop classification.

The current research focus is:

- Physics-guided masked autoencoding for SAR time series
- Masking strategy improvements for temporal SAR representation learning
- Dual-branch polarization/physics feature fusion
- Transfer of pretrained encoders to crop classification and parcel inference

## Structure

```text
src/
  data/       Dataset readers for pixel, parcel, processed zarr, and monthly inputs
  datasets/   Dataset-building utilities
  infer/      Parcel-level inference and vector export
  models/     SAR-TS-BERT encoders, classifiers, and DoRA modules
  tools/      Data clipping and preprocessing helpers
  train/      Pretraining, finetuning, testing, and evaluation entry points
docs/         Architecture notes and diagrams
```

Main entry points:

- `src.train.pretrain_encoder`: self-supervised encoder pretraining
- `src.train.finetune_classifier_processed`: parcel classification finetuning on processed zarr
- `src.train.finetune_monthly_classifier`: monthly crop classification
- `src.train.test_classifier`: classification testing and reports
- `src.infer.parcel_infer`: parcel-level inference and shapefile export

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Before training or evaluation, use a Python environment with PyTorch, rasterio, fiona, and zarr installed. If using GPU, verify CUDA first:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

Self-supervised pretraining:

```bash
python -m src.train.pretrain_encoder \
  --zarr_path "<raw_zarr>" \
  --epochs 40 \
  --batch_size 128 \
  --d_model 256 \
  --nhead 8 \
  --layers 6 \
  --ff 512 \
  --mask_ratio 0.15 \
  --noise_scale 0.1 \
  --samples 10000 \
  --seed 42 \
  --train_ratio 0.85 \
  --weight_decay 1e-4 \
  --patience 8 \
  --clip_norm 1.0 \
  --amp \
  --log_path "outputs/pgmae/pretrain_log.csv" \
  --save_best "outputs/pgmae/encoder_best.pt" \
  --save_last "outputs/pgmae/encoder_last.pt"
```

Parcel classification finetuning:

```bash
python -m src.train.finetune_classifier_processed \
  --zarr_path "<processed_zarr>" \
  --epochs 200 \
  --batch_size 256 \
  --d_model 256 \
  --nhead 8 \
  --layers 6 \
  --ff 512 \
  --lr_head 1e-3 \
  --lr_enc 1e-4 \
  --warmup_epochs 10 \
  --patience 25 \
  --train_ratio 0.4 \
  --val_ratio 0.1 \
  --log_path "outputs/exp/cls_train_log.csv" \
  --save_best "outputs/exp/parcel_cls_best.pt" \
  --save_last "outputs/exp/parcel_cls_last.pt" \
  --use_dora \
  --dora_rank 32 \
  --amp
```

Testing:

```bash
python -m src.train.test_classifier \
  --zarr_path "<processed_zarr>" \
  --model_weights "outputs/exp/parcel_cls_best.pt" \
  --batch_size 128 \
  --d_model 256 \
  --nhead 8 \
  --layers 6 \
  --ff 512 \
  --train_ratio 0.4 \
  --val_ratio 0.1 \
  --output_dir "outputs/exp-test" \
  --use_dora \
  --dora_rank 32
```

Parcel inference:

```bash
python -m src.infer.parcel_infer \
  --processed_zarr "<processed_zarr>" \
  --weights "outputs/exp/parcel_cls_best.pt" \
  --vector_path "<parcel_shapefile>" \
  --out_path "outputs/predicted_parcels.shp" \
  --metrics_dir "outputs/exp-infer"
```

## Data and Outputs

Large or local artifacts are not committed:

- `datasets/`
- `outputs/`
- `classifier_outputs/`
- model checkpoints such as `*.pt`, `*.pth`, `*.ckpt`
- local shapefiles and generated vector products

Use a separate `outputs/<experiment_name>/` directory for each experiment to avoid overwriting logs and weights.
