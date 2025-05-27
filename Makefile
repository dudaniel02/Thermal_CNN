.PHONY: prep train eval analysis report all

# Default data root; override with `make DATA_ROOT=...`
DATA_ROOT ?= NADIRPlots/HannaHammock

prep:
	python data_prep.py \
	  --input_dirs "$(DATA_ROOT)/plot1/duringburn/geo_thermal_tiff_celsius" \
	               "$(DATA_ROOT)/plot2/duringburn/geo_thermal_tiff_celsius" \
	  --output_dir masks \
	  --threshold 50.0 \
	  --workers 16

train:
	python train.py \
	  --data_dirs "$(DATA_ROOT)/plot1/duringburn/geo_thermal_tiff_celsius" \
	              "$(DATA_ROOT)/plot2/duringburn/geo_thermal_tiff_celsius" \
	  --mask_dir masks \
	  --runs_dir runs \
	  --ckpt_dir checkpoints \
	  --epochs 2 \
	  --batch_size 16

eval:
	python eval_viz.py \
	  --data_dirs "$(DATA_ROOT)/plot1/duringburn/geo_thermal_tiff_celsius" \
	              "$(DATA_ROOT)/plot2/duringburn/geo_thermal_tiff_celsius" \
	  --mask_dir masks \
	  --ckpt checkpoints/epoch2.pth \
	  --output_dir overlays_cmp \
	  --num_samples 10

analysis:
	python burn_analysis.py

report:
	python burn_report.py

all: prep train eval analysis report