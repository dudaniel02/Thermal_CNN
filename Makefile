
# Path to your HannaHammock plots
ROOT := NADIRPlots/HannaHammock

prep:
	python data_prep.py \
	  --input_dirs "$(ROOT)"/plot*/duringburn \
	  --output_dir masks \
	  --threshold 50.0 \
	  --workers 16

train:
	python train.py \
	  --data_dirs "$(ROOT)"/plot*/duringburn \
	  --mask_dir masks \
	  --runs_dir runs \
	  --ckpt_dir checkpoints \
	  --epochs 200 \
	  --batch_size 16

eval:
	python eval_viz.py \
	  --data_dirs "$(ROOT)"/plot*/duringburn \
	  --ckpt checkpoints/epoch200.pth \
	  --mask_dir masks \
	  --output_dir overlays_cmp \
	  --num_samples 10

analysis:
	python burn_analysis.py

report:
	python burn_report.py

all: prep train eval analysis report
