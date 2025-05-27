# Hanna Hammock Fire Segmentation Pipeline

This project trains a U-Net model to segment fire regions in thermal UAV imagery. It covers data prep, training, evaluation, vectorization, and reporting.

## Requirements

Install with:

```
pip install -r requirements.txt
```

**requirements.txt**:
```
rasterio>=1.2.0
numpy>=1.21.0
torch>=2.0.0
torchvision>=0.15.0
tensorboard>=2.12.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
Pillow>=9.0.0
tqdm>=4.65.0
geopandas>=0.12.0
shapely>=2.0.0
pandas>=1.3.0
```



## Workflow

1. **prep**: generate masks  
2. **train**: fit U-Net  
3. **eval**: create comparison overlays  
4. **analysis**: export polygons and compute area/perimeter  
5. **report**: aggregate metrics, merge extent, plot time series  

Run `make all` to execute the full pipeline end‐to‐end.
