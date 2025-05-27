
import os
import glob
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

def find_mask_paths(mask_dir):
    patterns = [os.path.join(mask_dir, '**', '*.tif'),
                os.path.join(mask_dir, '**', '*.TIFF')]
    paths = []
    for pat in patterns:
        paths += glob.glob(pat, recursive=True)
    return sorted(paths)

def mask_to_polygons(mask_path, out_shp_dir):
    # ensure output directory exists
    os.makedirs(out_shp_dir, exist_ok=True)
    fname = os.path.splitext(os.path.basename(mask_path))[0]
    shp_path = os.path.join(out_shp_dir, f"{fname}.shp")

    # read mask as uint8
    with rasterio.open(mask_path) as src:
        raw = src.read(1)
        transform = src.transform
        crs = src.crs

    # binary mask: 1 for fire, 0 for no-fire
    binmask = (raw > 0).astype('uint8')

    # vectorize only fire pixels
    geoms = []
    for geom, val in shapes(binmask, transform=transform):
        if val == 1:
            geoms.append(shape(geom))

    # build GeoDataFrame
    if geoms:
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
        gdf.to_file(shp_path)
    else:
        gdf = gpd.GeoDataFrame(columns=['geometry'], crs=crs)

    return gdf

def compute_metrics(gdf, metric_crs):
    # project and measure
    gdf_m = gdf.to_crs(metric_crs)
    total_area = gdf_m.geometry.area.sum()
    total_perim = gdf_m.geometry.length.sum()
    return total_area, total_perim

def analyze_masks(mask_dir, out_shp_dir, metric_crs="EPSG:3857"):
    paths = find_mask_paths(mask_dir)
    print(f"Found {len(paths)} mask files in '{mask_dir}'")
    metrics = []
    for path in paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        gdf = mask_to_polygons(path, out_shp_dir)
        area, perim = compute_metrics(gdf, metric_crs)
        metrics.append({'mask': fname, 'area_m2': area, 'perim_m': perim})
    return pd.DataFrame(metrics)

if __name__ == "__main__":
    mask_directory     = "masks"
    output_shapefiles  = "mask_shapefiles"
    csv_out            = "mask_metrics.csv"

    df = analyze_masks(mask_directory, output_shapefiles)
    df.to_csv(csv_out, index=False)
    print(f"Saved metrics to {csv_out}")
    print(df)
