import os
import glob
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import unary_union

def extract_frame(name):
    """
    Extract frame number from names like "IRX_0529_ref_geo"
    """
    m = re.search(r'IRX_(\d+)_ref_geo', name)
    return int(m.group(1)) if m else None

def load_shapefiles(shp_dir):
    """
    Load all shapefiles, compute per-patch metrics,
    and return list of (frame, GeoDataFrame).
    """
    paths = glob.glob(os.path.join(shp_dir, '**', '*.shp'), recursive=True)
    records = []
    for path in sorted(paths):
        fname = os.path.splitext(os.path.basename(path))[0]
        frame = extract_frame(fname)
        gdf = gpd.read_file(path)
        if gdf.empty or frame is None:
            continue
        # project to metric CRS
        metric_crs = 'EPSG:3857'
        gdf = gdf.to_crs(metric_crs)
        # patch metrics
        gdf['area_m2']    = gdf.geometry.area
        gdf['perim_m']    = gdf.geometry.length
        gdf['complexity'] = gdf['perim_m']**2 / gdf['area_m2']
        records.append((frame, gdf))
    return records

def aggregate_metrics(records):
    """
    Aggregate per-frame metrics into a DataFrame.
    """
    rows = []
    for frame, gdf in records:
        rows.append({
            'frame': frame,
            'total_area_m2':      gdf['area_m2'].sum(),
            'mean_patch_area_m2': gdf['area_m2'].mean(),
            'mean_patch_perim_m': gdf['perim_m'].mean(),
            'mean_complexity':    gdf['complexity'].mean(),
        })
    df = pd.DataFrame(rows)
    return df.sort_values('frame')

def save_final_extent(records, out_path):
    """
    Merge all geometries into one shapefile for final burn-extent.
    """
    all_geoms = []
    for _, gdf in records:
        all_geoms.extend(gdf.geometry.tolist())
    if not all_geoms:
        print("No geometries to merge.")
        return
    merged = unary_union(all_geoms)
    final_gdf = gpd.GeoDataFrame(geometry=[merged], crs='EPSG:3857')
    final_gdf.to_file(out_path)
    print(f"Saved final merged extent to {out_path}")

def plot_time_series(df, x, y, title, out_png):
    """
    Plot and save a time-series chart from DataFrame.
    """
    plt.figure(figsize=(8,4))
    plt.plot(df[x], df[y], marker='o')
    plt.xlabel(x.replace('_',' ').title())
    plt.ylabel(y.replace('_',' ').title())
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")
    plt.close()

if __name__ == "__main__":
    # ensure results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load shapefiles and compute per-patch metrics
    shapefile_dir = "mask_shapefiles"
    records = load_shapefiles(shapefile_dir)

    # 2. Aggregate per-frame metrics and save CSV
    df_metrics = aggregate_metrics(records)
    csv_path = os.path.join(results_dir, "burn_report_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    print(df_metrics.head())

    # 3. Merge all extents into one shapefile
    shp_path = os.path.join(results_dir, "final_burn_extent.shp")
    save_final_extent(records, shp_path)

    # 4. Plot total burned area over time
    ts1 = os.path.join(results_dir, "burned_area_timeseries.png")
    plot_time_series(
        df_metrics,
        'frame', 'total_area_m2',
        'Total Burned Area Over Time',
        ts1
    )

    # 5. Plot mean patch area over time
    ts2 = os.path.join(results_dir, "mean_patch_area_timeseries.png")
    plot_time_series(
        df_metrics,
        'frame', 'mean_patch_area_m2',
        'Mean Patch Area Over Time',
        ts2
    )