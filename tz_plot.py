import os
from datetime import datetime, timedelta
import re
import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

import geopandas as gpd
import rasterio
import rasterio.plot as rplt
import time


def get_hazard_color(h):
    if 15 <= h < 27:
        return '#4344F8'  # RGB: [67, 68, 248]
    elif 27 <= h < 40:
        return '#EDFC5B'  # RGB: [237, 252, 91]
    elif 40 <= h < 60:
        return '#FFBC52'  # RGB: [255, 188, 82]
    elif h >= 60:
        return '#FF453D'  # RGB: [255, 69, 73]
    else:
        return (1, 1, 1, 0)  # Transparent (RGBA)


def cellHRVis_res(fig_path: str, mesh_gpd: gpd.GeoDataFrame, resHDF5_path: str, var_name: str, var_label: str, vmin: float, vmax: float, res_id_col="CellID", path_prefix=None, base_map=None, title=None, w_h_ratio=None, base_size=1.0, building_shp=None, road_shp=None, boundary_shp=None):
    res_db = h5py.File(resHDF5_path, "r")
    res_val = res_db["h"][()] * 100

    res_val = np.where(res_val < vmin, np.nan, res_val)

    res_id_col = "CID"
    res_id = res_db[res_id_col][()]

    res_h_all = np.zeros(len(mesh_gpd))
    res_h_all[res_id] = res_val
    res_h_all = np.where(res_h_all < vmin, np.nan, res_h_all)

    mesh_gpd["h"] = res_h_all

    mesh_gpd["color"] = mesh_gpd["h"].apply(get_hazard_color)

    boundary_gpd = gpd.read_file(boundary_shp)
    
    w = w_h_ratio * base_size
    h = base_size
    
    fig, ax = plt.subplots(1, 1, figsize=(w, h))

    # ------only keep the cells with water depth larger than vmin
    mesh_gpd_used = mesh_gpd[mesh_gpd["h"] >= vmin]

    for color in set(mesh_gpd_used["color"]):
        mesh_gpd_used[mesh_gpd_used["color"] == color].plot(ax=ax, color=color, linewidth=0.5, edgecolor="none", alpha=1.0)

    if base_map is not None:
        base_img = rasterio.open(base_map)
        base_transform = base_img.transform

        rplt.show(base_img.read(), transform=base_transform, ax=ax)
    
    if building_shp is not None:
        bd_gpd = gpd.read_file(building_shp)
        bd_gpd.plot(ax=ax, linewidth=0.5, edgecolor="slategray", facecolor="slategray")
    
    if road_shp is not None:
        rd_gpd = gpd.read_file(road_shp)
        rd_gpd.plot(ax=ax, linewidth=0.5, edgecolor="dimgray", facecolor="none")
        
    boundary_gpd.plot(ax=ax, linewidth=2.0, edgecolor="black", facecolor="none")

    # ------add custom legend
    #cn_font = FontProperties(fname="/home/lyy/dev/fonts/SimSun.ttf")
    legend_labels = [
        mpatches.Patch(color='#4344F8', label='较低风险'),
        mpatches.Patch(color='#EDFC5B', label='中等风险'),
        mpatches.Patch(color='#FFBC52', label='较高风险'),
        mpatches.Patch(color='#FF453D', label='高风险')
    ]
    plt.legend(handles=legend_labels, loc='lower right')
    
    ax.set_title(title)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    path_prefix = str(path_prefix or '')
    plt.savefig(os.path.join(path_prefix, fig_path), dpi=400)
    plt.clf()
    plt.close(fig)

    # ------record flow information
    res_h = res_db["h"][()]
    res_hu = res_db["hu"][()]
    res_hv = res_db["hv"][()]

    res_id = res_db["CID"][()]
    res_id_all = np.arange(len(mesh_gpd))
    res_h_all = np.zeros(len(mesh_gpd))
    res_h_all[res_id] = res_h
    res_vel_all = np.zeros(len(mesh_gpd))
    res_vel_all[res_id] = np.sqrt(res_hu**2 + res_hv**2) / res_h
    res_vel_all = np.where(np.isnan(res_vel_all), 0.0, res_vel_all)

    mask = (res_h_all >= 0.15)
    res_id_export = res_id_all[mask]
    res_h_export = res_h_all[mask]
    res_vel_export = res_vel_all[mask]
    df_export = pd.DataFrame({"CID": res_id_export, "H": res_h_export, "V": res_vel_export, "HV": res_h_export * res_vel_export})
    df_export.to_csv(os.path.join(path_prefix, fig_path.replace(".png", ".csv")), index=False)

    res_db.close()


def batch_cellHRVis_res(cell_shp: str, resHDF5_dir: str, var_name: str, var_label: str, vmin: float, vmax: float, res_format="res_(\d+.\d+).h5", start_time=None, start_time_d=None, time_step=None, fig_prefix=None, res_id_col="CellID", path_prefix=None, base_map=None, w_h_ratio=None, base_size=1.0, building_shp=None, road_shp=None, boundary_shp=None):
    res_list = [f for f in os.listdir(resHDF5_dir) if len(re.findall(r"{0}".format(res_format), f)) > 0]
    res_list = sorted(res_list, key=lambda x: float(re.findall(r"\w+_(\d+.\d+).h5", x)[0]))

    if time_step is None:
        time_step = 1.0

    if start_time is None:
        start_time = 0.0
    
    if start_time_d is None:
        date_flag = False
    else:
        date_flag = True
        start_time_d = datetime.strptime(start_time_d, "%Y/%m/%d %H:%M:%S")
        time_step_d = timedelta(seconds=time_step)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    mesh_gpd = gpd.read_file(cell_shp, engine='pyogrio', use_arrow=True)
    print(mesh_gpd)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    path_prefix = str(path_prefix or "")
    fig_prefix = str(fig_prefix or "")
    next_time = start_time

    if date_flag:
        next_time_d = start_time_d
        for res in res_list:
            res_path = os.path.join(resHDF5_dir, res)
            res_time = re.findall(r"\w+_(\d+.\d+).h5", res)[0]
            output_name = os.path.join(fig_prefix, next_time_d.strftime("%Y%m%d%H%M%S")+".png")
            title = "洪涝风险等级分布 ({0})".format(next_time_d.strftime("%Y.%m.%d %H:%M:%S"))
            if (float(res_time) == next_time):
                cellHRVis_res(fig_path=output_name, mesh_gpd=mesh_gpd, resHDF5_path=res_path, base_map=base_map,
                                building_shp=building_shp, boundary_shp=boundary_shp, res_id_col=res_id_col, road_shp=road_shp,
                                var_name=var_name, var_label=var_label, vmin=vmin, vmax=vmax, title=title, w_h_ratio=w_h_ratio, base_size=base_size)
                next_time_d = next_time_d + time_step_d
                next_time = next_time + time_step
    else:
        for res in res_list:
            res_path = os.path.join(resHDF5_dir, res)
            res_time = re.findall(r"\w+_(\d+.\d+).h5", res)[0]
            output_name = os.path.join(fig_prefix, res_time+".png")
            title = "洪涝风险等级分布 ({0})".format(next_time_d.strftime("%Y.%m.%d %H:%M:%S"))
            
            if (float(res_time) == next_time):
                cellHRVis_res(fig_path=output_name, mesh_gpd=mesh_gpd, resHDF5_path=res_path, base_map=base_map,
                                building_shp=building_shp, boundary_shp=boundary_shp, res_id_col=res_id_col, road_shp=road_shp,
                                var_name=var_name, var_label=var_label, vmin=vmin, vmax=vmax, title=title, w_h_ratio=w_h_ratio, base_size=base_size)
                next_time += time_step


def cellVis_res(fig_path: str, mesh_gpd: gpd.GeoDataFrame, resHDF5_path: str, var_name: str, var_label: str, vmin: float, vmax: float, res_id_col="CellID", path_prefix=None, base_map=None, title=None, w_h_ratio=None, base_size=1.0, building_shp=None, road_shp=None, boundary_shp=None):
    res_db = h5py.File(resHDF5_path, "r")
    res_val = res_db[var_name][()] * 100

    res_val = np.where(res_val < vmin, np.nan, res_val)

    res_id_col = "CID"
    res_id = res_db[res_id_col][()]

    res_h_all = np.zeros(len(mesh_gpd))
    res_h_all[res_id] = res_val
    res_h_all = np.where(res_h_all < vmin, np.nan, res_h_all)

    mesh_gpd[var_name] = res_h_all

    boundary_gpd = gpd.read_file(boundary_shp)
    
    w = w_h_ratio * base_size
    h = base_size
    
    fig, ax = plt.subplots(1, 1, figsize=(w, h))

    # ------only keep the cells with water depth larger than vmin
    mesh_gpd_used = mesh_gpd[mesh_gpd[var_name] >= vmin]
    im = mesh_gpd_used.plot(column=var_name, cmap="jet", ax=ax, vmin=vmin, vmax=vmax, edgecolor="none", alpha=0.75)

    if base_map is not None:
        base_img = rasterio.open(base_map)
        base_transform = base_img.transform
        rplt.show(base_img.read(), transform=base_transform, ax=ax)
    
    if building_shp is not None:
        bd_gpd = gpd.read_file(building_shp)
        bd_gpd.plot(ax=ax, linewidth=0.5, edgecolor="slategray", facecolor="slategray")
    
    if road_shp is not None:
        rd_gpd = gpd.read_file(road_shp)
        rd_gpd.plot(ax=ax, linewidth=0.5, edgecolor="dimgray", facecolor="none")
        
    boundary_gpd.plot(ax=ax, linewidth=2.0, edgecolor="black", facecolor="none")

    # ------add custom legend
    

    fig = im.get_figure()
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar_position = fig.add_axes([0.23, 0.07, 0.65, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_position, orientation="horizontal")
    _ = cbar.ax.text(-0.10, 0.40, var_label, size=10, va="center", ha="center", transform=cbar.ax.transAxes,)
    
    ax.set_title(title)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    path_prefix = str(path_prefix or '')
    plt.savefig(os.path.join(path_prefix, fig_path), dpi=400)
    plt.clf()
    plt.close(fig)

    res_db.close()


def batch_cellVis_res(cell_shp: str, resHDF5_dir: str, var_name: str, var_label: str, vmin: float, vmax: float, res_format="res_(\d+.\d+).h5", start_time=None, start_time_d=None, time_step=None, fig_prefix=None, res_id_col="CellID", path_prefix=None, base_map=None, w_h_ratio=None, base_size=1.0, building_shp=None, road_shp=None, boundary_shp=None):
    res_list = [f for f in os.listdir(resHDF5_dir) if len(re.findall(r"{0}".format(res_format), f)) > 0]
    res_list = sorted(res_list, key=lambda x: float(re.findall(r"\w+_(\d+.\d+).h5", x)[0]))

    if time_step is None:
        time_step = 1.0

    if start_time is None:
        start_time = 0.0
    
    if start_time_d is None:
        date_flag = False
    else:
        date_flag = True
        start_time_d = datetime.strptime(start_time_d, "%Y/%m/%d %H:%M:%S")
        time_step_d = timedelta(seconds=time_step)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    mesh_gpd = gpd.read_file(cell_shp, engine='pyogrio', use_arrow=True)
    print(mesh_gpd)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    path_prefix = str(path_prefix or "")
    fig_prefix = str(fig_prefix or "")
    next_time = start_time

    if date_flag:
        next_time_d = start_time_d
        for res in res_list:
            res_path = os.path.join(resHDF5_dir, res)
            res_time = re.findall(r"\w+_(\d+.\d+).h5", res)[0]
            output_name = os.path.join(fig_prefix, next_time_d.strftime("%Y%m%d%H%M%S")+".png")
            title = "积水深分布 ({0})".format(next_time_d.strftime("%Y.%m.%d %H:%M:%S"))
            if (float(res_time) == next_time):
                cellVis_res(fig_path=output_name, mesh_gpd=mesh_gpd, resHDF5_path=res_path, base_map=base_map,
                                building_shp=building_shp, boundary_shp=boundary_shp, res_id_col=res_id_col, road_shp=road_shp,
                                var_name=var_name, var_label=var_label, vmin=vmin, vmax=vmax, title=title, w_h_ratio=w_h_ratio, base_size=base_size)
                next_time_d = next_time_d + time_step_d
                next_time = next_time + time_step
    else:
        for res in res_list:
            res_path = os.path.join(resHDF5_dir, res)
            res_time = re.findall(r"\w+_(\d+.\d+).h5", res)[0]
            output_name = os.path.join(fig_prefix, res_time+".png")
            title = "积水深分布 ({0})".format(next_time_d.strftime("%Y.%m.%d %H:%M:%S"))
            
            if (float(res_time) == next_time):
                cellVis_res(fig_path=output_name, mesh_gpd=mesh_gpd, resHDF5_path=res_path, base_map=base_map,
                                building_shp=building_shp, boundary_shp=boundary_shp, res_id_col=res_id_col, road_shp=road_shp,
                                var_name=var_name, var_label=var_label, vmin=vmin, vmax=vmax, title=title, w_h_ratio=w_h_ratio, base_size=base_size)
                next_time += time_step


if __name__ == "__main__":
    cell_shp_path = "C:/Users/86130/Desktop/代码1/算例数据/tz_cell_reorder.gpkg"
    base_map_path = "C:/Users/86130/Desktop/代码1/栅格数据/研究区影像图.tif"
    boundary_shp_path = "C:/Users/86130/Desktop/代码1/矢量数据/shp/研究区边界.shp"

    case_ref = {
        "base": {
            "resHDF5_dir": "C:/Users/86130/Desktop/代码1/算例数据/results_20230729_floodR",
            "fig_prefix": "C:/Users/86130/Desktop/代码1",
        },
    }

    for cas in case_ref.keys():
        if not os.path.exists(case_ref[cas]["fig_prefix"]):
            os.makedirs(case_ref[cas]["fig_prefix"])
        batch_cellVis_res(cell_shp=cell_shp_path, 
                            resHDF5_dir=case_ref[cas]["resHDF5_dir"], 
                            fig_prefix=case_ref[cas]["fig_prefix"],
                            base_map=base_map_path, 
                            boundary_shp=boundary_shp_path, 
                            start_time_d="2023/7/29 12:00:00", 
                            start_time=180000, res_format="_swesRes_(\d+.\d+).h5",
                            time_step=3600, var_name="h", var_label="水深 [cm]", 
                            vmin=10.0, vmax=100.0, w_h_ratio=0.9, base_size=10.0)
        '''
        batch_cellHRVis_res(cell_shp=cell_shp_path, 
                            resHDF5_dir=case_ref[cas]["resHDF5_dir"], 
                            fig_prefix=case_ref[cas]["fig_prefix"],
                            base_map=base_map_path, 
                            boundary_shp=boundary_shp_path, 
                            start_time_d="2023/7/29 12:00:00", 
                            start_time=18000, res_format="_swesRes_(\d+.\d+).h5",
                            time_step=3600, var_name="h", var_label="$h$ [cm]", 
                            vmin=15.0, vmax=100.0, w_h_ratio=1.0, base_size=10.0)
        '''