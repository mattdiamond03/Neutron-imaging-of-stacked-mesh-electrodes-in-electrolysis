# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:30:38 2026

@author: mattd+markl
"""

# -*- coding: utf-8 -*-
"""
Standalone Gas Fraction Plotter

Reads:
    all_images_roi_pixels_with_gasfraction.csv

Expected columns:
    file, image_type, x, y, T_value, gas_fraction

This script:
    1. Loads the CSV
    2. Filters rows with valid gas_fraction
    3. Computes a straightened x-coordinate:
           x_straight = x - x_left(file, y)
    4. Plots a 3D graph of gas fraction vs x for 5 y-percentiles
       (10th, 30th, 50th, 70th, 90th)
    5. Plots mean gas fraction vs y for each image_type
    6. Plots gas fraction over time for post image type
    7. Plots mean gas fraction over time for electrolysis image types
    8. Optionally plots 2D gas-fraction maps per image_type

Plots are shown on screen and not saved automatically.
"""

import time
import tkinter as tk
from tkinter import filedialog, messagebox


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import re
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

PIXEL_SIZE_MM = 0.02305684

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

def create_root():
    root = tk.Tk()
    root.title("Gas Fraction Plotter")
    root.geometry("300x100+0+0")
    root.iconify()
    root.update()
    return root


def select_csv(root):
    print("\n-------------------------------------------")
    print("SELECT GAS FRACTION CSV")
    print("-------------------------------------------")
    print("Select the CSV with gas_fraction values.")
    time.sleep(0.1)

    root.deiconify()
    root.iconify()
    root.update()

    path = filedialog.askopenfilename(
        parent=root,
        title="Select CSV with gas_fraction",
        filetypes=[("CSV files", "*.csv")]
    )

    root.update()
    root.iconify()

    if path:
        print(f"\nSelected CSV:\n{path}")
    else:
        print("\nNo CSV selected.")

    time.sleep(0.1)
    return path if path else None


def add_straightened_x(df):
    """
    For each file and each y-row, find the leftmost x value in the ROI.
    Then define:
        x_straight = x - x_left
    so that the left edge becomes x_straight = 0 for every row.
    """
    print("\nComputing straightened x-coordinate...")

    df = df.copy()
    df["x_left"] = df.groupby(["file", "y"])["x"].transform("min")
    df["x_straight"] = df["x"] - df["x_left"]

    print("Added columns:")
    print("  - x_left")
    print("  - x_straight")
    time.sleep(0.1)

    return df


def load_data(csv_path):
    print("\nLoading CSV...")
    df = pd.read_csv(csv_path, sep=';', decimal=',')

    required_cols = {"file", "image_type", "x", "y", "gas_fraction"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns:\n"
            "  file, image_type, x, y, gas_fraction\n"
            f"Found: {list(df.columns)}"
        )

    df = df[df["gas_fraction"].notna()].copy()

    if df.empty:
        raise ValueError("No non-NaN gas_fraction values found in this CSV.")

    df["x"] = pd.to_numeric(df["x"], errors="raise")
    df["y"] = pd.to_numeric(df["y"], errors="raise")
    df["gas_fraction"] = pd.to_numeric(df["gas_fraction"], errors="raise")


    df = add_straightened_x(df)

    print(f"Loaded {len(df)} rows with valid gas_fraction.")
    print("Image types found:")
    for t in sorted(df["image_type"].dropna().unique()):
        print(f"  - {t}")

    time.sleep(0.1)
    return df


def choose_x_mode(root):
    use_straight = messagebox.askyesno(
        "X coordinate choice",
        "Do you want to use the STRAIGHTENED x-coordinate?\n\n"
        "Yes = use x_straight (left edge becomes x = 0 in every row)\n"
        "No = use original x",
        parent=root
    )

    if use_straight:
        print("\nUsing straightened x-coordinate: x_straight")
        return "x_straight", "Horizontal position"
    else:
        print("\nUsing original x-coordinate: x")
        return "x", "Original x"


def get_percentile_y_values(df, percentiles=(10, 30, 50, 70, 90)):
    """
    Compute requested y percentiles, then snap each percentile value
    to the nearest actual y present in the data.
    """
    y_unique = np.sort(df["y"].dropna().unique())

    if len(y_unique) == 0:
        raise ValueError("No valid y values found.")

    raw_percentile_values = np.percentile(y_unique, percentiles)
    selected_y = []

    for yp in raw_percentile_values:
        nearest_y = y_unique[np.argmin(np.abs(y_unique - yp))]
        selected_y.append(nearest_y)

    return list(percentiles), selected_y


def plot_electrode_3d(df, ax, x_col, x_label, electrode="cathode", colors=None):

    def parse_current_density(img_type):
        """
        Extract numeric current density from strings like '0.8A'
        Returns float (or None for 'post')
        """
        if isinstance(img_type, str) and "post" in img_type.lower():
            return None
        match = re.search(r"[\d.]+", str(img_type))
        return float(match.group()) if match else None


    percentiles, y_levels = get_percentile_y_values(
        df,
        percentiles=(10, 30, 50, 70, 90)
    )

    current_density_map = {
        "0.2A": "0.2 A",
        "0.4A": "0.4 A",
        "0.8A": "0.8 A",
    }

    image_types = df["image_type"].dropna().unique()

    # Sort image types by current density
    def sort_key(it):
        val = parse_current_density(it)
        return -1 if val is None else val  # post goes last

    image_types = sorted(image_types, key=sort_key)

    # Colormap selection
    cmap = plt.cm.Blues if electrode == "cathode" else plt.cm.Reds

    colors_map = {}
    labels = []

    for i, img in enumerate(image_types):
        if isinstance(img, str) and "post" in img.lower():
            colors_map[img] = "#C200FF"
            label = "post"
        else:
            val = parse_current_density(img)
            label = f"{val:.1f} A/m²" if val is not None else str(img)

            # Match colors to current density
            norm = i / max(len(image_types) - 1, 1)
            colors_map[img] = cmap(norm)

        labels.append(img)

    # Plot
    for image_type in image_types:

        sub_type = df[df["image_type"] == image_type]
        color = colors_map[image_type]

        for y_target in y_levels:

            sub = sub_type[sub_type["y"] == y_target]
            if sub.empty:
                continue

            grouped = (
                sub.groupby(x_col)["gas_fraction"]
                .mean()
                .reset_index()
                .sort_values(x_col)
            )

            xs = grouped[x_col].to_numpy() * PIXEL_SIZE_MM
            ys = np.full(len(xs), y_target * PIXEL_SIZE_MM)
            zs = grouped["gas_fraction"].to_numpy()

            ax.plot(
                xs, ys, zs,
                color=color,
                linewidth=2
            )

    # Labels
    ax.set_xlabel(f"{x_label} (mm)")
    ax.xaxis.set_major_locator(MultipleLocator(0.3))
    
    ax.set_ylabel("Vertical position (mm)", labelpad=14)
    max_y_val = max(y_levels) * PIXEL_SIZE_MM
    ax.set_ylim(0, max_y_val)
    y_ticks_clean = list(range(0, int(max_y_val) + 10, 10))
    ax.set_yticks(y_ticks_clean)
    ax.set_yticklabels([str(tick) for tick in y_ticks_clean])

    ax.set_zlabel("Gas fraction", labelpad=10)
    ax.set_zlim(0, 0.6)
    ax.set_zticks([0, 0.2, 0.4, 0.6])

    ax.set_title(electrode.capitalize())

    # Legend
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=colors_map[it],
            label=("Post-operando" if "post" in str(it).lower()
                   else current_density_map.get(str(it), str(it)))
        )
        for it in image_types
    ]

    ax.legend(handles=legend_handles, loc="upper left", fontsize=14, frameon=True)

    ax.view_init(elev=25, azim=-60)

def plot_3d_figure(cathode_df, anode_df, x_col, x_label, roi_image_path):

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18,7))

    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.4, 0.55, 1.4],   # Controls width of the three columns
        wspace=0.13
    )

    # Anode
    ax1 = fig.add_subplot(gs[0,0], projection="3d")
    ax1.set_box_aspect((1.2, 3.0, 1.0))
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    plot_electrode_3d(anode_df, ax1, x_col, x_label, electrode="anode")
    ax1.set_title("Anode", fontsize=18, pad=10)

    # Middle figure
    ax2 = fig.add_subplot(gs[0,1])

    img = plt.imread(roi_image_path)

    height, width = img.shape[:2]

    ax2.imshow(
        img,
        extent=[
            0,
            width * PIXEL_SIZE_MM,
            0,
            height * PIXEL_SIZE_MM
        ],
        aspect="auto"
    )

    ax2.set_xlabel("Horizontal position (mm)")
    ax2.set_ylabel("Vertical position (mm)")
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax2.set_title("Electrode ROIs", fontsize=18, pad=10)

    # Cathode
    ax3 = fig.add_subplot(gs[0,2], projection="3d")
    ax3.set_box_aspect((1.2, 3.0, 1.0))
    ax3.yaxis.set_major_locator(MultipleLocator(10))
    plot_electrode_3d(cathode_df, ax3, x_col, x_label, electrode="cathode")
    ax3.set_title("Cathode", fontsize=18, pad=10)


    plt.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.93,
        bottom=0.06
    )

def run_once(root):
    print("\n===========================================")
    print("GAS FRACTION PLOTTER STARTED")
    print("===========================================")
    print("This program will:")
    print("1. Load a CSV with gas_fraction values")
    print("2. Compute a straightened x-coordinate")
    print("3. Plot 3D gas fraction vs x for 5 y-percentiles")
    print("4. Plot mean gas fraction vs y for each image_type")
    print("5. Plot gas fraction over time for post images")
    print("6. Plot mean gas fraction over time for electrolysis image types")
    print("7. Optionally plot 2D gas-fraction maps for each image type")
    time.sleep(0.1)

    # Select cathode CSV
    print("\nSelect CATHODE csv")

    cathode_path = select_csv(root)

    if not cathode_path:
        print("Operation cancelled.")
        return False

    cathode_df = load_data(cathode_path)

    # Select anode CSV
    print("\nSelect ANODE csv")

    anode_path = select_csv(root)

    if not anode_path:
        print("Operation cancelled.")
        return False

    anode_df = load_data(anode_path)

    # Choose x-coordinate
    x_col, x_label = choose_x_mode(root)

    roi_image_path = filedialog.askopenfilename(
        title="Select ROI image (cathode/anode overlay)",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")]
    )

    if not roi_image_path:
        print("No ROI image selected.")
        return False

    # Create 3d figure
    plot_3d_figure(
        cathode_df,
        anode_df,
        x_col,
        x_label,
        roi_image_path
    )

    print("\nShowing plots...")
    plt.show()

    messagebox.showinfo(
        "Finished",
        "Plots have been generated and shown on screen.",
        parent=root
    )
    return True


def main():
    root = create_root()

    print("Gas-fraction plotter started.")
    print("Please keep this console window open while using the program.")
    time.sleep(0.1)

    try:
        while True:
            try:
                run_once(root)
            except Exception as e:
                print("\nERROR:")
                print(str(e))
                root.update()
                messagebox.showerror("Error", str(e), parent=root)

            root.update()
            time.sleep(0.1)

            again = messagebox.askyesno(
                "Plot another CSV?",
                "Do you want to plot another gas-fraction CSV?",
                parent=root
            )

            if not again:
                break
            else:
                print("Starting a new plotting round...")
                time.sleep(0.1)

    finally:
        print("Closing application.")
        time.sleep(0.5)
        root.destroy()


if __name__ == "__main__":
    main()