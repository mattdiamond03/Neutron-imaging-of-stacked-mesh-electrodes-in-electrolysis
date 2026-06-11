# -*- coding: utf-8 -*-
"""
Mean gas fraction above and below surface level per image type
+ heatmaps for visual checking.

Reads:
    all_images_roi_pixels_with_gasfraction.csv

Expected columns:
    file, image_type, x, y, gas_fraction

Output:
    summary_mean_gas_fraction_above_below_surface.csv
"""

import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import matplotlib.pyplot as plt

PIXEL_TO_MM = 0.023605684
SURFACE_LEVEL_MM = 32.0
MAX_Y_MM = 53.0

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})


def create_root():
    root = tk.Tk()
    root.title("Gas Fraction Surface Summary")
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

    return path if path else None


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

    print("\nConverting coordinates from pixels to millimeters...")
    df["x_mm"] = df["x"] * PIXEL_TO_MM
    df["y_mm"] = df["y"] * PIXEL_TO_MM

    df["image_type"] = df["image_type"].replace({
        "post": "post operando"
    })

    print(f"Loaded {len(df)} rows with valid gas_fraction.")
    print("Image types found:")
    for t in sorted(df["image_type"].dropna().unique()):
        print(f"  - {t}")

    return df


def filter_data(df):
    print("\nFiltering data to y_mm <= 53 mm...")
    df_filtered = df[df["y_mm"] <= MAX_Y_MM].copy()

    if df_filtered.empty:
        raise ValueError("No data remains after filtering to y_mm <= 53 mm.")

    return df_filtered


def compute_surface_summary(df):
    print("\nAssigning region relative to surface level at 32 mm...")
    df = df.copy()
    df["region"] = pd.NA
    df.loc[df["y_mm"] < SURFACE_LEVEL_MM, "region"] = "below_surface"
    df.loc[(df["y_mm"] >= SURFACE_LEVEL_MM) & (df["y_mm"] <= MAX_Y_MM), "region"] = "above_surface"

    df = df[df["region"].notna()].copy()

    summary = (
        df.groupby(["image_type", "region"])["gas_fraction"]
        .mean()
        .unstack("region")
        .reset_index()
    )

    summary = summary.rename(columns={
        "below_surface": "mean_gas_fraction_below_32mm",
        "above_surface": "mean_gas_fraction_32_to_53mm"
    })

    if "mean_gas_fraction_below_32mm" not in summary.columns:
        summary["mean_gas_fraction_below_32mm"] = pd.NA

    if "mean_gas_fraction_32_to_53mm" not in summary.columns:
        summary["mean_gas_fraction_32_to_53mm"] = pd.NA

    summary = summary[
        ["image_type", "mean_gas_fraction_below_32mm", "mean_gas_fraction_32_to_53mm"]
    ]

    return summary


def save_summary_excel(summary, input_csv_path):
    output_folder = os.path.dirname(input_csv_path)
    output_path = os.path.join(output_folder, "summary_mean_gas_fraction_above_below_surface.xlsx")

    summary.to_excel(output_path, index=False, sheet_name="summary")

    print(f"\nSaved Excel file:\n{output_path}")
    return output_path

def plot_heatmaps(df):
    print("\nPlotting heatmaps for visual checking...")

    for image_type, sub in df.groupby("image_type"):
        grouped = (
            sub.groupby(["y_mm", "x_mm"], as_index=False)["gas_fraction"]
            .mean()
        )

        pivot = grouped.pivot(index="y_mm", columns="x_mm", values="gas_fraction")
        pivot = pivot.sort_index().sort_index(axis=1)

        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 8))

        im = ax.imshow(
            pivot.values,
            cmap="viridis",
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=0.5,
            extent=[
                pivot.columns.min(), pivot.columns.max(),
                pivot.index.min(), pivot.index.max()
            ]
        )

        ax.axhline(SURFACE_LEVEL_MM, color="white", linestyle="--", linewidth=1.5, label="Surface level (32 mm)")
        ax.axhline(MAX_Y_MM, color="red", linestyle="--", linewidth=1.5, label="Cutoff (53 mm)")

        ax.set_title(f"Gas fraction heatmap - {image_type}")
        ax.set_xlabel("Horizontal position (mm)")
        ax.set_ylabel("Vertical position (mm)")
        ax.legend(loc="upper right", frameon=True)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Gas fraction")

        fig.tight_layout()

    plt.show()


def main():
    root = create_root()

    try:
        print("Gas fraction surface summary started.")
        print("Please keep this console window open while using the program.")

        csv_path = select_csv(root)
        if not csv_path:
            print("Operation cancelled: no CSV selected.")
            return

        df = load_data(csv_path)
        df_filtered = filter_data(df)
        summary = compute_surface_summary(df_filtered)

        print("\nSummary table:")
        print(summary)

        output_path = save_summary_excel(summary, csv_path)

        plot_heatmaps(df_filtered)

        root.update()
        messagebox.showinfo(
            "Finished",
            f"Summary table saved as:\n{output_path}",
            parent=root
        )

    except Exception as e:
        print("\nERROR:")
        print(str(e))
        root.update()
        messagebox.showerror("Error", str(e), parent=root)

    finally:
        print("Closing application.")
        time.sleep(0.5)
        root.destroy()


if __name__ == "__main__":
    main()