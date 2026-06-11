# -*- coding: utf-8 -*-
"""
Standalone Gas Fraction Plotter (mm units)

Reads:
    all_images_roi_pixels_with_gasfraction.csv

Expected columns:
    file, image_type, x, y, T_value, gas_fraction
"""

import os
import re
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


PIXEL_TO_MM = 0.023605684
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
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


def select_output_folder(root):
    print("\n-------------------------------------------")
    print("SELECT OUTPUT FOLDER FOR PDF FILES")
    print("-------------------------------------------")
    print("Select the folder where the PDF files should be saved.")
    time.sleep(0.1)

    root.deiconify()
    root.iconify()
    root.update()

    folder = filedialog.askdirectory(
        parent=root,
        title="Select folder to save PDF files"
    )

    root.update()
    root.iconify()

    if folder:
        print(f"\nSelected output folder:\n{folder}")
    else:
        print("\nNo output folder selected.")

    time.sleep(0.1)
    return folder if folder else None


def add_straightened_x(df):
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

    print("\nConverting pixel coordinates to millimeters...")
    df["x_mm"] = df["x"] * PIXEL_TO_MM
    df["y_mm"] = df["y"] * PIXEL_TO_MM
    df["x_left_mm"] = df["x_left"] * PIXEL_TO_MM
    df["x_straight_mm"] = df["x_straight"] * PIXEL_TO_MM

    y_max_mm = df["y_mm"].max()
    df["y_plot_mm"] = y_max_mm - df["y_mm"]

    df["image_type"] = df["image_type"].replace({
        "post": "post operando"
    })

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
        "Yes = use x_straight in mm (left edge becomes x = 0 mm in every row)\n"
        "No = use original x in mm",
        parent=root
    )

    if use_straight:
        print("\nUsing straightened x-coordinate: x_straight_mm")
        return "x_straight_mm", "Straightened x (mm)"
    else:
        print("\nUsing original x-coordinate: x_mm")
        return "x_mm", "Original x (mm)"


def safe_filename(name):
    name = name.strip().lower()
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-z0-9_\-]+", "", name)
    return name


def save_figures_as_separate_pdfs(figures, output_folder):
    if not figures:
        print("No figures to save.")
        return []

    saved_paths = []

    for i, (fig, filename) in enumerate(figures, start=1):
        safe_name = safe_filename(filename)
        pdf_path = os.path.join(output_folder, f"{i:02d}_{safe_name}.pdf")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        saved_paths.append(pdf_path)
        print(f"Saved PDF:\n{pdf_path}")

    return saved_paths


def plot_mean_vs_y(df):
    print("\nPlotting mean gas fraction vs y...")
    grouped = (
        df.groupby(["image_type", "y_plot_mm"])["gas_fraction"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(5, 10))
    for image_type, sub in grouped.groupby("image_type"):
        sub = sub.sort_values("y_plot_mm")
        ax.plot(sub["mean"], sub["y_plot_mm"], label=image_type)

    ax.set_xlabel("Mean gas fraction over x")
    ax.set_ylabel("Vertical position (mm)")
    ax.set_title("Mean gas fraction across the electrode stack", fontsize=16, pad=10)
    ax.grid(True, alpha=0.3)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    return fig, "mean_gas_fraction_vs_y"


def plot_mean_vs_x(df, x_col, x_label):
    print(f"\nPlotting mean gas fraction vs {x_label.lower()}...")

    grouped = (
        df.groupby(["image_type", x_col])["gas_fraction"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for image_type, sub in grouped.groupby("image_type"):
        sub = sub.sort_values(x_col)
        ax.plot(sub[x_col], sub["mean"], label=image_type, linewidth=2)

    ax.set_xlabel("Horizontal position across the electrode-membrane-electrode stack (mm)")
    ax.set_ylabel("Mean gas fraction taken over y")
    ax.set_title("Mean gas fraction across electrode stack", fontsize=16, pad =10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 0.5)
    fig.tight_layout()
    return fig, "mean_gas_fraction_vs_x"


def plot_post_gasfraction_over_time(df):
    print("\nPlotting gas fraction over time for post operando images...")

    sub = df[df["image_type"] == "post operando"].copy()
    if sub.empty:
        print("No post operando images found.")
        return None

    grouped = (
        sub.groupby("file", as_index=False)["gas_fraction"]
        .mean()
        .sort_values("file")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(range(1, len(grouped) + 1), grouped["gas_fraction"], color="tab:red", linewidth=2)
    ax.set_xlabel("# image")
    ax.set_ylabel("Mean gas fraction")
    ax.set_title("Mean gas fraction over time for post operando images", fontsize=16, pad =10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig, "post_operando_gas_fraction_over_time"


def plot_electrolysis_mean_gasfraction_over_time(df):
    print("\nPlotting average gas fraction for electrolysis image types...")

    sub = df[df["image_type"] != "post operando"].copy()
    if sub.empty:
        print("No electrolysis image types found.")
        return None

    grouped = (
        sub.groupby(["image_type", "file"], as_index=False)["gas_fraction"]
        .mean()
        .sort_values("file")
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for image_type, part in grouped.groupby("image_type"):
        part = part.sort_values("file").reset_index(drop=True)
        ax.plot(range(1, len(part) + 1), part["gas_fraction"], label=image_type, linewidth=2)

    ax.set_xlabel("# image")
    ax.set_ylabel("Mean gas fraction")
    ax.set_title("Mean gas fraction for electrolysis image types over time", fontsize=16, pad =10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig, "electrolysis_mean_gas_fraction_over_time"


def plot_2d_maps(df, x_col, x_label):
    """
    Plot one 2D gas-fraction heatmap per image_type.

    Important:
        - The heatmap is flipped vertically so it appears correct.
        - No mean-vs-y or mean-vs-x side plots are shown.
    """
    print(f"\nPlotting 2D gas-fraction maps using {x_label}...")

    figures = []

    grouped = (
        df.groupby(["image_type", x_col, "y_mm"], as_index=False)["gas_fraction"]
        .mean()
    )

    for image_type, sub in grouped.groupby("image_type"):
        pivot = sub.pivot(index="y_mm", columns=x_col, values="gas_fraction")
        pivot = pivot.sort_index().sort_index(axis=1)

        data = np.flipud(pivot.values)
        y_vals = pivot.index.to_numpy()
        x_vals = pivot.columns.to_numpy()

        n_rows, n_cols = data.shape
        if n_rows == 0 or n_cols == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 12))

        im = ax.imshow(
            data,
            cmap="viridis",
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=0.5,
            extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]
        )

        ax.set_title(f"Gas fraction map - {image_type}", fontsize=16, pad =10)
        ax.set_xlabel("Horizontal position (mm)")
        ax.set_ylabel("Vertical position (mm)")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Gas fraction")

        fig.tight_layout()
        figures.append((fig, f"gas_fraction_map_{image_type}"))

    return figures


def run_once(root):
    print("\n===========================================")
    print("GAS FRACTION PLOTTER STARTED")
    print("===========================================")
    print("This program will:")
    print("1. Load a CSV with gas_fraction values")
    print("2. Compute a straightened x-coordinate (in mm)")
    print("3. Plot mean gas fraction vs x for each image_type")
    print("4. Plot mean gas fraction vs y for each image_type")
    print("5. Plot gas fraction over time for post operando images")
    print("6. Plot mean gas fraction over time for electrolysis image types")
    print("7. Optionally plot advanced 2D gas-fraction heatmaps")
    print("8. Optionally save all plots as separate PDF files")
    time.sleep(0.1)

    csv_path = select_csv(root)
    if not csv_path:
        print("Operation cancelled: no CSV selected.")
        return False

    df = load_data(csv_path)

    x_col, x_label = choose_x_mode(root)

    figures = []

    result = plot_mean_vs_x(df, x_col, x_label)
    if result is not None:
        figures.append(result)

    result = plot_mean_vs_y(df)
    if result is not None:
        figures.append(result)

    result = plot_post_gasfraction_over_time(df)
    if result is not None:
        figures.append(result)

    result = plot_electrolysis_mean_gasfraction_over_time(df)
    if result is not None:
        figures.append(result)

    do_maps = messagebox.askyesno(
        "2D maps",
        "Do you also want to plot advanced 2D gas-fraction maps for each image type?",
        parent=root
    )

    if do_maps:
        map_figures = plot_2d_maps(df, x_col, x_label)
        figures.extend(map_figures)

    save_pdf = messagebox.askyesno(
        "Save PDF files",
        "Do you want to save all generated plots as separate PDF files?",
        parent=root
    )

    if save_pdf:
        output_folder = select_output_folder(root)
        if output_folder:
            saved_paths = save_figures_as_separate_pdfs(figures, output_folder)

            messagebox.showinfo(
                "PDF files saved",
                f"Saved {len(saved_paths)} PDF files in:\n{output_folder}",
                parent=root
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