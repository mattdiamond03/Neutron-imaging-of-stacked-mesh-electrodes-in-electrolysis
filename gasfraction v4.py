 # -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:28:40 2026

@author: mattd
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:09:06 2026

@author: mattd
"""

# -*- coding: utf-8 -*-
"""
Gas fraction calculator for big ROI CSV files produced by the ROI analyzer.

Input CSV format (one per run):
    file, image_type, x, y, T_value

Workflow:
    1. Ask user to select one big ROI CSV (e.g. all_images_roi_pixels.csv).
    2. Show available image_type values.
    3. Ask user which image_type(s) are:
         - GAS reference
         - WATER reference
         - ELECTROLYSIS/POST (where gas fraction should be computed)
    4. Average GAS and WATER per pixel (x, y) across all selected files.
    5. Compute gas fraction pixel-by-pixel for all rows whose image_type is
       in the ELECTROLYSIS/POST set.
    6. Add a 'gas_fraction' column to the full table:
         - numeric values for electrolysis/post rows
         - NaN for all other rows
    7. Save a new CSV next to the input file:
         <original_name>_with_gasfraction.csv
    8. Create gas-fraction map images for all electrolysis/post images in
       a 'gas_fraction_images' subfolder.
"""

import time
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_root():
    root = tk.Tk()
    root.title("Gas Fraction from Big ROI CSV")
    root.geometry("300x100+0+0")
    root.iconify()
    root.update()
    return root


def select_big_csv(root):
    print("\n-------------------------------------------")
    print("STEP 1: SELECT BIG ROI CSV")
    print("-------------------------------------------")
    print("Select the CSV produced by the ROI analyzer")
    print("(e.g. all_images_roi_pixels.csv).")
    time.sleep(0.1)

    root.deiconify()
    root.iconify()
    root.update()

    path = filedialog.askopenfilename(
        parent=root,
        title="Select big ROI CSV (all_images_roi_pixels.csv)",
        filetypes=[("CSV files", "*.csv")]
    )

    root.update()
    root.iconify()

    if path:
        print(f"\nSelected CSV:\n{path}")
    else:
        print("\nNo CSV file selected.")

    time.sleep(0.1)
    return path if path else None


def select_types_from_list(root, all_types, title, prompt):
    """
    Show a modal Tkinter window with a multi-selection Listbox
    where the user can choose one or more image_type values.

    Returns a list of selected type strings.
    """
    # Keep root alive, but hidden from the taskbar/screen
    root.update_idletasks()

    win = tk.Toplevel(root)
    win.title(title)
    win.geometry("420x360+200+150")
    win.transient(root)
    win.resizable(True, True)

    selected = []

    label = tk.Label(win, text=prompt, justify="left", wraplength=380)
    label.pack(padx=10, pady=10, anchor="w")

    listbox = tk.Listbox(
        win,
        selectmode=tk.MULTIPLE,
        exportselection=False,
        height=min(len(all_types), 12)
    )
    for t in all_types:
        listbox.insert(tk.END, t)
    listbox.pack(padx=10, pady=(0, 10), fill="both", expand=True)

    btn_frame = tk.Frame(win)
    btn_frame.pack(padx=10, pady=10, fill="x")

    def on_ok(event=None):
        selected.clear()
        selected.extend([all_types[i] for i in listbox.curselection()])
        win.grab_release()
        win.destroy()

    def on_cancel(event=None):
        selected.clear()
        win.grab_release()
        win.destroy()

    tk.Button(btn_frame, text="OK", width=12, command=on_ok).pack(side="right", padx=5)
    tk.Button(btn_frame, text="Cancel", width=12, command=on_cancel).pack(side="right", padx=5)

    win.bind("<Return>", on_ok)
    win.bind("<Escape>", on_cancel)
    win.protocol("WM_DELETE_WINDOW", on_cancel)

    # Force popup to appear properly
    win.update_idletasks()
    win.deiconify()
    win.lift()
    win.focus_force()
    win.wait_visibility()
    win.grab_set()

    if all_types:
        listbox.selection_set(0)
        listbox.activate(0)
        listbox.focus_set()

    win.wait_window()
    return selected
def ask_image_type_groups(root, df):
    """
    Show available image_type values and ask the user which ones correspond
    to GAS, WATER, and ELECTROLYSIS/POST images using multi-select listboxes.
    Returns three lists of strings: gas_types, water_types, elec_types.
    """
    types = sorted(df["image_type"].dropna().unique())
    print("\nAvailable image_type values in this CSV:")
    for t in types:
        print(f"  - {t}")
    time.sleep(0.1)

    gas_types = select_types_from_list(
        root,
        types,
        "Select GAS image types",
        "Select the image_type value(s) that represent the GAS reference\n"
        "(typically your dry / gas-filled state).\n\n"
        "Use Ctrl/Shift to select multiple, then click OK."
    )

    water_types = select_types_from_list(
        root,
        types,
        "Select WATER image types",
        "Select the image_type value(s) that represent the WATER reference\n"
        "(typically your fully wetted / KOH-filled state).\n\n"
        "Use Ctrl/Shift to select multiple, then click OK."
    )

    elec_types = select_types_from_list(
        root,
        types,
        "Select ELECTROLYSIS/POST image types",
        "Select the image_type value(s) for ELECTROLYSIS and POST images\n"
        "where gas fraction should be computed.\n\n"
        "Use Ctrl/Shift to select multiple, then click OK."
    )

    print("\nSelected type groups:")
    print(f"  GAS types:        {gas_types}")
    print(f"  WATER types:      {water_types}")
    print(f"  ELECTROLYSIS/POST:{elec_types}")
    time.sleep(0.1)

    if not gas_types or not water_types or not elec_types:
        raise ValueError(
            "You must select at least one image_type for each of:\n"
            "  GAS, WATER, and ELECTROLYSIS/POST."
        )

    return gas_types, water_types, elec_types


def compute_reference_means(df_all, gas_types, water_types):
    """
    Compute per-pixel mean T_value for GAS and WATER references.

    Returns:
        df_gas_mean: columns x, y, T_value_gas
        df_water_mean: columns x, y, T_value_water
    """
    print("\nComputing per-pixel GAS and WATER reference means...")
    time.sleep(0.1)

    df_gas = df_all[df_all["image_type"].isin(gas_types)].copy()
    df_water = df_all[df_all["image_type"].isin(water_types)].copy()

    if df_gas.empty:
        raise ValueError("No rows found for GAS reference types.")
    if df_water.empty:
        raise ValueError("No rows found for WATER reference types.")

    # Average over all GAS images per pixel
    df_gas_mean = (
        df_gas.groupby(["x", "y"], as_index=False)["T_value"]
        .mean()
        .rename(columns={"T_value": "T_value_gas"})
    )

    # Average over all WATER images per pixel
    df_water_mean = (
        df_water.groupby(["x", "y"], as_index=False)["T_value"]
        .mean()
        .rename(columns={"T_value": "T_value_water"})
    )

    print(f"GAS reference:   {len(df_gas)} rows -> {df_gas_mean.shape[0]} pixels.")
    print(f"WATER reference: {len(df_water)} rows -> {df_water_mean.shape[0]} pixels.")
    time.sleep(0.1)

    return df_gas_mean, df_water_mean


def compute_gas_fraction_for_elec(df_all, df_gas_mean, df_water_mean, elec_types):
    """
    For all rows in df_all whose image_type is in elec_types,
    compute the gas fraction based on the GAS and WATER reference means.

    Returns a DataFrame with columns:
        file, image_type, x, y, gas_fraction
    """
    print("\nComputing gas fraction for ELECTROLYSIS/POST images...")
    time.sleep(0.1)

    df_elec = df_all[df_all["image_type"].isin(elec_types)].copy()
    if df_elec.empty:
        raise ValueError("No rows found for ELECTROLYSIS/POST image types.")

    # Merge per-pixel WATER and GAS references onto each electrolysis/post pixel
    df = (
        df_elec.merge(df_water_mean, on=["x", "y"], how="left")
               .merge(df_gas_mean, on=["x", "y"], how="left")
    )

    # T_value is electrolysis intensity
    Ie = df["T_value"].to_numpy(dtype=np.float64)
    Iw = df["T_value_water"].to_numpy(dtype=np.float64)
    Ig = df["T_value_gas"].to_numpy(dtype=np.float64)

    eps = 1e-12
    Iw = np.clip(Iw, eps, None)
    Ig = np.clip(Ig, eps, None)
    Ie = np.clip(Ie, eps, None)

    numerator = np.log(Iw) - np.log(Ie)
    denominator = np.log(Iw) - np.log(Ig)

    gas_fraction = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = np.abs(denominator) > eps
    gas_fraction[valid] = numerator[valid] / denominator[valid]
    gas_fraction = np.clip(gas_fraction, 0.0, 1.0)

    df["gas_fraction"] = gas_fraction

    print("Gas fraction computation finished for ELECTROLYSIS/POST rows.")
    time.sleep(0.1)

    # Return only the keys + gas_fraction column to merge back later
    return df[["file", "image_type", "x", "y", "gas_fraction"]]


def add_gas_fraction_column(df_all, df_gf):
    """
    Merge the gas_fraction values back into the full table df_all.

    All non-electrolysis rows will get NaN in gas_fraction.
    """
    print("\nMerging gas_fraction back into the full table...")
    time.sleep(0.1)

    # Left-join on (file, image_type, x, y)
    df_merged = df_all.merge(
        df_gf,
        on=["file", "image_type", "x", "y"],
        how="left",
        suffixes=("", "_gf")
    )

    # Ensure there is a single 'gas_fraction' column name
    # (df_gf already uses 'gas_fraction', so we should be fine)
    return df_merged


def save_output_csv(df_full, input_csv_path):
    folder = os.path.dirname(input_csv_path)
    base = os.path.splitext(os.path.basename(input_csv_path))[0]
    out_path = os.path.join(folder, f"{base}_with_gasfraction.csv")

    # Use semicolon separator for European Excel
    df_full.to_csv(out_path, index=False, sep=';', decimal=',')
    print("\nSaved updated CSV with gas_fraction column:")
    print(out_path)
    time.sleep(0.1)
    return out_path


def create_gas_fraction_images(df_full, input_csv_path):
    """
    Create gas-fraction map images for all rows with non-NaN gas_fraction,
    grouped per (file, image_type), and save them into a subfolder
    'gas_fraction_images' next to the input CSV.
    """
    folder = os.path.dirname(input_csv_path)
    images_folder = os.path.join(folder, "gas_fraction_images")
    os.makedirs(images_folder, exist_ok=True)

    df_gf = df_full[df_full["gas_fraction"].notna()].copy()
    if df_gf.empty:
        print("\nNo gas_fraction values found; no images will be created.")
        return None

    print("\nCreating gas-fraction images...")
    time.sleep(0.1)

    for (file_name, image_type), g in df_gf.groupby(["file", "image_type"]):
        # Pivot to 2D array (rows = y, columns = x)
        pivot = g.pivot(index="y", columns="x", values="gas_fraction")
        pivot = pivot.sort_index().sort_index(axis=1)

        data = pivot.values
        n_rows, n_cols = data.shape

        base_width = 6.0
        if n_cols > 0:
            fig_width = base_width
            fig_height = base_width * (n_rows / n_cols)
        else:
            fig_width = base_width
            fig_height = base_width

        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(
            data,
            cmap="viridis",
            aspect="equal",
            origin="upper",
            vmin=0.0,
            vmax=1.0
        )
        plt.colorbar(label="Gas fraction")
        title = f"{file_name} ({image_type})"
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()

        base = os.path.splitext(file_name)[0]
        safe_type = str(image_type).replace(" ", "_")
        png_name = f"{base}_{safe_type}_gas_fraction.png"
        png_path = os.path.join(images_folder, png_name)
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved gas-fraction image: {png_path}")
        time.sleep(0.05)

    print("\nAll gas-fraction images saved in:")
    print(images_folder)
    time.sleep(0.1)
    return images_folder


def run_once(root):
    print("\n===========================================")
    print("GAS FRACTION FROM BIG ROI CSV - STARTED")
    print("===========================================")
    print("This program will:")
    print("1. Load a big ROI CSV produced by the ROI analyzer.")
    print("2. Ask which image_type(s) are GAS, WATER, and ELECTROLYSIS/POST.")
    print("3. Compute per-pixel gas fraction for ELECTROLYSIS/POST images.")
    print("4. Save a new CSV with an added 'gas_fraction' column.")
    print("5. Create gas-fraction map images for all ELECTROLYSIS/POST images.")
    time.sleep(0.1)

    csv_path = select_big_csv(root)
    if not csv_path:
        print("Operation cancelled: no CSV selected.")
        return False

    print("\nLoading big ROI CSV...")
    df_all = pd.read_csv(csv_path, sep=';', decimal=',')
    required_cols = {"file", "image_type", "x", "y", "T_value"}
    if not required_cols.issubset(df_all.columns):
        raise ValueError(
            "Input CSV must contain columns:\n"
            "  file, image_type, x, y, T_value\n"
            f"Found: {list(df_all.columns)}"
        )

    gas_types, water_types, elec_types = ask_image_type_groups(root, df_all)

    df_gas_mean, df_water_mean = compute_reference_means(
        df_all, gas_types, water_types
    )
    df_gf = compute_gas_fraction_for_elec(
        df_all, df_gas_mean, df_water_mean, elec_types
    )
    df_full = add_gas_fraction_column(df_all, df_gf)
    out_csv = save_output_csv(df_full, csv_path)
    images_folder = create_gas_fraction_images(df_full, csv_path)

    print("\n===========================================")
    print("CALCULATION COMPLETE")
    print("===========================================")
    print(f"Updated CSV with gas_fraction saved at:\n{out_csv}")
    if images_folder:
        print(f"Gas-fraction images folder:\n{images_folder}")
    time.sleep(0.1)

    msg = (
        "Gas fraction calculation complete.\n\n"
        f"Updated CSV saved at:\n{out_csv}"
    )
    if images_folder:
        msg += f"\n\nGas-fraction images folder:\n{images_folder}"

    messagebox.showinfo(
        "Finished",
        msg,
        parent=root
    )
    return True


def main():
    root = create_root()

    print("Gas-fraction (big CSV) calculator started.")
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
                "Run another gas-fraction computation?",
                "Do you want to process another big ROI CSV?",
                parent=root
            )

            if not again:
                break
            else:
                print("Starting a new gas-fraction computation round...")
                time.sleep(0.1)

    finally:
        print("Closing application.")
        time.sleep(2)
        root.destroy()


if __name__ == "__main__":
    main()