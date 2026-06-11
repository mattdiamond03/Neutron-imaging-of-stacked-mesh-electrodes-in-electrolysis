# -*- coding: utf-8 -*-
"""
ROI image analyzer with polygon ROI and normalization
- Asks for multiple DARK and multiple OPEN-BEAM TIFF images
- Averages DARK images into 1 avg dark image
- Averages OPEN-BEAM images into 1 avg open-beam image
- Normalizes to transmission T = (I - Idark_avg) / (I0_avg - Idark_avg)
- ROI is selected once on a user-chosen image and reused for all images
- Exports only pixels inside the polygon ROI to a single batch CSV
- Uses a brighter, lower-contrast preview image for easier ROI selection
- Saves all ROI overlay images in a single folder per run
"""

import time
import os
import csv
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon
from matplotlib.path import Path


def create_root():
    root = tk.Tk()
    root.title("ROI Analyzer")
    root.geometry("300x100+0+0")
    root.iconify()
    root.update()
    return root


def select_single_image(root, title):
    print("\n-------------------------------------------")
    print(title)
    print("-------------------------------------------")
    print("A file selection window will now open.")
    time.sleep(0.1)

    root.deiconify()
    root.iconify()
    root.update()

    path = filedialog.askopenfilename(
        parent=root,
        title=title,
        filetypes=[("TIFF files", "*.tif *.tiff")]
    )

    root.update()
    root.iconify()

    if path:
        print(f"\nSelected: {os.path.basename(path)}")
    else:
        print("\nNo file was selected.")

    time.sleep(0.1)
    return path if path else None


def select_multiple_images(root, title):
    print("\n-------------------------------------------")
    print(title)
    print("-------------------------------------------")
    print("A file selection window will now open.")
    print("You can select multiple TIFF files.")
    time.sleep(0.1)

    root.deiconify()
    root.iconify()
    root.update()

    paths = filedialog.askopenfilenames(
        parent=root,
        title=title,
        filetypes=[("TIFF files", "*.tif *.tiff")]
    )

    root.update()
    root.iconify()

    paths = list(paths)

    if paths:
        print(f"\nSelected {len(paths)} image(s):")
        for p in paths:
            print(f"  {os.path.basename(p)}")
    else:
        print("\nNo files were selected.")

    time.sleep(0.1)
    return paths


def select_output_folder(root):
    print("\n-------------------------------------------")
    print("STEP 2: SELECT OUTPUT FOLDER")
    print("-------------------------------------------")
    print("A folder selection window will now open.")
    print("Please choose the folder where the result files should be saved.")
    time.sleep(0.1)

    root.deiconify()
    root.iconify()
    root.update()

    folder = filedialog.askdirectory(
        parent=root,
        title="Select folder to save results"
    )

    root.update()
    root.iconify()

    if folder:
        print("\nSelected output folder:")
        print(folder)
    else:
        print("\nNo output folder was selected.")

    time.sleep(0.1)
    return folder if folder else None


def create_run_output_folder(base_output_folder):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base_output_folder, f"roi_analysis_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"\nCreated output folder for this run:\n{run_folder}")
    time.sleep(0.1)
    return run_folder


def load_grayscale_image(path):
    print("\nLoading TIFF image:")
    print(f"  {os.path.basename(path)}")
    image = Image.open(path)

    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        print("TIFF is 16-bit/integer grayscale. Keeping original grayscale values.")
        image_array = np.array(image)
    elif image.mode == "L":
        print("TIFF is 8-bit grayscale.")
        image_array = np.array(image, dtype=np.uint8)
    else:
        raise ValueError(
            f"Unsupported TIFF mode '{image.mode}'. "
            "Expected grayscale TIFF (8-bit or 16-bit/integer)."
        )

    print(f"Image loaded successfully. Shape: {image_array.shape}")
    print(f"Pixel range in full image: min={np.min(image_array)}, max={np.max(image_array)}")
    time.sleep(0.1)

    return image_array


def load_and_average_images(paths, label):
    if not paths:
        raise ValueError(f"No {label} images were selected.")

    print("\n-------------------------------------------")
    print(f"AVERAGING {label.upper()} IMAGES")
    print("-------------------------------------------")

    arrays = []
    reference_shape = None

    for i, path in enumerate(paths, start=1):
        print(f"\nLoading {label} image {i} of {len(paths)}")
        arr = load_grayscale_image(path).astype(np.float64)

        if reference_shape is None:
            reference_shape = arr.shape
        elif arr.shape != reference_shape:
            raise ValueError(
                f"{label.capitalize()} image shape mismatch:\n"
                f"Expected {reference_shape}, got {arr.shape} for {os.path.basename(path)}.\n"
                "All images in the averaging stack must have the same dimensions."
            )

        arrays.append(arr)

    avg_array = np.mean(np.stack(arrays, axis=0), axis=0)

    print(f"\nAveraged {len(arrays)} {label} image(s).")
    print(f"Averaged {label} image shape: {avg_array.shape}")
    print(f"Averaged {label} pixel range: min={np.min(avg_array):.3f}, max={np.max(avg_array):.3f}")
    time.sleep(0.1)

    return avg_array


def get_display_image(image_array, gamma=0.5):
    img = image_array.astype(np.float64)

    low = np.nanpercentile(img, 5)
    high = np.nanpercentile(img, 95)

    if low == high:
        high = low + 1

    img = np.clip((img - low) / (high - low), 0, 1)
    img = np.power(img, gamma)

    return img


def polygon_to_mask(image_shape, polygon_vertices):
    h, w = image_shape
    y_grid, x_grid = np.mgrid[:h, :w]
    points = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    path = Path(polygon_vertices)
    mask = path.contains_points(points).reshape(h, w)
    return mask


def choose_roi_reference_image(root, image_paths):
    print("\n-------------------------------------------")
    print("STEP 5: CHOOSE ROI REFERENCE IMAGE")
    print("-------------------------------------------")
    print("A file selection window will now open to choose the image")
    print("on which the ROI will be drawn.")
    time.sleep(0.1)

    chosen_path = select_single_image(root, "Select TIFF image to use for ROI selection")
    if not chosen_path:
        raise ValueError("Operation cancelled: no ROI reference image selected.")

    print(f"ROI will be selected on manually chosen image: {os.path.basename(chosen_path)}")
    time.sleep(0.1)
    return chosen_path


def select_polygon_roi(image_array, image_name):
    roi_data = {"vertices": None}
    display_image = get_display_image(image_array, gamma=0.5)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(display_image, cmap="gray", origin="upper", vmin=0, vmax=1)
    ax.set_xlabel("X pixels")
    ax.set_ylabel("Y pixels")
    ax.set_title(
        f"{os.path.basename(image_name)}\n"
        "Click polygon vertices. Double-click to finish. Press Enter to confirm."
    )

    print("\n-------------------------------------------")
    print("STEP 6: POLYGON ROI SELECTION")
    print("-------------------------------------------")
    print(f"Opened ROI reference image: {os.path.basename(image_name)}")
    print("Display preview uses higher brightness and lower contrast for easier selection.")
    print("Click around the ROI to create a polygon.")
    print("Double-click to finish the polygon.")
    print("Press Enter to confirm the polygon.")
    print("Press Esc to start again.")
    time.sleep(0.1)

    polygon_patch = [None]

    def on_select(verts):
        if len(verts) < 3:
            print("Polygon must have at least 3 points.")
            return

        roi_data["vertices"] = np.array(verts, dtype=float)

        if polygon_patch[0] is not None:
            polygon_patch[0].remove()

        polygon_patch[0] = Polygon(
            roi_data["vertices"],
            closed=True,
            fill=False,
            edgecolor="red",
            linewidth=2
        )
        ax.add_patch(polygon_patch[0])
        fig.canvas.draw_idle()

        print(f"Polygon selected with {len(verts)} vertices.")
        print("Press Enter to confirm, or press Esc and redraw.")

    def on_key(event):
        if event.key == "enter":
            if roi_data["vertices"] is None:
                print("No polygon selected yet.")
                return
            print("Polygon ROI confirmed. Closing image window.")
            plt.close(fig)
        elif event.key == "escape":
            roi_data["vertices"] = None
            if polygon_patch[0] is not None:
                polygon_patch[0].remove()
                polygon_patch[0] = None
                fig.canvas.draw_idle()
            print("Polygon cleared. Draw a new polygon.")

    selector = PolygonSelector(
        ax,
        on_select,
        useblit=True,
        props=dict(
            color="red",
            linestyle="-",
            linewidth=2,
            alpha=1.0
        ),
        handle_props=dict(
            marker="o",
            markersize=6,
            markerfacecolor="red",
            markeredgecolor="white",
            alpha=1.0
        )
    )

    fig._polygon_selector = selector
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show(block=True)

    if roi_data["vertices"] is None:
        raise ValueError(
            f"No polygon ROI selected for {image_name}. "
            "Draw a polygon and press Enter."
        )

    return roi_data["vertices"]


def analyze_region(image_array, polygon_vertices, dark_array=None, openbeam_array=None):
    print("Analyzing polygon ROI...")

    region = image_array.astype(np.float64)
    mask = polygon_to_mask(region.shape, polygon_vertices)

    if not np.any(mask):
        raise ValueError("Selected polygon ROI is empty.")

    if dark_array is not None and openbeam_array is not None:
        dark_region = dark_array.astype(np.float64)
        open_region = openbeam_array.astype(np.float64)

        print("Applying normalization: T = (I - Idark_avg) / (I0_avg - Idark_avg)")
        denom = open_region - dark_region

        with np.errstate(divide="ignore", invalid="ignore"):
            T_full = np.where(denom != 0.0, (region - dark_region) / denom, np.nan)
    else:
        print("No normalization images provided. Using raw intensities as T.")
        T_full = region

    T_masked = np.where(mask, T_full, np.nan)
    valid_values = T_masked[mask]

    stats = {
        "mean_T": float(np.nanmean(valid_values)),
        "min_T": float(np.nanmin(valid_values)),
        "max_T": float(np.nanmax(valid_values)),
        "std_T": float(np.nanstd(valid_values)),
        "shape_y": T_masked.shape[0],
        "shape_x": T_masked.shape[1],
        "region_array": T_masked.copy(),
        "mask_array": mask.copy(),
        "dtype": str(T_masked.dtype),
        "pixel_count": int(np.sum(mask))
    }

    print(f"Valid ROI pixels: {stats['pixel_count']}")
    print(f"Mean T:           {stats['mean_T']:.6f}")
    print(f"Min T:            {stats['min_T']:.6f}")
    print(f"Max T:            {stats['max_T']:.6f}")
    print(f"Std T:            {stats['std_T']:.6f}")
    time.sleep(0.1)

    return stats


def save_overlay_image(image_array, polygon_vertices, image_name, image_output_folder):
    display_image = get_display_image(image_array, gamma=0.5)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(display_image, cmap="gray", origin="upper", vmin=0, vmax=1)
    ax.set_title(f"Polygon ROI overlay - {os.path.basename(image_name)}")
    ax.set_xlabel("X pixels")
    ax.set_ylabel("Y pixels")

    poly = Polygon(
        polygon_vertices,
        closed=True,
        fill=False,
        edgecolor="red",
        linewidth=2
    )
    ax.add_patch(poly)

    output_name = os.path.splitext(os.path.basename(image_name))[0] + "_ROI.png"
    output_path = os.path.join(image_output_folder, output_name)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Overlay image saved")
    time.sleep(0.1)
    return output_path


def format_decimal_comma(value):
    return str(value).replace(".", ",")


def save_all_pixels_csv(results, run_output_folder):
    csv_path = os.path.join(run_output_folder, "all_images_roi_pixels.csv")

    with open(csv_path, mode="w",  newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["file", "image_type", "x", "y", "T_value"])

        for res in results:
            roi_array = res["region_array"]
            mask_array = res["mask_array"]
            height, width = roi_array.shape

            for row in range(height):
                for col in range(width):
                    if mask_array[row, col]:
                        T_value = roi_array[row, col]
                        y_flipped = height - 1 - row
                        writer.writerow([
                            res["file"],
                            res["image_type"],
                            col,
                            y_flipped,
                            format_decimal_comma(float(T_value))
                        ])

    print("Combined per-pixel CSV for all images saved:", csv_path)
    time.sleep(0.1)
    return csv_path

def define_image_types_and_images(root):
    print("\n-------------------------------------------")
    print("STEP 1: DEFINE IMAGE TYPES AND SELECT IMAGES")
    print("-------------------------------------------")
    print("You will:")
    print("  - Enter a type name (e.g. 'dry sample', 'KOH wet', '0.4A').")
    print("  - Select all TIFF images that belong to that type.")
    print("Repeat for as many types as you need.")
    time.sleep(0.1)

    image_type_map = {}

    while True:
        type_name = simpledialog.askstring(
            "Image type name",
            "Enter a name for this image type\n"
            "(e.g. 'dry sample', 'KOH wet', '0.4A')\n\n"
            "Leave empty or press Cancel to stop adding types.",
            parent=root
        )

        if not type_name:
            break

        print(f"\nDefine images for type: '{type_name}'")
        root.deiconify()
        root.iconify()
        root.update()

        paths = filedialog.askopenfilenames(
            parent=root,
            title=f"Select TIFF images for type: {type_name}",
            filetypes=[("TIFF files", "*.tif *.tiff")]
        )

        root.update()
        root.iconify()

        if not paths:
            print(f"No images selected for type '{type_name}'.")
        else:
            print(f"Selected {len(paths)} image(s) for type '{type_name}':")
            for p in paths:
                base = os.path.basename(p)
                if p in image_type_map and image_type_map[p] != type_name:
                    messagebox.showwarning(
                        "Image already assigned",
                        f"Image '{base}' was already assigned to type "
                        f"'{image_type_map[p]}'. It will not be reassigned.",
                        parent=root
                    )
                    continue

                image_type_map[p] = type_name
                print(f"  {base}")

        again = messagebox.askyesno(
            "Add another type?",
            "Do you want to define another image type?",
            parent=root
        )
        if not again:
            break

    image_paths = sorted(image_type_map.keys())

    if image_paths:
        print("\nFinal list of images and types:")
        for p in image_paths:
            print(f"  {os.path.basename(p)} -> {image_type_map[p]}")
    else:
        print("\nNo images defined for any type.")

    time.sleep(0.1)
    return image_paths, image_type_map


def process_images(root):
    print("\n===========================================")
    print("ROI IMAGE ANALYZER STARTED")
    print("===========================================")
    print("This program will guide you through these steps 1 by 1:")
    print("1. Define image types and select TIFF image(s)")
    print("2. Select output folder")
    print("3. Select multiple DARK TIFF images")
    print("4. Select multiple OPEN-BEAM TIFF images")
    print("5. Choose which image to use for ROI selection")
    print("6. Draw polygon ROI once")
    print("7. Save results (T values)")
    time.sleep(0.1)

    image_paths, image_type_map = define_image_types_and_images(root)

    if not image_paths:
        print("Operation cancelled: no images selected.")
        return False

    output_folder = select_output_folder(root)
    if not output_folder:
        print("Operation cancelled: no output folder selected.")
        return False

    dark_paths = select_multiple_images(root, "Select DARK TIFF image(s) (beam OFF)")
    if not dark_paths:
        print("Operation cancelled: no dark images selected.")
        return False

    openbeam_paths = select_multiple_images(root, "Select OPEN-BEAM TIFF image(s) (no sample, beam ON)")
    if not openbeam_paths:
        print("Operation cancelled: no open-beam images selected.")
        return False

    dark_array = load_and_average_images(dark_paths, "dark")
    openbeam_array = load_and_average_images(openbeam_paths, "open-beam")

    run_output_folder = create_run_output_folder(output_folder)

    roi_images_folder = os.path.join(run_output_folder, "images_with_roi")
    os.makedirs(roi_images_folder, exist_ok=True)

    roi_reference_path = choose_roi_reference_image(root, image_paths)
    roi_reference_array = load_grayscale_image(roi_reference_path)

    first_image_array = load_grayscale_image(image_paths[0])

    if roi_reference_array.shape != first_image_array.shape:
        raise ValueError(
            "The chosen ROI reference image does not have the same shape as the selected analysis images. "
            "Please choose an ROI image with matching dimensions."
        )

    if roi_reference_array.shape != dark_array.shape or roi_reference_array.shape != openbeam_array.shape:
        raise ValueError(
            "ROI reference image, averaged dark, and averaged open-beam must all have the same shape."
        )

    polygon_vertices = select_polygon_roi(roi_reference_array, roi_reference_path)
    print(f"\nReference polygon ROI fixed with {len(polygon_vertices)} vertices.")

    results = []

    for i, path in enumerate(image_paths, start=1):
        print("\n===========================================")
        print(f"PROCESSING IMAGE {i} OF {len(image_paths)}")
        print("===========================================")
        print(f"Current file: {os.path.basename(path)}")
        time.sleep(0.1)

        image_array = load_grayscale_image(path)

        if image_array.shape != roi_reference_array.shape:
            raise ValueError(
                f"Image shape mismatch for {os.path.basename(path)}. "
                "All analysis images must have the same shape as the ROI reference image."
            )

        if image_array.shape != dark_array.shape or image_array.shape != openbeam_array.shape:
            raise ValueError(
                f"Image shape mismatch for {os.path.basename(path)} versus averaged dark/open-beam."
            )

        print(f"Reusing polygon ROI from reference image with {len(polygon_vertices)} vertices.")

        stats = analyze_region(
            image_array,
            polygon_vertices,
            dark_array=dark_array,
            openbeam_array=openbeam_array
        )
        overlay_path = save_overlay_image(image_array, polygon_vertices, path, roi_images_folder)

        result = {
            "file": os.path.basename(path),
            "image_type": image_type_map[path],
            "polygon_vertices": polygon_vertices,
            "overlay_file": overlay_path,
            **stats
        }
        results.append(result)

        print(f"Finished processing: {os.path.basename(path)}")
        time.sleep(0.1)

    all_pixels_csv = save_all_pixels_csv(results, run_output_folder)

    print("\n===========================================")
    print("ALL PROCESSING COMPLETE")
    print("===========================================")
    print(f"Processed {len(results)} image(s).")
    print(f"Results folder: {run_output_folder}")
    print(f"All pixels CSV: {all_pixels_csv}")
    print(f"ROI overlay images folder: {roi_images_folder}")

    time.sleep(0.1)

    messagebox.showinfo(
        "Finished",
        f"Analysis complete.\n\nProcessed {len(results)} image(s).\n\n"
        f"Saved in:\n{run_output_folder}",
        parent=root
    )
    return True


def main():
    root = create_root()

    print("Program started successfully.")
    print("Please keep this console window open while using the program.")
    time.sleep(0.1)
    try:
        while True:
            try:
                process_images(root)
            except Exception as e:
                print("\nERROR:")
                print(str(e))
                root.update()
                messagebox.showerror("Error", str(e), parent=root)

            root.update()
            time.sleep(0.1)

            again = messagebox.askyesno(
                "Analyze more images?",
                "Do you want to analyze another image or batch of images?",
                parent=root
            )

            if not again:
                break
            else:
                print("Starting a new analysis round...")
                time.sleep(0.1)

    finally:
        print("Closing application.")
        time.sleep(2)
        root.destroy()


if __name__ == "__main__":
    main()