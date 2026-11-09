"""
Smart Photo Cropper — Evaluation App

Run from the directory containing:
    - Photo_Cropper_Evaluation.xlsx
    - utilities.py
    - downloaded_images/<Primary_Category>/<ID>_image.<ext>

Command:
    streamlit run evaluation_app.py

The app has two parts:
1. Dataset review: manually identify the intended subject in the detections.
2. Functional tests: compare aspect ratios, image-boundary behaviour, and padding.

Results are saved locally:
    evaluation_results.csv
    functional_test_results.csv
"""

from pathlib import Path
import csv
import math

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from openpyxl import load_workbook
from PIL import Image

from utilities import model, crop_single_object


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

WORKBOOK_PATH = Path("Photo_Cropper_Evaluation.xlsx")
SHEET_NAME = "Sheet1"
IMAGE_ROOT = Path("downloaded_images")

REVIEW_RESULTS_PATH = Path("evaluation_results.csv")
FUNCTIONAL_RESULTS_PATH = Path("functional_test_results.csv")

MAX_OBJECTS = 10
ASPECT_RATIOS = [
    "1:1", "2:3", "3:2", "4:3", "3:4",
    "4:5", "5:4", "9:16", "16:9"
]
PADDING_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@st.cache_data
def load_dataset(workbook_path: str, sheet_name: str):
    """Read the first occurrence of the required headers from Sheet1."""
    workbook = load_workbook(workbook_path, data_only=True, read_only=True)
    worksheet = workbook[sheet_name]

    headers = {}
    for col in range(1, worksheet.max_column + 1):
        value = worksheet.cell(1, col).value
        if value is not None:
            # setdefault intentionally keeps the FIRST occurrence.
            headers.setdefault(str(value).strip(), col)

    required = ["ID", "Primary_Category"]
    missing = [name for name in required if name not in headers]
    if missing:
        raise ValueError(f"Missing required workbook columns: {missing}")

    id_col = headers["ID"]
    category_col = headers["Primary_Category"]

    records = []
    for row in range(2, worksheet.max_row + 1):
        raw_id = worksheet.cell(row, id_col).value
        category = worksheet.cell(row, category_col).value

        if raw_id is None or category is None:
            continue

        image_id = str(int(raw_id)) if isinstance(raw_id, (int, float)) and float(raw_id).is_integer() else str(raw_id).strip()
        records.append({
            "ID": image_id,
            "Primary_Category": str(category).strip()
        })

    workbook.close()
    return pd.DataFrame(records)


def locate_image(image_id: str, category: str, image_root: Path):
    """Find an image using the downloader's ID_image naming convention."""
    category_dir = image_root / category
    matches = []
    for extension in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG", "webp", "WEBP"):
        matches.extend(category_dir.glob(f"{image_id}_image.{extension}"))
    if not matches:
        return None
    return sorted(matches)[0]


def read_image(path: Path):
    """Read an image as an OpenCV BGR array."""
    image = cv2.imread(str(path))
    return image


# ---------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------

def get_all_detections(image: np.ndarray):
    """
    Return all YOLO detections sorted by bounding-box area, largest first.

    Important: sort BEFORE taking the top 10. This is the correct way to
    evaluate whether the intended subject survives the app's object limit.
    """
    result = model(image, verbose=False)[0]
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        area = max(0, x2 - x1) * max(0, y2 - y1)

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "class_name": model.names[class_id],
            "class_id": class_id,
            "confidence": confidence,
            "area": area,
        })

    detections.sort(key=lambda item: item["area"], reverse=True)
    return detections


def draw_detections(image: np.ndarray, detections, max_to_draw=40):
    """Draw boxes, highlighting the largest ten separately from the rest."""
    annotated = image.copy()

    for index, item in enumerate(detections[:max_to_draw]):
        x1, y1, x2, y2 = item["bbox"]
        in_top_limit = index < MAX_OBJECTS
        colour = (0, 220, 0) if in_top_limit else (0, 165, 255)
        thickness = 3 if in_top_limit else 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)
        label = f"{index + 1}: {item['class_name']} ({item['confidence']:.2f})"
        cv2.putText(
            annotated, label, (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA
        )

    return annotated


def detection_options(detections):
    options = ["Not detected / no suitable box"]
    for i, item in enumerate(detections):
        limit_text = "TOP 10" if i < MAX_OBJECTS else "OUTSIDE TOP 10"
        options.append(
            f"{i + 1} | {item['class_name']} | "
            f"confidence {item['confidence']:.2f} | {limit_text}"
        )
    return options


# ---------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------

REVIEW_COLUMNS = [
    "ID", "Primary_Category", "Reviewed",
    "Intended_Subject_Detected", "Within_Top_10",
    "Usable_Bounding_Box", "Selected_Detection_Number",
    "Detected_Class", "Confidence", "BBox"
]


def load_review_results():
    if REVIEW_RESULTS_PATH.exists():
        return pd.read_csv(REVIEW_RESULTS_PATH, dtype={"ID": str})
    return pd.DataFrame(columns=REVIEW_COLUMNS)


def save_review_result(record):
    results = load_review_results()
    results = results[results["ID"].astype(str) != str(record["ID"])]
    results = pd.concat([results, pd.DataFrame([record])], ignore_index=True)
    results.to_csv(REVIEW_RESULTS_PATH, index=False)


def save_functional_result(record):
    new_row = pd.DataFrame([record])
    if FUNCTIONAL_RESULTS_PATH.exists():
        existing = pd.read_csv(FUNCTIONAL_RESULTS_PATH)
        existing = pd.concat([existing, new_row], ignore_index=True)
    else:
        existing = new_row
    existing.to_csv(FUNCTIONAL_RESULTS_PATH, index=False)


# ---------------------------------------------------------------------
# Display and helper functions
# ---------------------------------------------------------------------

def show_bgr(image, caption=None):
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)


def parse_ratio(ratio_text):
    width, height = map(int, ratio_text.split(":"))
    return width, height


def cropped_centre_overlay(cropped_image, bbox, crop_origin):
    """Draw centre markers on the final cropped image."""

    overlay = cropped_image.copy()
    crop_h, crop_w = cropped_image.shape[:2]

    crop_x1, crop_y1, _, _ = crop_origin
    x1, y1, x2, y2 = bbox

    subject_centre = (
        (x1 + x2) / 2 - crop_x1,
        (y1 + y2) / 2 - crop_y1
    )

    crop_centre = (crop_w / 2, crop_h / 2)

    # Translate bounding box into cropped-image coordinates
    bx1 = max(0, min(crop_w - 1, x1 - crop_x1))
    by1 = max(0, min(crop_h - 1, y1 - crop_y1))
    bx2 = max(0, min(crop_w - 1, x2 - crop_x1))
    by2 = max(0, min(crop_h - 1, y2 - crop_y1))

    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 220, 0), 2)

    image_centre_point = tuple(map(int, map(round, crop_centre)))
    subject_centre_point = tuple(map(int, map(round, subject_centre)))

    # Red = crop centre
    cv2.drawMarker(
        overlay, image_centre_point, (0, 0, 255),
        cv2.MARKER_CROSS, 28, 2
    )

    # Blue = subject centre
    cv2.drawMarker(
        overlay, subject_centre_point, (255, 0, 0),
        cv2.MARKER_CROSS, 28, 2
    )

    # Yellow = displacement
    cv2.line(
        overlay, image_centre_point, subject_centre_point,
        (0, 255, 255), 2
    )

    dx = subject_centre[0] - crop_centre[0]
    dy = subject_centre[1] - crop_centre[1]

    return overlay, dx, dy, crop_w, crop_h


def selected_image_and_detections(dataset, image_root, key_prefix):
    """Common image and detection selector for the functional-test tabs."""
    choices = []
    for _, row in dataset.iterrows():
        choices.append(f"{row['ID']} — {row['Primary_Category']}")

    selected = st.selectbox("Select an evaluation image", choices, key=f"{key_prefix}_image")
    image_id = selected.split(" — ", 1)[0]
    record = dataset[dataset["ID"].astype(str) == image_id].iloc[0]
    path = locate_image(image_id, record["Primary_Category"], image_root)

    if path is None:
        st.error(
            f"Could not find {image_id}_image.* in "
            f"{image_root / record['Primary_Category']}"
        )
        return None, None, None, None, None

    image = read_image(path)
    if image is None:
        st.error(f"OpenCV could not read: {path}")
        return None, None, None, None, None

    detections = get_all_detections(image)
    if not detections:
        st.warning("YOLO did not return any bounding boxes for this image.")
        return image_id, record, path, image, []

    labels = [
        f"{i + 1} | {d['class_name']} | confidence {d['confidence']:.2f}"
        for i, d in enumerate(detections)
    ]
    selected_label = st.selectbox(
        "Select the intended subject's bounding box",
        labels,
        key=f"{key_prefix}_bbox"
    )
    detection_index = labels.index(selected_label)
    return image_id, record, path, image, (detections, detection_index)


# ---------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------

st.set_page_config(page_title="Photo Cropper Evaluation", layout="wide")
st.title("Smart Photo Cropper — Evaluation")

st.caption(
    "Manual labels are saved as you review each image. "
    "The workbook's first Primary_Category column is used."
)

with st.sidebar:
    st.header("Paths")
    workbook_text = st.text_input("Evaluation workbook", str(WORKBOOK_PATH))
    image_root_text = st.text_input("Downloaded image folder", str(IMAGE_ROOT))
    st.caption(f"Object limit evaluated: {MAX_OBJECTS} largest boxes")

workbook_path = Path(workbook_text)
image_root = Path(image_root_text)

if not workbook_path.exists():
    st.error(f"Workbook not found: {workbook_path.resolve()}")
    st.stop()

try:
    dataset = load_dataset(str(workbook_path), SHEET_NAME)
except Exception as error:
    st.exception(error)
    st.stop()

if dataset.empty:
    st.error("No evaluation records were found in the workbook.")
    st.stop()

review_results = load_review_results()
reviewed_ids = set(review_results.loc[review_results["Reviewed"] == True, "ID"].astype(str)) if not review_results.empty else set()

tab_review, tab_summary, tab_ratios, tab_boundaries, tab_padding = st.tabs([
    "1. Review detections",
    "2. Summary",
    "3. Aspect ratios",
    "4. Image boundaries",
    "5. Padding"
])


# ---------------------------------------------------------------------
# 1. Manual dataset review
# ---------------------------------------------------------------------

with tab_review:
    st.subheader("Review one image at a time")

    dataset = dataset.copy()
    dataset["display"] = dataset.apply(
        lambda row: f"{row['ID']} — {row['Primary_Category']}", axis=1
    )

    remaining = dataset[~dataset["ID"].astype(str).isin(reviewed_ids)]
    review_pool = remaining if not remaining.empty else dataset

    selected_display = st.selectbox(
        "Image to review",
        review_pool["display"].tolist(),
        key="review_image"
    )
    image_id = selected_display.split(" — ", 1)[0]
    image_record = dataset[dataset["ID"].astype(str) == image_id].iloc[0]
    category = image_record["Primary_Category"]
    image_path = locate_image(image_id, category, image_root)

    if image_path is None:
        st.error(
            f"Image file not found. Expected a file like "
            f"{image_root / category / (image_id + '_image.jpg')}"
        )
    else:
        image = read_image(image_path)
        if image is None:
            st.error(f"Could not read image: {image_path}")
        else:
            detections = get_all_detections(image)
            annotated = draw_detections(image, detections)

            left, right = st.columns([1.2, 1])
            with left:
                show_bgr(image, f"Original — ID {image_id}")
                if detections:
                    show_bgr(
                        annotated,
                        "Green = within largest 10 by area; orange = outside largest 10"
                    )
                else:
                    st.warning("No bounding boxes detected.")

            with right:
                st.write(f"**Category:** {category}")
                st.write(f"**Total detections:** {len(detections)}")
                st.write(f"**Detections retained by app limit:** {min(len(detections), MAX_OBJECTS)}")

                existing = review_results[
                    review_results["ID"].astype(str) == str(image_id)
                ]
                old = existing.iloc[0].to_dict() if not existing.empty else {}

                options = detection_options(detections)
                old_number = old.get("Selected_Detection_Number", "")
                try:
                    old_number_int = int(old_number)
                    default_index = old_number_int if 1 <= old_number_int <= len(detections) else 0
                except (ValueError, TypeError):
                    default_index = 0

                with st.form(f"review_form_{image_id}"):
                    selected_option = st.selectbox(
                        "Which box corresponds to the intended subject?",
                        options,
                        index=default_index,
                        help=(
                            "Choose the matching box even if YOLO assigned it the wrong class. "
                            "Choose 'Not detected' if no box encloses the intended subject."
                        )
                    )
                    usable_default = str(old.get("Usable_Bounding_Box", "False")).lower() in ("true", "1", "yes")
                    usable_bbox = st.checkbox(
                        "The selected box is usable for cropping the intended subject",
                        value=usable_default,
                        help="Judge the box geometry, not whether the class label is correct."
                    )
                    submitted = st.form_submit_button("Save review")

                if submitted:
                    selected_position = options.index(selected_option)
                    if selected_position == 0:
                        detected = False
                        within_top10 = False
                        selected_detection_number = ""
                        detected_class = ""
                        confidence = ""
                        bbox_text = ""
                    else:
                        detection_index = selected_position - 1
                        chosen = detections[detection_index]
                        detected = True
                        within_top10 = detection_index < MAX_OBJECTS
                        selected_detection_number = detection_index + 1
                        detected_class = chosen["class_name"]
                        confidence = chosen["confidence"]
                        bbox_text = str(chosen["bbox"])

                    record = {
                        "ID": str(image_id),
                        "Primary_Category": category,
                        "Reviewed": True,
                        "Intended_Subject_Detected": detected,
                        "Within_Top_10": within_top10,
                        "Usable_Bounding_Box": bool(usable_bbox and detected),
                        "Selected_Detection_Number": selected_detection_number,
                        "Detected_Class": detected_class,
                        "Confidence": confidence,
                        "BBox": bbox_text,
                    }
                    save_review_result(record)
                    st.success(f"Saved review for image {image_id}.")
                    st.rerun()

            st.caption(
                "Review all 100 images. A selected detection counts as a detection success; "
                "the top-10 result is calculated from the box's area rank. "
                "For unknown subjects, tick the usable-box checkbox even if the class label is wrong."
            )


# ---------------------------------------------------------------------
# 2. Category-wise summary
# ---------------------------------------------------------------------

with tab_summary:
    st.subheader("Detection and filtering results")

    results = load_review_results()
    if results.empty:
        st.info("No images have been reviewed yet.")
    else:
        results["ID"] = results["ID"].astype(str)
        results["Reviewed"] = results["Reviewed"].astype(str).str.lower().isin(["true", "1", "yes"])
        for col in ["Intended_Subject_Detected", "Within_Top_10", "Usable_Bounding_Box"]:
            results[col] = results[col].astype(str).str.lower().isin(["true", "1", "yes"])

        summary_rows = []
        for category, group in dataset.groupby("Primary_Category", sort=False):
            category_ids = set(group["ID"].astype(str))
            subset = results[results["ID"].isin(category_ids) & results["Reviewed"]]
            reviewed_count = len(subset)
            total_count = len(group)
            detected_count = int(subset["Intended_Subject_Detected"].sum()) if reviewed_count else 0
            top10_count = int(subset["Within_Top_10"].sum()) if reviewed_count else 0
            usable_count = int(subset["Usable_Bounding_Box"].sum()) if reviewed_count else 0

            detected_denominator = detected_count
            summary_rows.append({
                "Category": category,
                "Reviewed": f"{reviewed_count}/{total_count}",
                "Detection success": f"{detected_count}/{reviewed_count} ({detected_count / reviewed_count:.1%})" if reviewed_count else "—",
                "Within top 10 / reviewed": f"{top10_count}/{reviewed_count} ({top10_count / reviewed_count:.1%})" if reviewed_count else "—",
                "Top-10 retention / detected": f"{top10_count}/{detected_denominator} ({top10_count / detected_denominator:.1%})" if detected_denominator else "—",
                "Usable boxes": f"{usable_count}/{reviewed_count}" if reviewed_count else "—",
            })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        total_reviewed = len(results[results["Reviewed"]])
        total_detected = int(results["Intended_Subject_Detected"].sum())
        total_top10 = int(results["Within_Top_10"].sum())
        st.markdown("**Overall (reviewed images only)**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Reviewed", f"{total_reviewed}/{len(dataset)}")
        c2.metric(
            "Detection success",
            f"{total_detected / total_reviewed:.1%}" if total_reviewed else "—"
        )
        c3.metric(
            "Top-10 retention among detected",
            f"{total_top10 / total_detected:.1%}" if total_detected else "—"
        )

        unknown_category = "Single unknown subject"
        unknown_ids = set(
            dataset.loc[dataset["Primary_Category"] == unknown_category, "ID"].astype(str)
        )
        unknown_reviewed = results[
            results["ID"].isin(unknown_ids) & results["Reviewed"]
        ]
        unknown_usable = int(unknown_reviewed["Usable_Bounding_Box"].sum())
        st.markdown("**Unknown-object bounding-box result**")
        st.metric(
            "Usable boxes for unknown-subject images",
            f"{unknown_usable}/{len(unknown_reviewed)} reviewed"
        )
        st.caption(
            "This counts manually judged usable boxes, regardless of the predicted class name. "
            "It is not a measure of correct class classification."
        )

        st.download_button(
            "Download review results CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="evaluation_results.csv",
            mime="text/csv"
        )


# ---------------------------------------------------------------------
# 3. Aspect-ratio functional test
# ---------------------------------------------------------------------

with tab_ratios:
    st.subheader("Test multiple aspect ratios with identical padding")
    image_id, record, path, image, selection = selected_image_and_detections(
        dataset, image_root, "ratio"
    )

    if image is not None and isinstance(selection, tuple):
        detections, detection_index = selection
        chosen = detections[detection_index]
        ratio_choices = st.multiselect(
            "Aspect ratios to compare",
            ASPECT_RATIOS,
            default=["1:1", "3:2", "3:4", "9:16"],
            key="ratio_choices"
        )
        same_padding = st.slider(
            "Padding used for every ratio",
            0.0, 1.0, 0.2, 0.05,
            key="ratio_padding"
        )

        if ratio_choices:
            cols = st.columns(min(3, len(ratio_choices)))
            for i, ratio_text in enumerate(ratio_choices):
                input_w, input_h = parse_ratio(ratio_text)
                cropped, applied = crop_single_object(
                    image, chosen["bbox"], input_w, input_h, same_padding
                )
                with cols[i % len(cols)]:
                    st.markdown(f"**{ratio_text}**")
                    if cropped is None or applied == -1:
                        st.error("Crop failed")
                        looks_good = False
                    else:
                        show_bgr(cropped, f"Applied padding: {applied:.2f}")
                        looks_good = st.checkbox(
                            "Crop looks good",
                            key=f"ratio_good_{image_id}_{ratio_text}"
                        )
                    if st.button("Save ratio result", key=f"save_ratio_{image_id}_{ratio_text}"):
                        save_functional_result({
                            "Test": "Aspect ratio",
                            "ID": image_id,
                            "Category": record["Primary_Category"],
                            "Detection": detection_index + 1,
                            "Aspect_Ratio": ratio_text,
                            "Requested_Padding": same_padding,
                            "Applied_Padding": applied,
                            "Looks_Good": bool(looks_good),
                            "Notes": "",
                        })
                        st.success(f"Saved {ratio_text} result.")


# ---------------------------------------------------------------------
# 4. Image-boundary / centre comparison test
# ---------------------------------------------------------------------

with tab_boundaries:
    st.subheader("Compare image centre and intended-subject centre")

    image_id, record, path, image, selection = selected_image_and_detections(
        dataset, image_root, "boundary"
    )

    if image is not None and isinstance(selection, tuple):

        detections, detection_index = selection
        chosen = detections[detection_index]

        ratio_text = st.selectbox(
            "Aspect ratio",
            ASPECT_RATIOS,
            index=ASPECT_RATIOS.index("3:2"),
            key="boundary_ratio"
        )

        padding = st.slider(
            "Padding",
            0.0, 1.0, 0.2, 0.05,
            key="boundary_padding"
        )

        input_w, input_h = parse_ratio(ratio_text)

        cropped, applied, crop_origin = crop_single_object(
            image,
            chosen["bbox"],
            input_w,
            input_h,
            padding,
            return_crop_coords=True
        )

        if cropped is None or applied == -1:
            st.error("The crop could not be generated.")

        else:
            overlay, dx, dy, width, height = cropped_centre_overlay(
                cropped, chosen["bbox"], crop_origin
            )

            show_bgr(
                overlay,
                "Red = crop centre; blue = subject centre; "
                "yellow = displacement; green = subject bounding box"
            )

            st.write({
                "Crop dimensions (W × H)": f"{width} × {height}",
                "Horizontal displacement (pixels)": round(dx, 2),
                "Vertical displacement (pixels)": round(dy, 2),
                "Horizontal displacement (% width)": round(dx / width * 100, 2),
                "Vertical displacement (% height)": round(dy / height * 100, 2),
                "Applied padding": applied
            })

        boundary_good = st.checkbox(
            "Boundary handling / resulting crop looks satisfactory",
            key=f"boundary_good_{image_id}_{ratio_text}_{padding}"
        )

        boundary_notes = st.text_input(
            "Notes (optional)",
            key=f"boundary_notes_{image_id}_{ratio_text}_{padding}"
        )

        if st.button(
            "Save boundary test result",
            key=f"save_boundary_{image_id}_{ratio_text}_{padding}",
            disabled=(cropped is None or applied == -1)
        ):
            save_functional_result({
                "Test": "Image boundaries",
                "ID": image_id,
                "Category": record["Primary_Category"],
                "Detection": detection_index + 1,
                "Aspect_Ratio": ratio_text,
                "Requested_Padding": padding,
                "Applied_Padding": applied,
                "Looks_Good": boundary_good,
                "Notes": boundary_notes,
            })

            st.success("Boundary test result saved.")

# ---------------------------------------------------------------------
# 5. Padding functional test
# ---------------------------------------------------------------------

with tab_padding:
    st.subheader("Test different padding values on the same image and ratio")
    image_id, record, path, image, selection = selected_image_and_detections(
        dataset, image_root, "padding"
    )

    if image is not None and isinstance(selection, tuple):
        detections, detection_index = selection
        chosen = detections[detection_index]
        ratio_text = st.selectbox(
            "Aspect ratio for padding comparison",
            ASPECT_RATIOS,
            index=ASPECT_RATIOS.index("3:2"),
            key="padding_ratio"
        )
        padding_choices = st.multiselect(
            "Padding values to compare",
            PADDING_LEVELS,
            default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            key="padding_choices"
        )

        input_w, input_h = parse_ratio(ratio_text)
        if padding_choices:
            cols = st.columns(min(3, len(padding_choices)))
            for i, requested_padding in enumerate(padding_choices):
                cropped, applied = crop_single_object(
                    image, chosen["bbox"], input_w, input_h, requested_padding
                )
                with cols[i % len(cols)]:
                    st.markdown(f"**Requested: {requested_padding:.2f}**")
                    if cropped is None or applied == -1:
                        st.error("Crop failed")
                        looks_good = False
                    else:
                        show_bgr(cropped, f"Actually applied: {applied:.2f}")
                        if applied < requested_padding:
                            st.warning("Requested padding was reduced by the crop function.")
                        looks_good = st.checkbox(
                            "Crop looks good",
                            key=f"padding_good_{image_id}_{requested_padding}"
                        )
                    if st.button(
                        "Save padding result",
                        key=f"save_padding_{image_id}_{requested_padding}"
                    ):
                        save_functional_result({
                            "Test": "Padding",
                            "ID": image_id,
                            "Category": record["Primary_Category"],
                            "Detection": detection_index + 1,
                            "Aspect_Ratio": ratio_text,
                            "Requested_Padding": requested_padding,
                            "Applied_Padding": applied,
                            "Looks_Good": bool(looks_good),
                            "Notes": "",
                        })
                        st.success(f"Saved padding {requested_padding:.2f} result.")


st.divider()
st.caption(
    "Functional-test decisions are saved separately from the 100-image detection review. "
    "Back up both CSV files if you want to preserve the evaluation."
)
