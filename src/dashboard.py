"""
Streamlit dashboard for BDD100K object detection analysis.

This dashboard visualizes dataset statistics, class distributions,
bounding box properties, and qualitative samples generated during
the analysis stage.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.config import FIGURES_DIR, SAMPLES_DIR, TABLES_DIR


def load_csv_safe(path: Path) -> pd.DataFrame:
    """
    Load a CSV file safely.

    Args:
        path (Path): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame or empty DataFrame if missing.
    """
    if not path.exists():
        st.warning(f"Missing file: {path.name}")
        return pd.DataFrame()
    return pd.read_csv(path)


def show_overview() -> None:
    """Display dataset overview."""
    st.header("📊 Dataset Overview")

    train_dist = load_csv_safe(TABLES_DIR / "class_distribution_train.csv")
    val_dist = load_csv_safe(TABLES_DIR / "class_distribution_val.csv")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Train Split")
        if not train_dist.empty:
            st.dataframe(train_dist)

    with col2:
        st.subheader("Validation Split")
        if not val_dist.empty:
            st.dataframe(val_dist)


def show_class_distribution() -> None:
    """Display class distribution plots."""
    st.header("📈 Class Distribution")

    for split in ["train", "val"]:
        fig_path = FIGURES_DIR / f"class_distribution_{split}.png"
        if fig_path.exists():
            st.subheader(f"{split.capitalize()} Split")
            st.image(str(fig_path), use_container_width=True)
        else:
            st.warning(f"Missing plot for {split} split")


def show_bbox_statistics() -> None:
    """Display bounding box statistics."""
    st.header("📐 Bounding Box Statistics")

    split = st.selectbox("Select split", ["train", "val"])
    bbox_csv = TABLES_DIR / f"bbox_statistics_{split}.csv"

    bbox_df = load_csv_safe(bbox_csv)
    if bbox_df.empty:
        return

    st.subheader("Summary Statistics")
    st.dataframe(bbox_df.describe())

    fig_path = FIGURES_DIR / f"bbox_area_distribution_{split}.png"
    if fig_path.exists():
        st.subheader("Bounding Box Area Distribution")
        st.image(str(fig_path), use_container_width=True)


def show_small_objects() -> None:
    """Display small-object analysis."""
    st.header("🔍 Small Object Analysis")

    split = st.selectbox("Select split", ["train", "val"], key="small_objects")
    csv_path = TABLES_DIR / f"small_objects_{split}.csv"

    small_df = load_csv_safe(csv_path)
    if small_df.empty:
        return

    st.write(
        f"Number of very small objects detected: **{len(small_df)}**"
    )
    st.dataframe(small_df.head(100))


def show_qualitative_samples() -> None:
    """Display qualitative image samples."""
    st.header("🖼️ Qualitative Samples")

    if not SAMPLES_DIR.exists():
        st.warning("No qualitative samples found.")
        return

    image_files = list(SAMPLES_DIR.glob("*.jpg")) + list(
        SAMPLES_DIR.glob("*.png")
    )

    if not image_files:
        st.warning("No images available in samples directory.")
        return

    selected = st.selectbox(
        "Select a sample image",
        image_files,
        format_func=lambda p: p.name,
    )

    image = Image.open(selected)
    st.image(image, caption=selected.name, use_container_width=True)


def main() -> None:
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="BDD100K Object Detection Dashboard",
        layout="wide",
    )

    st.title("BDD100K Object Detection – Data Analysis Dashboard")

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Class Distribution",
            "Bounding Box Statistics",
            "Small Objects",
            "Qualitative Samples",
        ],
    )

    if menu == "Overview":
        show_overview()
    elif menu == "Class Distribution":
        show_class_distribution()
    elif menu == "Bounding Box Statistics":
        show_bbox_statistics()
    elif menu == "Small Objects":
        show_small_objects()
    elif menu == "Qualitative Samples":
        show_qualitative_samples()


if __name__ == "__main__":
    main()