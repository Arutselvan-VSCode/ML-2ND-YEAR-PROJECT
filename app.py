import os

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset

from module_1 import xBDDataset
from module_3 import DisasterUNet
from train_baselines import FEATURE_COLUMNS as PAIR_FEATURE_COLUMNS
from train_baselines import create_algorithms, extract_pair_features


CLASS_NAMES = ["Background", "No Damage", "Minor Damage", "Major Damage", "Destroyed"]
CLASS_IDS = list(range(len(CLASS_NAMES)))
DAMAGE_CLASS_IDS = [1, 2, 3, 4]
DAMAGE_CLASS_NAMES = [CLASS_NAMES[class_id] for class_id in DAMAGE_CLASS_IDS]
CUSTOM_FEATURE_COLUMNS = [
    "red_mean",
    "green_mean",
    "blue_mean",
    "red_std",
    "green_std",
    "blue_std",
    "brightness_mean",
    "brightness_std",
    "edge_density",
    "texture_laplacian",
]
COLOR_MAP = {
    0: [0, 0, 0],
    1: [0, 255, 0],
    2: [255, 255, 0],
    3: [255, 128, 0],
    4: [255, 0, 0],
}
ALGORITHM_NOTES = {
    "Random Forest": "Bagging tree ensemble; handles nonlinear image features well and uses balanced class weights.",
    "Extra Trees": "Randomized tree ensemble; strong for small tabular feature sets and reduces variance.",
    "Gradient Boosting": "Sequential boosted trees; learns hard examples step by step from feature errors.",
    "SVM (RBF Kernel)": "Kernel margin classifier; uses scaling and an RBF kernel for nonlinear class boundaries.",
    "Logistic Regression": "Linear baseline; uses scaling and balanced weights for a clean reference model.",
}
BASELINE_MODEL_FILES = {
    "Random Forest": "random_forest_baseline.pkl",
    "Extra Trees": "extra_trees_baseline.pkl",
    "Gradient Boosting": "gradient_boosting_baseline.pkl",
    "SVM (RBF Kernel)": "svm_rbf_kernel_baseline.pkl",
    "Logistic Regression": "logistic_regression_baseline.pkl",
}
UPLOADED_MODEL_FILES = {
    "Random Forest": "uploaded_random_forest_model.pkl",
    "Extra Trees": "uploaded_extra_trees_model.pkl",
    "Gradient Boosting": "uploaded_gradient_boosting_model.pkl",
    "SVM (RBF Kernel)": "uploaded_svm_rbf_kernel_model.pkl",
    "Logistic Regression": "uploaded_logistic_regression_model.pkl",
}


st.set_page_config(page_title="Disaster Assessment & Rescue GUI", layout="wide")

dark_theme = st.sidebar.toggle("Dark Theme", value=True, help="Switch between dark operations mode and light mode.")

st.markdown(
    """
    <style>
    :root {
        --page-bg: #f6f8fb;
        --panel-bg: #ffffff;
        --ink: #172033;
        --muted: #667085;
        --line: #dde4ee;
        --brand: #0f766e;
        --brand-strong: #115e59;
        --accent: #dc2626;
        --amber: #d97706;
        --blue: #2563eb;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 34rem),
            linear-gradient(180deg, #f8fafc 0%, var(--page-bg) 48%, #eef3f8 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }

    h3 {
        font-size: 1.15rem;
        font-weight: 760;
    }

    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #e5edf6;
    }

    [data-testid="stSidebar"] input {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #ffffff;
    }

    .app-hero {
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 94, 89, 0.94)),
            url("https://images.unsplash.com/photo-1527482797697-8795b05a13fe?auto=format&fit=crop&w=1800&q=80");
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        color: #ffffff;
        min-height: 265px;
        padding: clamp(1.6rem, 4vw, 3.2rem);
        margin-bottom: 1.25rem;
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.16);
    }

    .eyebrow {
        color: #99f6e4;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        margin-bottom: 0.7rem;
        text-transform: uppercase;
    }

    .app-hero h1 {
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3.65rem);
        line-height: 1.04;
        margin: 0;
        max-width: 900px;
    }

    .app-hero p {
        color: #d7f4ef;
        font-size: 1.05rem;
        line-height: 1.65;
        margin: 1rem 0 0;
        max-width: 760px;
    }

    .hero-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.45rem;
    }

    .hero-stat {
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        min-width: 150px;
    }

    .hero-stat strong {
        color: #ffffff;
        display: block;
        font-size: 1.05rem;
    }

    .hero-stat span {
        color: #b7c8d6;
        display: block;
        font-size: 0.78rem;
        margin-top: 0.15rem;
    }

    .section-lead {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.65;
        margin-top: -0.25rem;
    }

    div[data-testid="stTabs"] button {
        border-radius: 8px 8px 0 0;
        color: #475467;
        font-weight: 700;
        padding: 0.75rem 1rem;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: #ffffff;
        border-bottom: 3px solid var(--brand);
        color: var(--brand-strong);
    }

    div[data-testid="stMetric"] {
        background: var(--panel-bg);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.95rem 1rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        min-height: 112px;
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 760;
        letter-spacing: 0.02em;
    }

    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 800;
    }

    div[data-testid="stFileUploader"] {
        background: var(--panel-bg);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 12px 34px rgba(15, 23, 42, 0.055);
    }

    div[data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed #8bb5ad;
        border-radius: 8px;
        min-height: 145px;
        align-items: center;
        justify-content: center;
        background: #f0fdfa;
        transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
    }

    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--brand);
        background: #e6fffb;
        transform: translateY(-1px);
    }

    div[data-testid="stFileUploaderDropzone"]::after {
        content: "Secure satellite imagery upload";
        color: #52716f;
        font-size: 0.9rem;
        font-weight: 650;
        margin-left: 1rem;
    }

    .stButton > button {
        border-radius: 8px;
        border: 1px solid rgba(15, 118, 110, 0.25);
        font-weight: 800;
        min-height: 3rem;
        padding: 0.65rem 1.15rem;
        transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
    }

    .stButton > button:hover:enabled {
        box-shadow: 0 14px 28px rgba(15, 118, 110, 0.18);
        transform: translateY(-1px);
    }

    .stButton > button[kind="primary"] {
        background: var(--brand);
        border-color: var(--brand);
        color: #ffffff;
    }

    .stButton > button[kind="primary"]:hover:enabled {
        background: var(--brand-strong);
        border-color: var(--brand-strong);
    }

    .stDataFrame, [data-testid="stImage"], [data-testid="stPyplot"] {
        border-radius: 8px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
        border-width: 1px;
    }

    div[data-testid="stProgress"] > div > div {
        background-color: var(--brand);
    }

    .legend-row {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin: 0.35rem 0 0.9rem;
    }

    .legend-pill {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 8px;
        color: #344054;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.45rem 0.7rem;
    }

    .legend-dot {
        border-radius: 999px;
        display: inline-block;
        height: 0.65rem;
        margin-right: 0.35rem;
        vertical-align: -0.05rem;
        width: 0.65rem;
    }

    @media (max-width: 700px) {
        .app-hero {
            min-height: auto;
            padding: 1.35rem;
        }

        .hero-stat {
            flex: 1 1 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if dark_theme:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #05070b;
            --panel-bg: #0f172a;
            --ink: #f8fafc;
            --muted: #98a2b3;
            --line: rgba(148, 163, 184, 0.22);
            --brand: #14b8a6;
            --brand-strong: #2dd4bf;
            --accent: #f87171;
            --amber: #f59e0b;
            --blue: #60a5fa;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(20, 184, 166, 0.17), transparent 31rem),
                radial-gradient(circle at 82% 12%, rgba(96, 165, 250, 0.13), transparent 28rem),
                linear-gradient(180deg, #030712 0%, #07111f 48%, #05070b 100%);
            color: var(--ink);
        }

        .app-hero {
            background:
                linear-gradient(135deg, rgba(2, 6, 23, 0.90), rgba(13, 74, 68, 0.82)),
                url("https://images.unsplash.com/photo-1527482797697-8795b05a13fe?auto=format&fit=crop&w=1800&q=80");
            background-size: cover;
            background-position: center;
            border-color: rgba(45, 212, 191, 0.22);
            box-shadow: 0 28px 80px rgba(0, 0, 0, 0.38);
        }

        h1, h2, h3,
        div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] h2,
        div[data-testid="stMarkdownContainer"] h3 {
            color: var(--ink);
        }

        p, label, span,
        div[data-testid="stMarkdownContainer"] p {
            color: #d0d5dd;
        }

        .section-lead {
            color: #aab4c4;
        }

        [data-testid="stSidebar"] {
            background: #020617;
            border-right: 1px solid rgba(45, 212, 191, 0.14);
        }

        div[data-testid="stTabs"] button {
            background: rgba(15, 23, 42, 0.82);
            color: #cbd5e1;
            border: 1px solid rgba(148, 163, 184, 0.16);
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: #0f172a;
            border-bottom: 3px solid var(--brand);
            color: #ccfbf1;
        }

        div[data-testid="stMetric"],
        div[data-testid="stFileUploader"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(11, 18, 32, 0.98));
            border-color: rgba(148, 163, 184, 0.20);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.24);
        }

        div[data-testid="stMetricLabel"] p {
            color: #aab4c4;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(20, 184, 166, 0.08);
            border-color: rgba(45, 212, 191, 0.42);
        }

        div[data-testid="stFileUploaderDropzone"]:hover {
            background: rgba(20, 184, 166, 0.14);
            border-color: var(--brand-strong);
        }

        div[data-testid="stFileUploaderDropzone"]::after {
            color: #99f6e4;
        }

        .legend-pill {
            background: #111827;
            border-color: rgba(148, 163, 184, 0.22);
            color: #e5e7eb;
        }

        div[data-testid="stAlert"] {
            background: rgba(15, 23, 42, 0.92);
            border-color: rgba(148, 163, 184, 0.24);
            color: #e5e7eb;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            background: #0f172a;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
        }

        input, textarea,
        div[data-baseweb="input"] input {
            background: #111827;
            border-color: rgba(148, 163, 184, 0.24);
            color: #f8fafc;
        }

        .stButton > button {
            background: #111827;
            border-color: rgba(45, 212, 191, 0.36);
            color: #e5e7eb;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0f766e, #14b8a6);
            border-color: #2dd4bf;
            color: #ffffff;
        }

        .stButton > button[kind="primary"]:hover:enabled {
            background: linear-gradient(135deg, #115e59, #0f766e);
            border-color: #5eead4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <section class="app-hero">
        <div class="eyebrow">{"Dark Operations Mode" if dark_theme else "Emergency Intelligence Dashboard"}</div>
        <h1>AI Disaster Damage Assessment & Rescue Planning</h1>
        <p>
            Upload paired satellite imagery, generate a four-class damage heatmap,
            and convert model output into a field-ready rescue priority plan.
        </p>
        <div class="hero-stats">
            <div class="hero-stat"><strong>4 Damage Classes</strong><span>No damage to destroyed</span></div>
            <div class="hero-stat"><strong>5 ML Algorithms</strong><span>Automatic training comparison</span></div>
            <div class="hero-stat"><strong>Rescue Priority</strong><span>Zone, score, timeline</span></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Model Configuration")
st.sidebar.caption("Select the U-Net weights used for heatmap generation.")
model_path = st.sidebar.text_input("U-Net Heatmap Weights Path", "disaster_unet_final.pth")

if "best_uploaded_model" not in st.session_state and os.path.exists("best_uploaded_class_model.pkl"):
    st.session_state["best_uploaded_model"] = joblib.load("best_uploaded_class_model.pkl")

@st.cache_resource
def load_trained_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DisasterUNet(in_channels=6, out_classes=5).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device


def build_color_mask(pred_mask):
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, rgb in COLOR_MAP.items():
        color_mask[pred_mask == class_idx] = rgb
    return color_mask


def build_damage_overlay(post_img, pred_mask, alpha=0.45):
    color_mask = build_color_mask(pred_mask)
    color_mask_resized = cv2.resize(color_mask, post_img.size, interpolation=cv2.INTER_NEAREST)
    foreground_mask = cv2.resize(
        np.isin(pred_mask, DAMAGE_CLASS_IDS).astype(np.uint8),
        post_img.size,
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    post_np = np.array(post_img)
    overlay = post_np.copy()
    blended = cv2.addWeighted(post_np, 1 - alpha, color_mask_resized, alpha, 0)
    overlay[foreground_mask] = blended[foreground_mask]
    return overlay, color_mask_resized


def summarize_damage_classes(pred_mask):
    unique, counts = np.unique(pred_mask, return_counts=True)
    class_counts = {int(class_id): int(count) for class_id, count in zip(unique, counts)}
    damage_pixels = sum(class_counts.get(class_id, 0) for class_id in DAMAGE_CLASS_IDS)

    rows = []
    for class_id in DAMAGE_CLASS_IDS:
        pixels = class_counts.get(class_id, 0)
        rows.append(
            {
                "Class": CLASS_NAMES[class_id],
                "Pixels": pixels,
                "Share of Damage Area": 0.0 if damage_pixels == 0 else pixels / damage_pixels,
                "Share of Full Image": pixels / pred_mask.size,
            }
        )
    return class_counts, pd.DataFrame(rows)


def predict_rescue_priority(class_counts, total_pixels):
    no_damage_px = class_counts.get(1, 0)
    destroyed_px = class_counts.get(4, 0)
    major_px = class_counts.get(3, 0)
    minor_px = class_counts.get(2, 0)
    assessed_px = no_damage_px + minor_px + major_px + destroyed_px
    affected_px = minor_px + major_px + destroyed_px
    severe_px = major_px + destroyed_px

    affected_share = affected_px / assessed_px if assessed_px else 0.0
    severe_share = severe_px / assessed_px if assessed_px else 0.0
    destroyed_share = destroyed_px / assessed_px if assessed_px else 0.0
    affected_full_share = affected_px / total_pixels if total_pixels else 0.0
    response_score = min(
        100,
        int(round((destroyed_share * 100) + (severe_share * 70) + (affected_share * 25))),
    )

    base_plan = {
        "affected_pixels": affected_px,
        "severe_pixels": severe_px,
        "affected_share": affected_share,
        "severe_share": severe_share,
        "destroyed_share": destroyed_share,
        "affected_full_share": affected_full_share,
        "response_score": response_score,
        "verification": [
            "Confirm model output with visual review of the post-disaster image before dispatch decisions.",
            "Treat blocked roads, smoke, flooding, and unstable debris as field hazards until cleared.",
            "Prioritize occupied buildings, hospitals, schools, and dense residential clusters if local GIS data is available.",
        ],
    }

    if destroyed_px > 500 or major_px > 1500 or severe_share >= 0.25 or destroyed_share >= 0.10:
        base_plan.update(
            {
            "label": "Critical",
            "zone": "Zone Alpha",
            "phase": "Immediate life-safety response",
            "status": "Severe structural failure pattern detected with high collapse or entrapment risk.",
            "objective": "Locate survivors, isolate unsafe structures, and open emergency access routes.",
            "action": "Deploy Urban Search and Rescue immediately and establish command control at the safest accessible edge of the affected zone.",
            "resources": "Heavy rescue, technical search, trauma medical support, debris clearance, utility shutoff support, and perimeter security.",
            "severity": "error",
            "timeline": "0-2 hours",
            "deployment": [
                "Send advance reconnaissance to validate the heatmap and identify access routes.",
                "Assign rescue teams first to destroyed and major-damage clusters.",
                "Create hot, warm, and cold zones around unstable structures.",
                "Start triage and casualty collection outside the collapse perimeter.",
            ],
            "hazards": [
                "Secondary collapse",
                "Debris instability",
                "Gas/electrical utility failure",
                "Restricted vehicle access",
            ],
            "staging": "Use nearby no-damage or low-damage pixels as staging, triage, and supply drop areas after field confirmation.",
        }
        )
        return base_plan
    if major_px > 0 or minor_px > 2000 or affected_share >= 0.30:
        base_plan.update(
            {
            "label": "High",
            "zone": "Zone Bravo",
            "phase": "Rapid stabilization and targeted rescue",
            "status": "Significant structural damage detected, including possible partial collapse zones.",
            "objective": "Stabilize dangerous areas, search likely occupancy zones, and keep damaged corridors open.",
            "action": "Deploy structural assessment teams with rescue support and establish perimeter control around major-damage clusters.",
            "resources": "Paramedics, structural engineers, light rescue, debris-clearing crews, traffic control, and utility inspection.",
            "severity": "warning",
            "timeline": "2-6 hours",
            "deployment": [
                "Verify major-damage clusters first using the heatmap and post-disaster imagery.",
                "Route paramedics and light rescue teams to buildings marked major or destroyed.",
                "Set controlled access points to keep non-essential traffic away from damaged structures.",
                "Clear debris from primary routes before opening secondary roads.",
            ],
            "hazards": [
                "Partial collapse",
                "Falling facade or roofing material",
                "Damaged road access",
                "Delayed medical evacuation",
            ],
            "staging": "Place staging near no-damage clusters adjacent to the affected zone, with separate ingress and egress paths.",
        }
        )
        return base_plan
    if minor_px > 0:
        base_plan.update(
            {
            "label": "Moderate",
            "zone": "Zone Charlie",
            "phase": "Survey, welfare checks, and hazard monitoring",
            "status": "Minor structural damage detected with limited immediate collapse signal.",
            "objective": "Confirm habitability, identify isolated damage, and prevent escalation from secondary hazards.",
            "action": "Conduct rapid wellness checks and inspect minor-damage buildings for unsafe conditions.",
            "resources": "Standard emergency response units, building inspectors, basic medical support, and public works crews.",
            "severity": "info",
            "timeline": "6-24 hours",
            "deployment": [
                "Inspect minor-damage clusters and nearby roads.",
                "Check for vulnerable residents and blocked access points.",
                "Mark unsafe buildings for follow-up structural review.",
                "Monitor weather, flooding, fire, and aftershock conditions if applicable.",
            ],
            "hazards": [
                "Loose roofing or facade material",
                "Localized debris",
                "Access disruption",
                "Unreported interior damage",
            ],
            "staging": "Use no-damage areas for short-term welfare checks, supply handoff, and mobile medical support.",
        }
        )
        return base_plan
    base_plan.update(
        {
        "label": "Low / Safe Zone",
        "zone": "Zone Delta",
        "phase": "Monitoring and support staging",
        "status": "No structural damage detected in the analyzed building area.",
        "objective": "Maintain situational awareness and reserve the area for support operations if field checks agree.",
        "action": "Use the area for support operations after confirming access, utilities, and environmental safety.",
        "resources": "Routine patrol, communications support, logistics staff, and transport coordination.",
        "severity": "success",
    }
    )
    base_plan["timeline"] = "24 hours"
    base_plan["deployment"] = [
        "Confirm no-damage classification with visual review.",
        "Keep the area available for triage, supply distribution, or temporary command support.",
        "Continue monitoring if nearby tiles show moderate or severe damage.",
    ]
    base_plan["hazards"] = [
        "Hidden roof or interior damage",
        "Utility disruption",
        "Road access constraints",
    ]
    base_plan["staging"] = "Suitable candidate for triage centers, supply drop zones, or helicopter landing pads after field validation."
    return base_plan


def evaluate_segmentation_model(weights_path, img_dir, label_dir, max_samples, batch_size):
    dataset = xBDDataset(img_dir, label_dir)
    if len(dataset) == 0:
        raise ValueError("No pre-disaster images were found in the selected image folder.")

    sample_count = min(max_samples, len(dataset))
    subset = Subset(dataset, list(range(sample_count)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    model, device = load_trained_model(weights_path)
    y_true_parts = []
    y_pred_parts = []
    y_score_parts = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            y_true_parts.append(masks.cpu().numpy().reshape(-1))
            y_pred_parts.append(predictions.cpu().numpy().reshape(-1))
            y_score_parts.append(
                probabilities.permute(0, 2, 3, 1).cpu().numpy().reshape(-1, len(CLASS_NAMES))
            )

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    y_score = np.concatenate(y_score_parts)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_IDS, zero_division=0
    )
    metrics_df = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Pixels": support,
        }
    )

    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    summary = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Weighted Precision": weighted[0],
        "Weighted Recall": weighted[1],
        "Weighted F1": weighted[2],
        "Macro F1": macro[2],
        "Evaluated Images": sample_count,
        "Evaluated Pixels": int(y_true.size),
    }

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    roc_data = build_roc_data(y_true, y_score)
    return summary, metrics_df, cm, roc_data


def build_roc_data(y_true, y_score, max_points=200_000):
    if y_true.size > max_points:
        sampled_idx = np.linspace(0, y_true.size - 1, max_points, dtype=np.int64)
        y_true = y_true[sampled_idx]
        y_score = y_score[sampled_idx]

    y_true_bin = label_binarize(y_true, classes=CLASS_IDS)
    roc_data = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_truth = y_true_bin[:, class_idx]
        if len(np.unique(class_truth)) < 2:
            continue
        fpr, tpr, _ = roc_curve(class_truth, y_score[:, class_idx])
        roc_data.append(
            {
                "class_name": class_name,
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc(fpr, tpr),
            }
        )
    return roc_data


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="Actual class",
        xlabel="Predicted class",
        title="5-Class Damage Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    return fig


def plot_named_confusion_matrix(cm, class_names, title="Performance Matrix"):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual class",
        xlabel="Predicted class",
        title=title,
    )

    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    return fig


def plot_roc_curves(roc_data):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    for item in roc_data:
        ax.plot(item["fpr"], item["tpr"], linewidth=2, label=f"{item['class_name']} (AUC={item['auc']:.3f})")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_binary_roc_curve(fpr, tpr, roc_auc, positive_class_name):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.plot(fpr, tpr, linewidth=2.5, color="#2563eb", label=f"{positive_class_name} (AUC={roc_auc:.3f})")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def cross_validated_scores(model, X, y, cv):
    if hasattr(model, "predict_proba"):
        probabilities = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")
        return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        return cross_val_predict(clone(model), X, y, cv=cv, method="decision_function")
    return None


def plot_algorithm_comparison(results_df):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    chart_df = results_df.sort_values("F1 Score", ascending=True)
    y_pos = np.arange(len(chart_df))

    ax.barh(y_pos - 0.18, chart_df["Accuracy"], height=0.18, label="Accuracy", color="#2563eb")
    ax.barh(y_pos, chart_df["Precision"], height=0.18, label="Precision", color="#16a34a")
    ax.barh(y_pos + 0.18, chart_df["F1 Score"], height=0.18, label="F1 Score", color="#dc2626")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(chart_df["Algorithm"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    title = "Top 5 Algorithm Comparison" if len(chart_df) == 5 else "Algorithm Performance"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def extract_single_image_features(image_file):
    if hasattr(image_file, "seek"):
        image_file.seek(0)
    image = Image.open(image_file).convert("RGB")
    return extract_pil_image_features(image)


def extract_pil_image_features(image):
    image = image.convert("RGB").resize((256, 256))
    image_np = np.asarray(image).astype(np.float32)
    gray = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return {
        "red_mean": float(np.mean(image_np[:, :, 0])),
        "green_mean": float(np.mean(image_np[:, :, 1])),
        "blue_mean": float(np.mean(image_np[:, :, 2])),
        "red_std": float(np.std(image_np[:, :, 0])),
        "green_std": float(np.std(image_np[:, :, 1])),
        "blue_std": float(np.std(image_np[:, :, 2])),
        "brightness_mean": float(np.mean(gray)),
        "brightness_std": float(np.std(gray)),
        "edge_density": float(np.mean(edges > 0)),
        "texture_laplacian": float(np.var(laplacian)),
    }


def build_uploaded_two_class_dataset(class_a_files, class_b_files, class_a_name, class_b_name):
    rows = []
    for image_file in class_a_files:
        row = extract_single_image_features(image_file)
        row["target"] = 0
        row["class_name"] = class_a_name
        rows.append(row)

    for image_file in class_b_files:
        row = extract_single_image_features(image_file)
        row["target"] = 1
        row["class_name"] = class_b_name
        rows.append(row)

    return pd.DataFrame(rows)


def predict_uploaded_model(model_package, image):
    features = pd.DataFrame([extract_pil_image_features(image)])[model_package["feature_columns"]]
    prediction = int(model_package["model"].predict(features)[0])
    class_names = model_package["class_names"]
    confidence = None

    model = model_package["model"]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])
    elif hasattr(model, "decision_function"):
        score = model.decision_function(features)
        score_value = float(np.ravel(score)[0])
        confidence = float(1.0 / (1.0 + np.exp(-abs(score_value))))

    return {
        "class_id": prediction,
        "class_name": class_names[prediction],
        "confidence": confidence,
    }


@st.cache_resource
def load_classifier_package(model_file, algorithm_name, source_name, feature_mode):
    loaded_model = joblib.load(model_file)
    if isinstance(loaded_model, dict):
        package = loaded_model.copy()
        package.setdefault("algorithm", algorithm_name)
        package.setdefault("feature_columns", CUSTOM_FEATURE_COLUMNS)
        package.setdefault("class_names", ["Class 1", "Class 2"])
    else:
        package = {
            "algorithm": algorithm_name,
            "model": loaded_model,
            "feature_columns": PAIR_FEATURE_COLUMNS,
            "class_names": CLASS_NAMES,
        }

    package["model_file"] = model_file
    package["source_name"] = source_name
    package["feature_mode"] = feature_mode
    return package


def discover_classifier_options():
    options = []
    for algorithm_name, model_file in UPLOADED_MODEL_FILES.items():
        if os.path.exists(model_file):
            options.append(
                {
                    "label": f"{algorithm_name} - Uploaded image model",
                    "model_file": model_file,
                    "algorithm": algorithm_name,
                    "source_name": "Uploaded image model",
                    "feature_mode": "single_image",
                }
            )

    for algorithm_name, model_file in BASELINE_MODEL_FILES.items():
        if os.path.exists(model_file):
            options.append(
                {
                    "label": f"{algorithm_name} - Baseline pair model",
                    "model_file": model_file,
                    "algorithm": algorithm_name,
                    "source_name": "Baseline pair model",
                    "feature_mode": "pair_image",
                }
            )

    if os.path.exists("best_uploaded_class_model.pkl"):
        options.insert(
            0,
            {
                "label": "Best uploaded model - Auto selected",
                "model_file": "best_uploaded_class_model.pkl",
                "algorithm": "Best uploaded model",
                "source_name": "Auto selected",
                "feature_mode": "single_image",
            },
        )

    if os.path.exists("best_baseline_model.pkl"):
        options.append(
            {
                "label": "Best baseline model - Auto selected",
                "model_file": "best_baseline_model.pkl",
                "algorithm": "Best baseline model",
                "source_name": "Auto selected",
                "feature_mode": "pair_image",
            }
        )

    return options


def default_classifier_index(options):
    for index, option in enumerate(options):
        if option["algorithm"] == "Random Forest" and option["feature_mode"] == "single_image":
            return index
    for index, option in enumerate(options):
        if option["algorithm"] == "Random Forest":
            return index
    return 0


def predict_pair_model(model_package, pre_img, post_img):
    features = pd.DataFrame([extract_pair_features(pre_img, post_img)])[model_package["feature_columns"]]
    prediction = int(model_package["model"].predict(features)[0])
    class_names = model_package["class_names"]
    confidence = None

    model = model_package["model"]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        model_classes = list(getattr(model, "classes_", range(len(probabilities))))
        if prediction in model_classes:
            confidence = float(probabilities[model_classes.index(prediction)])
    elif hasattr(model, "decision_function"):
        score = model.decision_function(features)
        score_value = float(np.max(np.ravel(score)))
        confidence = float(1.0 / (1.0 + np.exp(-abs(score_value))))

    class_name = class_names[prediction] if prediction < len(class_names) else str(prediction)
    return {
        "class_id": prediction,
        "class_name": class_name,
        "confidence": confidence,
    }


def train_uploaded_two_class_models(class_a_files, class_b_files, class_a_name, class_b_name):
    dataset = build_uploaded_two_class_dataset(class_a_files, class_b_files, class_a_name, class_b_name)
    if len(class_a_files) < 2 or len(class_b_files) < 2:
        raise ValueError("Upload at least 2 images for each class so cross-validation can test both classes.")

    X = dataset[CUSTOM_FEATURE_COLUMNS]
    y = dataset["target"].astype(int)
    min_class_count = int(y.value_counts().min())
    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    available_algorithms = create_algorithms()

    rows = []
    artifacts = {}
    final_packages = {}
    for algorithm_name, model in available_algorithms.items():
        predictions = cross_val_predict(clone(model), X, y, cv=cv)
        score_values = cross_validated_scores(model, X, y, cv)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predictions, average="weighted", zero_division=0
        )
        accuracy = accuracy_score(y, predictions)
        correct_predictions = int(np.sum(y.to_numpy() == predictions))
        roc_auc = np.nan
        if score_values is not None and len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y, score_values)

        final_model = clone(model)
        final_model.fit(X, y)
        model_package = {
            "algorithm": algorithm_name,
            "model": final_model,
            "feature_columns": CUSTOM_FEATURE_COLUMNS,
            "class_names": [class_a_name, class_b_name],
            "validation_method": f"{n_splits}-fold stratified cross-validation",
            "accuracy": accuracy,
        }

        model_file = f"uploaded_{algorithm_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model.pkl"
        joblib.dump(model_package, model_file)
        final_packages[algorithm_name] = model_package
        rows.append(
            {
                "Algorithm": algorithm_name,
                "Accuracy": accuracy,
                "Exact Accuracy %": accuracy * 100,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC AUC": roc_auc,
                "Correct Predictions": correct_predictions,
                "Total Samples": int(len(y)),
                "Validation Method": f"{n_splits}-fold stratified CV",
                "Model File": model_file,
                "Training Used For Final Model": int(len(X)),
            }
        )
        artifacts[algorithm_name] = {
            "y_true": y.to_numpy(),
            "y_pred": predictions,
            "scores": score_values,
            "class_names": [class_a_name, class_b_name],
            "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]),
        }

    results = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    results.to_csv("uploaded_two_class_algorithm_comparison.csv", index=False)
    best_algorithm = results.iloc[0]["Algorithm"]
    best_package = final_packages[best_algorithm]
    best_package["model_file"] = "best_uploaded_class_model.pkl"
    joblib.dump(best_package, best_package["model_file"])
    return results, artifacts[best_algorithm], best_package


assessment_tab, training_performance_tab = st.tabs(["Assessment Console", "Training & Performance"])

with assessment_tab:
    classifier_options = discover_classifier_options()
    selected_classifier_package = None

    if classifier_options:
        selected_label = st.selectbox(
            "Classification Algorithm",
            [option["label"] for option in classifier_options],
            index=default_classifier_index(classifier_options),
            help=(
                "Choose an individual trained algorithm. Uploaded image models classify each image; "
                "baseline pair models classify the pre/post pair."
            ),
        )
        selected_option = classifier_options[[option["label"] for option in classifier_options].index(selected_label)]
        selected_classifier_package = load_classifier_package(
            selected_option["model_file"],
            selected_option["algorithm"],
            selected_option["source_name"],
            selected_option["feature_mode"],
        )
        st.success(
            f"Classification will use {selected_classifier_package['algorithm']} "
            f"from {selected_classifier_package['model_file']}."
        )
    else:
        st.info("No saved classifier models were found. The U-Net heatmap can still run if weights are available.")

    st.markdown("### Upload Satellite Pair")
    st.markdown(
        '<p class="section-lead">Provide matching pre- and post-disaster images. '
        "The assessment button activates after both files are ready.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pre-Disaster Satellite Image")
        pre_file = st.file_uploader("Upload Pre-Disaster Image (.png)", type=["png", "jpg", "jpeg"], key="pre")

    with col2:
        st.subheader("Post-Disaster Satellite Image")
        post_file = st.file_uploader("Upload Post-Disaster Image (.png)", type=["png", "jpg", "jpeg"], key="post")

    assessment_ready = bool(pre_file and post_file)

    if not assessment_ready:
        st.button(
            "Run Damage Assessment",
            type="primary",
            disabled=True,
            help="Upload both pre- and post-disaster images to enable assessment.",
            use_container_width=True,
        )
        st.caption("Waiting for both satellite images before running the damage model.")

    if assessment_ready:
        pre_img = Image.open(pre_file).convert("RGB")
        post_img = Image.open(post_file).convert("RGB")

        st.markdown("### Image Review")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(pre_img, caption="Pre-Disaster View", width="stretch")
        with col_img2:
            st.image(post_img, caption="Post-Disaster View", width="stretch")

        if st.button("Run Damage Assessment", type="primary", use_container_width=True):
            with st.spinner("Analyzing spatial geometry and generating damage heatmap..."):
                if selected_classifier_package:
                    st.subheader("Selected Algorithm Classification")

                    if selected_classifier_package["feature_mode"] == "single_image":
                        pre_result = predict_uploaded_model(selected_classifier_package, pre_img)
                        post_result = predict_uploaded_model(selected_classifier_package, post_img)

                        result_cols = st.columns(3)
                        result_cols[0].metric("Algorithm", selected_classifier_package["algorithm"])
                        result_cols[1].metric("Pre Image Class", pre_result["class_name"])
                        result_cols[2].metric("Post Image Class", post_result["class_name"])

                        confidence_cols = st.columns(2)
                        pre_conf = "N/A" if pre_result["confidence"] is None else f"{pre_result['confidence'] * 100:.2f}%"
                        post_conf = "N/A" if post_result["confidence"] is None else f"{post_result['confidence'] * 100:.2f}%"
                        confidence_cols[0].metric("Pre Confidence", pre_conf)
                        confidence_cols[1].metric("Post Confidence", post_conf)

                        if post_result["class_id"] != pre_result["class_id"]:
                            st.warning(
                                f"The selected algorithm detected a class change from "
                                f"{pre_result['class_name']} to {post_result['class_name']}."
                            )
                        elif post_result["class_id"] == 1:
                            st.error(f"The post-disaster image is classified as {post_result['class_name']}.")
                        else:
                            st.success(f"The post-disaster image is classified as {post_result['class_name']}.")
                    else:
                        pair_result = predict_pair_model(selected_classifier_package, pre_img, post_img)
                        pair_conf = "N/A" if pair_result["confidence"] is None else f"{pair_result['confidence'] * 100:.2f}%"

                        result_cols = st.columns(4)
                        result_cols[0].metric("Algorithm", selected_classifier_package["algorithm"])
                        result_cols[1].metric("Model Type", "Pre/Post Pair")
                        result_cols[2].metric("Predicted Damage", pair_result["class_name"])
                        result_cols[3].metric("Confidence", pair_conf)

                        if pair_result["class_id"] in [3, 4]:
                            st.error(f"The selected baseline model predicts {pair_result['class_name']} damage.")
                        elif pair_result["class_id"] == 2:
                            st.warning(f"The selected baseline model predicts {pair_result['class_name']} damage.")
                        else:
                            st.success(f"The selected baseline model predicts {pair_result['class_name']}.")

                    st.caption(
                        "The dropdown controls the classifier used above. "
                        "The 4-class heatmap and rescue prediction below still use the U-Net segmentation weights path."
                    )

                inf_transform = transforms.Compose(
                    [
                        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

                tensor_pre = inf_transform(pre_img)
                tensor_post = inf_transform(post_img)
                input_tensor = torch.cat((tensor_pre, tensor_post), dim=0).unsqueeze(0)

                model, device = load_trained_model(model_path)
                input_tensor = input_tensor.to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                overlay, color_mask_resized = build_damage_overlay(post_img, pred_mask)
                class_counts, damage_df = summarize_damage_classes(pred_mask)
                rescue_prediction = predict_rescue_priority(class_counts, pred_mask.size)
                dominant_damage_row = damage_df.sort_values("Pixels", ascending=False).iloc[0]

                st.success("Analysis Complete!")
                st.subheader("Four-Class Damage Heatmap")
                st.markdown(
                    """
                    <div class="legend-row">
                        <span class="legend-pill"><span class="legend-dot" style="background:#00c853;"></span>No Damage</span>
                        <span class="legend-pill"><span class="legend-dot" style="background:#ffd600;"></span>Minor Damage</span>
                        <span class="legend-pill"><span class="legend-dot" style="background:#ff8f00;"></span>Major Damage</span>
                        <span class="legend-pill"><span class="legend-dot" style="background:#e53935;"></span>Destroyed</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                heatmap_col, mask_col = st.columns(2)
                with heatmap_col:
                    st.image(
                        overlay,
                        caption="Post-disaster image with 4-class damage heatmap overlay",
                        width="stretch",
                    )
                with mask_col:
                    st.image(
                        color_mask_resized,
                        caption="Predicted 4-class damage map",
                        width="stretch",
                    )

                st.markdown("### Damage Severity Breakdown")
                col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
                col_m1.metric("No Damage", f"{class_counts.get(1, 0):,} px")
                col_m2.metric("Minor Damage", f"{class_counts.get(2, 0):,} px")
                col_m3.metric("Major Damage", f"{class_counts.get(3, 0):,} px")
                col_m4.metric("Destroyed", f"{class_counts.get(4, 0):,} px")
                col_m5.metric("Current Damage Class", dominant_damage_row["Class"])
                col_m6.metric("Rescue Prediction", rescue_prediction["label"])

                st.dataframe(
                    damage_df.style.format(
                        {
                            "Share of Damage Area": "{:.2%}",
                            "Share of Full Image": "{:.2%}",
                        }
                    ),
                    width="stretch",
                )

                st.markdown("### Rescue Team Action Plan & Priorities")
                response_cols = st.columns(4)
                response_cols[0].metric("Priority Zone", rescue_prediction["zone"])
                response_cols[1].metric("Operational Phase", rescue_prediction["phase"])
                response_cols[2].metric("Response Score", f"{rescue_prediction['response_score']}/100")
                response_cols[3].metric("Target Timeline", rescue_prediction["timeline"])
                st.progress(rescue_prediction["response_score"] / 100)

                rescue_message = (
                    f"**{rescue_prediction['label'].upper()} PRIORITY ({rescue_prediction['zone']})**\n\n"
                    f"**Current Assessment:** {rescue_prediction['status']}\n\n"
                    f"**Incident Objective:** {rescue_prediction['objective']}\n\n"
                    f"**Primary Action:** {rescue_prediction['action']}\n\n"
                    f"**Recommended Resources:** {rescue_prediction['resources']}"
                )
                if rescue_prediction["severity"] == "error":
                    st.error(rescue_message)
                elif rescue_prediction["severity"] == "warning":
                    st.warning(rescue_message)
                elif rescue_prediction["severity"] == "info":
                    st.info(rescue_message)
                else:
                    st.success(rescue_message)

                evidence_cols = st.columns(3)
                evidence_cols[0].metric(
                    "Affected Damage Area",
                    f"{rescue_prediction['affected_share']:.2%}",
                    help="Minor, major, and destroyed pixels divided by all detected building/damage-class pixels.",
                )
                evidence_cols[1].metric(
                    "Severe Damage Area",
                    f"{rescue_prediction['severe_share']:.2%}",
                    help="Major and destroyed pixels divided by all detected building/damage-class pixels.",
                )
                evidence_cols[2].metric(
                    "Destroyed Area",
                    f"{rescue_prediction['destroyed_share']:.2%}",
                    help="Destroyed pixels divided by all detected building/damage-class pixels.",
                )

                plan_col1, plan_col2 = st.columns(2)
                with plan_col1:
                    st.markdown("#### Deployment Sequence")
                    for step in rescue_prediction["deployment"]:
                        st.markdown(f"- {step}")
                    st.markdown("#### Staging Guidance")
                    st.write(rescue_prediction["staging"])
                with plan_col2:
                    st.markdown("#### Field Hazards To Check")
                    for hazard in rescue_prediction["hazards"]:
                        st.markdown(f"- {hazard}")
                    st.markdown("#### Verification Notes")
                    for note in rescue_prediction["verification"]:
                        st.markdown(f"- {note}")

                st.caption(
                    "Rescue priority is a decision-support estimate from the segmentation heatmap. "
                    "Final dispatch should be confirmed with field reports, local occupancy data, and command judgment."
                )

with training_performance_tab:
    st.subheader("5-Class Damage Model Performance")
    st.markdown(
        '<p class="section-lead">Evaluate the main U-Net damage model using the project damage classes, '
        "not pre/post image labels. The confusion matrix compares actual and predicted pixel classes.</p>",
        unsafe_allow_html=True,
    )

    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        eval_img_dir = st.text_input("Evaluation Image Folder", "dataset/train/train/images")
        eval_max_samples = st.number_input("Evaluation Image Pairs", min_value=1, max_value=500, value=25, step=5)
    with eval_col2:
        eval_label_dir = st.text_input("Evaluation Label Folder", "dataset/train/train/labels")
        eval_batch_size = st.number_input("Batch Size", min_value=1, max_value=16, value=2, step=1)

    if st.button("Evaluate 5-Class Damage Model", type="primary", use_container_width=True):
        with st.spinner("Evaluating U-Net segmentation model on 5 damage classes..."):
            try:
                summary, metrics_df, cm, roc_data = evaluate_segmentation_model(
                    model_path,
                    eval_img_dir,
                    eval_label_dir,
                    int(eval_max_samples),
                    int(eval_batch_size),
                )
            except Exception as exc:
                st.error(f"Could not evaluate 5-class model: {exc}")
            else:
                metric_cols = st.columns(5)
                metric_cols[0].metric("Accuracy", f"{summary['Accuracy'] * 100:.2f}%")
                metric_cols[1].metric("Weighted F1", f"{summary['Weighted F1']:.4f}")
                metric_cols[2].metric("Macro F1", f"{summary['Macro F1']:.4f}")
                metric_cols[3].metric("Images", f"{summary['Evaluated Images']:,}")
                metric_cols[4].metric("Pixels", f"{summary['Evaluated Pixels']:,}")

                st.dataframe(
                    metrics_df.style.format(
                        {
                            "Precision": "{:.4f}",
                            "Recall": "{:.4f}",
                            "F1 Score": "{:.4f}",
                            "Pixels": "{:,}",
                        }
                    ),
                    width="stretch",
                )

                matrix_col, roc_col = st.columns(2)
                with matrix_col:
                    st.pyplot(plot_confusion_matrix(cm), width="stretch")
                with roc_col:
                    if roc_data:
                        st.pyplot(plot_roc_curves(roc_data), width="stretch")
                    else:
                        st.warning("ROC curve is not available for this evaluation sample.")

                st.caption(
                    "The main project confusion matrix uses the classes: Background, No Damage, "
                    "Minor Damage, Major Damage, and Destroyed. Pre-disaster and post-disaster are inputs only."
                )

    st.divider()

    st.subheader("Optional: Train Model With Two Uploaded Classes")
    st.markdown(
        '<p class="section-lead">Upload two labeled image groups to train every available algorithm, '
        "compare validation metrics, and save the strongest classifier for the assessment console. "
        "This section is only for custom binary experiments.</p>",
        unsafe_allow_html=True,
    )

    name_col1, name_col2 = st.columns(2)
    with name_col1:
        class_a_name = st.text_input("Class 1 Name", "Class 1")
    with name_col2:
        class_b_name = st.text_input("Class 2 Name", "Class 2")
    st.caption("All five algorithms will be trained, cross-validated, saved, and compared from these uploaded images.")

    upload_col1, upload_col2 = st.columns(2)
    with upload_col1:
        class_a_files = st.file_uploader(
            f"Drag and drop or upload {class_a_name} images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="class_a_upload",
            help="Drop multiple images here or click Upload to browse.",
        )
    with upload_col2:
        class_b_files = st.file_uploader(
            f"Drag and drop or upload {class_b_name} images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="class_b_upload",
            help="Drop multiple images here or click Upload to browse.",
        )

    training_ready = bool(class_a_files and class_b_files)
    if st.button(
        "Train and Compare All 5 Algorithms",
        type="primary",
        disabled=not training_ready,
        help="Upload images for both classes to enable training.",
        use_container_width=True,
    ):
        if not training_ready:
            st.error("Upload images for both classes before training.")
        else:
            with st.spinner("Training and comparing all five algorithms from the uploaded classes..."):
                try:
                    uploaded_results, uploaded_artifacts, best_model_package = train_uploaded_two_class_models(
                        class_a_files,
                        class_b_files,
                        class_a_name,
                        class_b_name,
                    )
                except Exception as exc:
                    st.error(f"Could not train model: {exc}")
                else:
                    st.session_state["best_uploaded_model"] = best_model_package
                    load_classifier_package.clear()
                    best = uploaded_results.iloc[0]
                    st.success(
                        f"Training complete. Best model: {best['Algorithm']} "
                        f"with exact CV accuracy {best['Exact Accuracy %']:.2f}% "
                        f"({int(best['Correct Predictions'])}/{int(best['Total Samples'])})."
                    )
                    st.dataframe(
                        uploaded_results.style.format(
                            {
                                "Accuracy": "{:.6f}",
                                "Exact Accuracy %": "{:.2f}",
                                "Precision": "{:.6f}",
                                "Recall": "{:.6f}",
                                "F1 Score": "{:.6f}",
                                "ROC AUC": "{:.6f}",
                            }
                        ),
                        width="stretch",
                    )
                    st.markdown("#### Algorithm Characteristics")
                    shown_algorithms = uploaded_results["Algorithm"].tolist()
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Algorithm": shown_algorithms,
                                "Characteristic": [ALGORITHM_NOTES[name] for name in shown_algorithms],
                            }
                        ),
                        width="stretch",
                    )
                    st.pyplot(plot_algorithm_comparison(uploaded_results), width="stretch")
                    st.markdown("#### Automatic Performance Matrix and ROC Curve")
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.pyplot(
                            plot_named_confusion_matrix(
                                uploaded_artifacts["confusion_matrix"],
                                uploaded_artifacts["class_names"],
                                title=f"{best['Algorithm']} Performance Matrix",
                            ),
                            width="stretch",
                        )
                    with chart_col2:
                        if uploaded_artifacts["scores"] is None:
                            st.warning("ROC curve is not available for this trained model.")
                        else:
                            fpr, tpr, _ = roc_curve(uploaded_artifacts["y_true"], uploaded_artifacts["scores"])
                            roc_auc = auc(fpr, tpr)
                            st.pyplot(
                                plot_binary_roc_curve(fpr, tpr, roc_auc, uploaded_artifacts["class_names"][1]),
                                width="stretch",
                            )
                    st.caption(
                        "Accuracy is calculated by stratified cross-validation: each uploaded image is tested once. "
                        "The saved model files are then retrained on all uploaded images."
                    )
    elif not training_ready:
        st.caption("Training is locked until both image groups contain files.")
