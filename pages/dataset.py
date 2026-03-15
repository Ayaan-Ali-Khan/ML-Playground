"""
Phase 2 — Dataset Module
Streamlit page: dataset selection, controls, and preview panel.
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from datasets.synthetic import get_synthetic_data, SYNTHETIC_DATASETS
from datasets.real import get_real_dataset, get_feature_names, REAL_DATASETS
from utils.plot_utils import plot_dataset

#------PAGE CONFIG------#
st.set_page_config(
    page_title="Dataset • ML Playground",
    page_icon="./assets/flask.png",
    layout="wide"
)

st.title("📊 Dataset Module")
st.caption("Select, configure and preview your dataset before training.")

#------SIDEBAR------#
with st.sidebar:
    st.header("Dataset Controls")
    source = st.selectbox(
        "Data Source",
        options=["Synthetic", "Real (sklearn)"],
        help="Choose where your data comes from."
    )
    st.divider()

    #---Shared Controls---#
    if source in ["Synthetic", "Real (sklearn)"]:
        test_split = st.slider(
            "Test Split Ratio",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Subset of data held out as the test set."
        )
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            step=1,
            help="Controls dataset generation and train/test split reproducibility."
        )
    
    #---Synthetic Settings---#
    if source == "Synthetic":
        st.subheader("Synthetic Data Settings")
        dataset_key = st.selectbox(
            "Dataset Type",
            options=list(SYNTHETIC_DATASETS.keys()),
            format_func=lambda k: SYNTHETIC_DATASETS[k]["label"]
        )
        st.caption(SYNTHETIC_DATASETS[dataset_key]["description"])

        n_samples = st.slider(
            "Number of samples",
            min_value=50,
            max_value=2000,
            value=300,
            step=50,
            help="Total samples generated (split into train + test)."
        )
        noise = st.slider(
            "Noise Level",
            min_value=0.00,
            max_value=0.50,
            value=0.10,
            step=0.01,
            help="Standard deviation of Gaussian noise added to features."
        )
    
    #---Real Settings---#
    elif source == "Real (sklearn)":
        st.subheader("Real Dataset Settings")
 
        dataset_key = st.selectbox(
            "Dataset",
            options=list(REAL_DATASETS.keys()),
            format_func=lambda k: REAL_DATASETS[k]["label"],
        )
        st.caption(REAL_DATASETS[dataset_key]["description"])
 
        all_features = get_feature_names(dataset_key)
        st.info(
            f"**{len(all_features)} features** available. "
            "Pick 2 for 2D visualization (or use all for training)."
        )
 
        use_2d = st.toggle(
            "2D Visualization Mode",
            value=True,
            help="Select 2 features to plot the decision boundary later.",
        )
 
        if use_2d:
            feat_x = st.selectbox(
                "X-axis Feature",
                options=range(len(all_features)),
                format_func=lambda i: all_features[i],
                index=0,
            )
            feat_y = st.selectbox(
                "Y-axis Feature",
                options=range(len(all_features)),
                format_func=lambda i: all_features[i],
                index=1,
            )
            feature_indices = (feat_x, feat_y)
        else:
            feature_indices = None

#------Load data------#
X = y = feature_names = class_names = None

if source == "Synthetic":
    X, y = get_synthetic_data(dataset_type=dataset_key, n_samples=n_samples,noise=noise, random_seed=random_seed)
    feature_names = ["Feature 0", "Feature 1"]
    n_classes = SYNTHETIC_DATASETS[dataset_key]["n_classes"]
    class_names = [f"Class {i}" for i in range(n_classes)]

elif source == "Real (sklearn)":
    X, y, feature_names, class_names = get_real_dataset(
        dataset_name=dataset_key,
        feature_indices=feature_indices if use_2d else None
    )

#------Data splitting------#
if X is not None and y is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=random_seed, stratify=y)

    # Store in session state for other pages (model training etc.)
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["feature_names"] = feature_names
    st.session_state["class_names"] = class_names
    st.session_state["dataset_ready"] = True

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Samples", X.shape[0])
    col2.metric("Features", X.shape[1])
    col3.metric("Train Samples", X_train.shape[0])
    col4.metric("Test Samples", X_test.shape[0])
    st.divider()

    #------Preview plot------
    preview_feature_names = feature_names[:2] if len(feature_names) >= 2 else feature_names
 
    fig = plot_dataset(
        X=X[:, :2],   # always use first 2 features for scatter
        y=y,
        feature_names=preview_feature_names,
        class_names=class_names,
        title=f"Dataset Preview",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 View raw data (first 50 rows)"):
        df_preview = pd.DataFrame(X, columns=feature_names)
        df_preview["target"] = y
        df_preview["class"] = [
            class_names[i] if i < len(class_names) else str(i) for i in y
        ]
        st.dataframe(df_preview.head(50), use_container_width=True)

else:
    st.info("Configure the controls in the sidebar and your dataset will appear here.")