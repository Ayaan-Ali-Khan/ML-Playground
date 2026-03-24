"""
Phase 3 + 4 + 5 — Model Training Page
Sidebar: model selector + dynamic hyperparameter widgets
Main:    metrics, confusion matrix, ROC curve
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as  make_subplots
import plotly.figure_factory as ff
import plotly.express as px
from models.registry import MODEL_REGISTRY, MODEL_GROUPS
from models.builder import build_model, get_default_params
from models.evaluator import train_and_evaluate, EvalResult

st.set_page_config(
    page_title="Train Model • ML Playground",
    page_icon="./assets/flask.png",
    layout="wide"
)

st.title("Model Training")
st.caption("Select a model, tune its hyperparameters, and evaluate performance.")

if not st.session_state.get("dataset_ready"):
    st.warning("⚠️ No dataset loaded. Please visit the **Dataset** page first.")
    st.stop()

X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]
feature_names = st.session_state["feature_names"]
class_names = st.session_state["class_names"]

#------Sidebar------#
with st.sidebar:
    st.header("Model Controls")

    all_model_keys = [key for key in MODEL_REGISTRY.keys()]
    def model_format(key):
        label = MODEL_REGISTRY[key]["label"]
        return label

    model_key = st.selectbox(
        "Model",
        options=all_model_keys,
        format_func=model_format,
        help="Select a classification algorithm to train."
    )

    entry = MODEL_REGISTRY[model_key]
    st.caption(entry["description"])
    st.markdown(f"[[sklearn docs]]({entry['docs_url']})", unsafe_allow_html=False)
    st.divider()

    #------Dynamic hyperparameter widgets------#
    st.subheader("⚙️ Hyperparameters")
    if st.button("Reset to defaults", width="stretch"):
        st.session_state[f"_reset_{model_key}"] = True
        st.rerun()
    # If reset flag is set, overwrite widget keys with defaults BEFORE widgets render
    if st.session_state.pop(f"_reset_{model_key}", False):
        for pk, spec in entry["params"].items():
            wkey  = f"{model_key}_{pk}"
            ptype = spec["type"]
            if ptype == "slider_float":
                st.session_state[wkey] = float(spec["default"])
            elif ptype == "slider_int":
                st.session_state[wkey] = int(spec["default"])
            else:
                st.session_state[wkey] = spec["default"]

    user_params = {}

    for param_key, spec in entry["params"].items():
        ptype = spec["type"]
        label = spec["label"]
        help = spec.get("help", "")
        wkey = f"{model_key}_{param_key}"
        if wkey in st.session_state:
            default = st.session_state[wkey]
        else:
            default = spec["default"]

        if ptype == "slider_float":
            val = st.slider(
                label,
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                value=float(default),
                step=float(spec["step"]),
                help=help,
                key=wkey,
            )
        elif ptype == "slider_int":
            val = st.slider(
                label,
                min_value=int(spec["min"]),
                max_value=int(spec["max"]),
                value=int(default),
                step=int(spec["step"]),
                help=help,
                key=wkey,
            )
        elif ptype == "selectbox":
            options = spec["options"]
            idx = options.index(default) if default in options else 0
            val = st.selectbox(
                label,
                options=options,
                index=idx,
                help=help,
                key=wkey,
            )
        elif ptype == "toggle":
            val = st.toggle(
                label, 
                value=bool(default),
                help=help,
                key=wkey
            )
        else: val = default
        user_params[param_key] = val

#------Build and Train------#
try:
    clf = build_model(model_key, user_params)
except Exception as e:
    st.error(f"Could not build model: {e}")
    st.stop()

#---Train button---#
col_btn, col_status = st.columns([1, 3])
with col_btn:
    train_clicked = st.button("Train Model", type="primary", width="stretch")

params_fingerprint = str(model_key) + str(user_params)
if train_clicked or st.session_state.get("last_fingerprint") != params_fingerprint:
    if train_clicked:
        st.session_state["last_fingerprint"] = params_fingerprint
        with st.spinner("Training..."):
            result:EvalResult = train_and_evaluate(clf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, class_names=class_names)
        # Store trained model for visualization page
        st.session_state["trained_model"] = clf
        st.session_state["eval_result"] = result
        st.session_state["model_key"] = model_key
    elif "eval_result" not in st.session_state:
        with col_status:
            st.info("Configure hyperparameters and click **Train Model** to begin.")
        st.stop()

result = st.session_state.get("eval_result")
if result.error:
    st.error(f"Training Error: {result.error}")
    st.stop()


#------Metrics Row------#
st.subheader("📊 Evaluation Metrics")

col_train, col_test = st.columns(2)

with col_train:
    st.markdown("**Train Set**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{result.train_accuracy:.3f}")
    m2.metric("F1 Score", f"{result.train_f1:.3f}")
    m3.metric("Precision", f"{result.train_precision:.3f}")
    m4.metric("Recall", f"{result.train_recall:.3f}")

with col_test:
    st.markdown("**Test Set**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{result.test_accuracy:.3f}", delta=f"{result.test_accuracy - result.train_accuracy:+.3f}")
    m2.metric("F1 Score", f"{result.test_f1:.3f}", delta=f"{result.test_f1 - result.train_f1:+.3f}")
    m3.metric("Precision", f"{result.test_precision:.3f}", delta=f"{result.test_precision - result.train_precision:+.3f}")
    m4.metric("Recall", f"{result.test_recall:.3f}", delta=f"{result.test_recall - result.train_recall:+.3f}")

st.caption(f"⏱️ Training time: **{result.train_time_ms:.1f} ms**")
st.divider()

#------Confusion Matrix + ROC + PR curve------#
col_cm, col_roc = st.columns(2)

with col_cm:
    st.subheader("Confusion Matrix")
    cm = result.conf_matrix
    display_labels = class_names if class_names is not None else [f"Class{i}" for i in range(cm.shape[0])]
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=display_labels,
        y=display_labels,
        colorscale="Blues",
        showscale=True,
        reversescale=False,
    )
    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis_autorange="reversed",
        height=360,
        margin=dict(l=20, r=20, t=30, b=60),
        template="plotly_white",
    )
    st.plotly_chart(fig_cm, width="stretch")

with col_roc:
    st.subheader("ROC Curve")
    if result.roc_auc is not None and result.fpr is not None:
        fig_roc = go.Figure()
 
        # Diagonal baseline
        fig_roc.add_trace(
            go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            name="Random (AUC = 0.50)",
            showlegend=True,
        ))
 
        colors = px.colors.qualitative.Set2
 
        if isinstance(result.fpr, dict):
            # Multiclass: one curve per class
            for i, cls in enumerate(result.fpr.keys()):
                label = class_names[cls] if cls < len(class_names) else f"Class {cls}"
                fig_roc.add_trace(go.Scatter(
                    x=result.fpr[cls],
                    y=result.tpr[cls],
                    mode="lines",
                    name=label,
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
        else:
            # Binary
            fig_roc.add_trace(
                go.Scatter(
                x=result.fpr,
                y=result.tpr,
                mode="lines",
                name=f"ROC (AUC = {result.roc_auc:.3f})",
                line=dict(color=colors[0], width=2),
                fill="tozeroy",
                fillcolor="rgba(102,194,165,0.15)",
            ))
 
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=360,
            margin=dict(l=20, r=20, t=30, b=60),
            template="plotly_white",
            legend=dict(x=0.55, y=0.1),
        )
        if result.roc_auc:
            st.plotly_chart(fig_roc, width="stretch")
            st.caption(f"Weighted ROC-AUC: **{result.roc_auc:.4f}**")
    else:
        st.info("ROC curve not available for this model/dataset combination.")

st.subheader("Precision-Recall Curve")
 
if result.y_test_proba is not None:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    colors = px.colors.qualitative.Set2
    n_classes = len(np.unique(y_test))
 
    fig_pr = go.Figure()
 
    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(y_test, result.y_test_proba[:, 1])
        ap = average_precision_score(y_test, result.y_test_proba[:, 1])
        baseline = y_test.sum() / len(y_test)   # positive class ratio
 
        # Baseline (random classifier)
        fig_pr.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            name=f"Random (AP = {baseline:.2f})",
        ))
        fig_pr.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            name=f"PR curve (AP = {ap:.3f})",
            line=dict(color=colors[0], width=2),
            fill="tozeroy",
            fillcolor="rgba(102,194,165,0.15)",
        ))
        st.caption(f"Average Precision: **{ap:.4f}**")
 
    else:
        # One-vs-Rest per class
        classes = np.unique(y_train)
        y_bin = label_binarize(y_test, classes=classes)
        ap_scores = []
 
        for i, cls in enumerate(classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, i], result.y_test_proba[:, i])
            ap = average_precision_score(y_bin[:, i], result.y_test_proba[:, i])
            ap_scores.append(ap)
            label = class_names[cls] if cls < len(class_names) else f"Class {cls}"
 
            fig_pr.add_trace(go.Scatter(
                x=recall, y=precision,
                mode="lines",
                name=f"{label} (AP = {ap:.3f})",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
 
        mean_ap = np.mean(ap_scores)
        st.caption(f"Mean Average Precision (mAP): **{mean_ap:.4f}**")
 
    fig_pr.update_layout(
        xaxis=dict(title="Recall", range=[0, 1]),
        yaxis=dict(title="Precision", range=[0, 1.05]),
        height=380,
        margin=dict(l=20, r=20, t=20, b=60),
        template="plotly_white",
        legend=dict(x=0.01, y=0.01, xanchor="left", yanchor="bottom"),
    )
    st.plotly_chart(fig_pr, width="stretch")
 
else:
    st.info("Precision-Recall curve requires `predict_proba` — not available for this model.")

st.divider()

#------Feature Importance------#
st.subheader("🔍 Model Insights")
 
# Extract the actual estimator from Pipeline if wrapped
actual_clf = clf
if hasattr(clf, "named_steps"):
    actual_clf = clf.named_steps.get("clf", clf)
 
shown_insight = False
 
# Feature importance (tree-based models)
if entry.get("supports_feature_importance") and hasattr(actual_clf, "feature_importances_"):
    importances = actual_clf.feature_importances_
    feat_labels = feature_names if len(feature_names) == len(importances) else [f"F{i}" for i in range(len(importances))]
    
    sorted_idx = np.argsort(importances)[::-1]
    fig_fi = go.Figure(go.Bar(
        x=[feat_labels[i] for i in sorted_idx],
        y=importances[sorted_idx],
        marker_color="steelblue",
    ))
    fig_fi.update_layout(
        title="Feature Importances",
        xaxis_title="Feature",
        yaxis_title="Importance",
        height=300,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=60),
    )
    st.plotly_chart(fig_fi, width="stretch")
    shown_insight = True
 
# Coefficients (LR, LDA)
elif entry.get("supports_coef") and hasattr(actual_clf, "coef_"):
    coef = actual_clf.coef_
    feat_labels = feature_names if len(feature_names) == coef.shape[1] else [f"F{i}" for i in range(coef.shape[1])]
 
    if coef.shape[0] == 1:
        # Binary
        fig_coef = go.Figure(go.Bar(
            x=feat_labels,
            y=coef[0],
            marker_color=["steelblue" if v > 0 else "tomato" for v in coef[0]],
        ))
    else:
        # Multi-class: grouped bars
        import plotly.express as px
        colors = px.colors.qualitative.Set2
        fig_coef = go.Figure()
        for i, row in enumerate(coef):
            label = class_names[i] if i < len(class_names) else f"Class {i}"
            fig_coef.add_trace(go.Bar(
                x=feat_labels, y=row, name=label,
                marker_color=colors[i % len(colors)],
            ))
        fig_coef.update_layout(barmode="group")
 
    fig_coef.update_layout(
        title="Model Coefficients",
        xaxis_title="Feature",
        yaxis_title="Coefficient value",
        height=300,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=60),
    )
    st.plotly_chart(fig_coef, width="stretch")
    shown_insight = True
 
if not shown_insight:
    st.info("No feature importance or coefficient visualization available for this model.")