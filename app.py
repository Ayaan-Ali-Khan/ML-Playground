import streamlit as st

# Configure the page setting (This might already be in .streamlit/config.toml, but acts as a fallback/explicit setting)
st.set_page_config(page_title="ML Playground", layout="wide", page_icon="🧠")

def main():
    # --- UI Wireframe ---

    # 1. Sidebar Controls
    st.sidebar.title("⚙️ Controls")
    st.sidebar.markdown("---")
    
    # Dataset Selection Placeholder
    st.sidebar.subheader("1. Dataset")
    dataset_name = st.sidebar.selectbox("Select Dataset", ["Moons", "Circles", "Blobs", "Classification"])
    
    # Model Selection Placeholder
    st.sidebar.subheader("2. Model")
    model_name = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
    
    # Hyperparameters Placeholder
    st.sidebar.subheader("3. Hyperparameters")
    st.sidebar.slider("Sample Slider (e.g. C)", 0.01, 10.0, 1.0)
    
    st.sidebar.markdown("---")
    st.sidebar.button("Train Model", use_container_width=True, type="primary")

    # 2. Main Header & Description
    st.title("🧠 Classical ML Playground")
    st.markdown("Explore, tune, and visualize classical machine learning algorithms directly from your browser.")
    st.markdown("---")

    # Metrics Panel Placeholder
    st.subheader("📊 Model Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy (Train)", "---")
    m2.metric("Accuracy (Test)", "---")
    m3.metric("F1 Score", "---")
    m4.metric("Train Time", "---")

    st.markdown("---")

    # 3. Main Canvas Split (Plots)
    st.subheader("📈 Decision Boundaries & Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Training Data")
        # Placeholder for Train Plotly Figure
        st.info("Train set decision boundary will appear here.")
        
    with col2:
        st.markdown("### Test Data")
        # Placeholder for Test Plotly Figure
        st.info("Test set decision boundary will appear here.")

if __name__ == "__main__":
    main()