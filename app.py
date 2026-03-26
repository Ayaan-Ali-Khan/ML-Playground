import streamlit as st

st.set_page_config(
    page_title="ML Playground",
    page_icon="🧪",
    layout="wide",
)

pages = st.navigation({
    "🧪 Playground": [
        st.Page("pages/dataset.py", title="Dataset", icon="📊"),
        st.Page("pages/model.py", title="Train Model", icon="🤖"),
        # st.Page("pages/3_Visualize.py", title="Decision Boundary", icon="🗺️"),
        # st.Page("pages/4_Insights.py", title="Model Insights", icon="🔍"),
        # st.Page("pages/5_Export.py", title="Export Code", icon="📋"),
    ],
    "ℹ️ Info": [
        # st.Page("pages/0_Home.py", title="Home", icon="🏠"),
    ],
})

pages.run()