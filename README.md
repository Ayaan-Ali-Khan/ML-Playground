# Machine Learning Playground

Interactive Streamlit app for experimenting with classification datasets and models, tuning hyperparameters, and visualizing results with rich diagnostics.

## Features

- Synthetic and real datasets with configurable controls
- Multiple sklearn models with dynamic hyperparameter UI
- Train/test metrics, confusion matrix, ROC and PR curves
- Decision boundary visualization with optional probability shading
- Learning curve and validation curve analysis
- Reproducible export code for selected setup

## Project Structure

- `app.py` - Streamlit entrypoint and global UI style
- `pages/` - Home, dataset selection, and model training pages
- `datasets/` - Dataset loaders and metadata
- `models/` - Model registry, builder, and evaluation engine
- `utils/` - Plotting, analysis helpers, and export generation

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open the app in your browser (typically `http://localhost:8501`).

## How To Use

1. Go to the **Dataset** page and choose synthetic or real data.
2. Configure split ratio and random seed.
3. Go to **Train Model**, select model + hyperparameters, and click **Train Model**.
4. Inspect metrics, plots, and model insights.
5. Use **Export Code** to generate a reproducible script.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set `app.py` as the entrypoint.
4. Deploy using `requirements.txt` for dependency installation.

## License

MIT License. See `LICENSE` for details.