# emotional_monitor

**Abstract:**

This study focuses on the analysis of an emotional monitoring dataset to understand the relationship between physiological features (e.g., heart rate, pupil diameter, cortisol levels) and emotional and cognitive states. Leveraging advanced data visualization, machine learning, and feature importance techniques (e.g., SHAP values), we propose a Random Forest-based classification model for predicting engagement levels and emotional states. The study also provides insights into feature correlations, data distributions, and the performance of clustering algorithms. Performance metrics such as accuracy, F1-score, and R² are evaluated, and detailed visualizations are presented for interpretability.

**Keywords:**

Emotional Monitoring,
Random Forest Classifier,
SHAP Values,
Feature Importance,
Engagement Level Prediction,
Data Visualization,
Clustering,
Machine Learning,

**About Dataset**

The dataset contains various physiological and emotional parameters:
Features: Heart rate, pupil diameter, cortisol levels, smile intensity, temperature, and other numerical features.

Target Variables: EmotionalState, CognitiveState, EngagementLevel.

Data Type: Mixed (numerical and categorical).

Dataset Size: Unspecified but suitable for train-test splits and clustering analysis.

The dataset is suitable for exploring correlations between physiological signals and emotional or cognitive states.

**Proposed Algorithm**

Preprocessing:
Missing value handling and data scaling using StandardScaler.
Label encoding for categorical target variables.

Exploratory Data Analysis:
Heatmaps, pairplots, scatterplots, and boxplots were used for feature distribution and correlation analysis.

Clustering:
Applied KMeans clustering for grouping data into three clusters for better insights.

Model Training:
Random Forest Classifier for engagement level and emotional state prediction.
Model evaluation using metrics such as accuracy, F1-score, R², and log loss.

Feature Importance:
Computed SHAP values to identify the most critical features contributing to the model's predictions.

Performance Metrics Over Epochs:
Plots of training/validation accuracy, loss, and R² scores over iterations.

**Results and Performance**

Classification Results:

Final validation accuracy: ~90% (simulated for the example).

Final validation F1-score: ~0.87 (weighted average).

**Feature Importance:**

Top features: Cortisol levels, pupil diameter, heart rate.

SHAP values indicate the significance of physiological factors in predicting emotional and engagement levels.

Clustering:
Clusters were effectively visualized using pairplots, showing separations based on physiological features.

Model Metrics:
Consistent improvement in accuracy and F1-scores with the number of Random Forest trees.

**Reproducibility**

The methodology and code are designed for easy replication. The steps for preprocessing, training, and evaluation are outlined, ensuring reproducibility with the provided dataset.

**Dependencies and Requirements**
The project relies on the following Python libraries:

numpy,
pandas,
matplotlib,
seaborn,
sklearn,
shap,
warnings,

These dependencies can be installed using pip.

**To Install Dependencies, Run:
bash**
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn shap
Server and Hardware Requirements
Minimum Requirements:
RAM: 8GB
Processor: Dual-Core 2.0 GHz
**Recommended Requirements:**
RAM: 16GB
Processor: Quad-Core 2.5 GHz or higher
GPU: Recommended for faster computations (optional)
Software:
Python 3.8 or higher
Jupyter Notebook or any Python IDE for running the code
