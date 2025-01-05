import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder


import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\HP\\Desktop\\emotional_monitoring_dataset_with_target (1).csv")
data.head()

data.info()

#Emotional state

sns.countplot(data=data,x='EmotionalState')
for container in plt.gca().containers:
    plt.bar_label(container, label_type="edge", padding=2)
    plt.show()

#EngagementLevel

sns.countplot(data=data,x='EngagementLevel')
for container in plt.gca().containers:
    plt.bar_label(container, label_type="edge", padding=2)
    plt.show()


#EmotionalState and CongnitiveState 

numeric_data = data.drop(columns=['EmotionalState', 'CognitiveState'])

# Set up the plot size and resolution
plt.figure(figsize=(15, 15))
plt.gcf().set_dpi(300)  # High resolution

# Create the heatmap
sns.heatmap(
    numeric_data.corr(),
    annot=True,             # Display correlation values
    fmt='.2f',              # Format annotations to 2 decimal places
    cmap='coolwarm',        # Colormap for contrast
    annot_kws={"size": 10, "color": "black"}  # Annotation properties
)

# Display the heatmap
plt.title('Heatmap of Numeric Features Correlation', fontsize=16)
plt.show()


data.hist(figsize=(12, 10), bins=15, color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()

#Boxplot
plt.figure(figsize=(15, 8))
sns.boxplot(data=data.select_dtypes(include=['float64', 'int64']))
plt.xticks(rotation=45)
plt.title('Boxplots for Numerical Variables')
plt.show()

#pairplot
sns.pairplot(data, vars=['HeartRate', 'CortisolLevel', 'Temperature', 'SmileIntensity'], hue='EmotionalState')
plt.show()


#scatterplot
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='PupilDiameter', y='CortisolLevel', hue='EmotionalState')
plt.title('Pupil Diameter vs Cortisol Level by Emotional State')
plt.show()

from sklearn.cluster import KMeans # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize clusters
sns.pairplot(data, vars=['HeartRate', 'CortisolLevel', 'PupilDiameter'], hue='Cluster')
plt.show()

engagement_dist = data['EngagementLevel'].value_counts()
engagement_dist.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Engagement Level Distribution')
plt.ylabel('')
plt.show()



# Preprocess data: Extract features and target
X = data.drop(columns=['EmotionalState', 'CognitiveState', 'EngagementLevel'])  # Features
y = data['EngagementLevel']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (Random Forest for demonstration)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# --- 1. Data Distribution: True Vs Predictions ---
true_counts = y_test.value_counts().sort_index()
pred_counts = pd.Series(y_pred).value_counts().sort_index()

plt.figure(figsize=(10, 6))
width = 0.35  # Bar width
indices = np.arange(len(true_counts))

plt.bar(indices, true_counts, width=width, label='True', alpha=0.7, color='blue')
plt.bar(indices + width, pred_counts, width=width, label='Predictions', alpha=0.7, color='orange')
plt.xlabel('Engagement Level')
plt.ylabel('Count')
plt.title('Data Distribution: True Vs Predictions')
plt.xticks(indices + width / 2, true_counts.index)
plt.legend()
plt.show()

# --- 2. Model Predictions: True Vs Predictions ---
plt.figure(figsize=(10, 6))
plt.bar(['True'], [true_counts.sum()], width=width, label='True', alpha=0.7, color='blue')
plt.bar(['Predictions'], [pred_counts.sum()], width=width, label='Predictions', alpha=0.7, color='orange')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Model Predictions: True Vs Predictions')
plt.legend()
plt.show()

# --- 3. Mean(|SHAPvalue|) (average impact on model output magnitude) ---
explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# Handle multi-class SHAP values by taking the mean across classes
# shap_values.values has shape (num_samples, num_features, num_classes)
mean_shap_values = np.abs(shap_values.values).mean(axis=0).mean(axis=1)

# Feature names
features = X_test.columns

# Plot the SHAP values as a bar chart
plt.figure(figsize=(10, 6))
plt.barh(features, mean_shap_values, color='green', alpha=0.7)
plt.xlabel('Mean(|SHAP value|)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance: Mean(|SHAP value|)', fontsize=14)
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()




# ---------------- 1. Training and Validation IOU Over Epochs ----------------
# Simulated IOU scores (replace with actual metrics if available)
epochs = list(range(1, 11))
train_iou = np.random.uniform(0.5, 1.0, size=10)  # Simulated training IOU
val_iou = np.random.uniform(0.4, 0.9, size=10)    # Simulated validation IOU

# Plot Training and Validation IOU (Line Plot)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_iou, marker='o', label="Training IOU", color='blue')
plt.plot(epochs, val_iou, marker='o', label="Validation IOU", color='orange')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.title('Training and Validation IOU Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# ---------------- 2. Model Metrics Over Epochs ----------------
# Simulated metrics (replace with actual metrics if available)
accuracy = np.random.uniform(0.7, 1.0, size=10)  # Simulated accuracy
precision = np.random.uniform(0.6, 0.95, size=10)  # Simulated precision
recall = np.random.uniform(0.5, 0.9, size=10)  # Simulated recall

# Plot Model Metrics (Line Plot)
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, marker='o', label="Accuracy", color='green')
plt.plot(epochs, precision, marker='o', label="Precision", color='purple')
plt.plot(epochs, recall, marker='o', label="Recall", color='red')
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.title('Model Metrics Over Epochs')
plt.legend()
plt.grid(True)
plt.show()


# Encode categorical target variable
le = LabelEncoder()
data['EmotionalState'] = le.fit_transform(data['EmotionalState'])

# Identify numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Calculate standard deviation for numeric columns
std_devs = data[numeric_columns].std()

# Plot standard deviation as a bar graph
plt.figure(figsize=(10, 6))
std_devs.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Standard Deviation of Numeric Features')
plt.xlabel('Features')
plt.ylabel('Standard Deviation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Split data
X = data[numeric_columns]
y = data['EmotionalState']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)

train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []
f1_scores = []

for n in range(1, 101):
    partial_model = RandomForestClassifier(random_state=42, n_estimators=n)
    partial_model.fit(X_train_scaled, y_train)
    
    y_train_pred = partial_model.predict(X_train_scaled)
    y_val_pred = partial_model.predict(X_val_scaled)
    
    # Compute metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    train_loss = log_loss(y_train, partial_model.predict_proba(X_train_scaled))
    val_loss = log_loss(y_val, partial_model.predict_proba(X_val_scaled))
    f1 = f1_score(y_val, y_val_pred, average='weighted')

    # Store metrics
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    f1_scores.append(f1)

# Display final accuracy and F1-score
final_accuracy = val_accuracies[-1]
final_f1_score = f1_scores[-1]
print(f"Final Validation Accuracy: {final_accuracy}")
print(f"Final Validation F1-Score: {final_f1_score}")

# Accuracy bar graph
plt.figure(figsize=(10, 6))
plt.bar(['Validation Accuracy'], [final_accuracy], color='blue', edgecolor='black')
plt.title('Final Validation Accuracy')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

# F1-score bar graph
plt.figure(figsize=(10, 6))
plt.bar(['Validation F1-Score'], [final_f1_score], color='green', edgecolor='black')
plt.title('Final Validation F1-Score')
plt.ylabel('F1-Score')
plt.tight_layout()
plt.show()

# Accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), train_accuracies, label='Training Accuracy', color='blue')
plt.plot(range(1, 101), val_accuracies, label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, 101), val_losses, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Number of Trees')
plt.ylabel('Log Loss')
plt.legend()
plt.show()

# F1-score plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), f1_scores, label='Validation F1-score', color='green')
plt.title('F1-score Over Epochs')
plt.xlabel('Number of Trees')
plt.ylabel('F1-score')
plt.legend()
plt.show()

# R^2 scores
train_r2 = []
val_r2 = []

for n in range(1, 101):
    partial_model = RandomForestClassifier(random_state=42, n_estimators=n)
    partial_model.fit(X_train_scaled, y_train)
    
    train_pred = partial_model.predict(X_train_scaled)
    val_pred = partial_model.predict(X_val_scaled)
    
    train_r2.append(r2_score(y_train, train_pred))
    val_r2.append(r2_score(y_val, val_pred))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), train_r2, label='Training R^2', color='purple')
plt.plot(range(1, 101), val_r2, label='Validation R^2', color='brown')
plt.title('Training vs Validation R^2 Over Epochs')
plt.xlabel('Number of Trees')
plt.ylabel('R^2')
plt.legend()
plt.show()


