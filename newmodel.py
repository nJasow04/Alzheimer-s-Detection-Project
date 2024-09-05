import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, recall_score
from sklearn.pipeline import Pipeline

# Load the data
data = pd.read_csv('alzheimer.csv') 

# Print statistics of each feature
print(data.describe())

# Prepare the features and target
X = data.drop(['Group'], axis=1)
y = data['Group']

# Encode the target variable
y = y.map({'Nondemented': 0, 'Demented': 1, 'Converted': 1})

# Split the data into training+validation (80%) and testing (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing, imputation, and model training
def create_pipeline(hidden_layers, alpha=0.0001, learning_rate_init=0.001):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.select_dtypes(include=['int64', 'float64']).columns),
            ('cat_mf', OneHotEncoder(drop='first'), ['M/F']),
            ('cat_ses', OrdinalEncoder(), ['SES'])
        ])
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('imputer', SimpleImputer(strategy='mean')),
        ('mlp', MLPClassifier(hidden_layer_sizes=hidden_layers, 
                              activation='relu',
                              solver='adam',
                              alpha=alpha,
                              learning_rate_init=learning_rate_init,
                              max_iter=1000, 
                              random_state=42))
    ])

# Define different models
models = {
    'Small': [(10,)],
    'Medium': [(20, 10)],
    'Large': [(30, 20, 10)],
    'Very Large': [(50, 30, 20, 10)],
    'Extra Large': [(100, 50, 30, 20, 10)]
}

# Define regularization types and constants
regularizations = {
    'No regularization': 0.0,
    'L2 (Ridge) - weak': 0.0001,
    'L2 (Ridge) - strong': 0.001,
    'L1 (Lasso) - weak': 0.0001,
    'L1 (Lasso) - strong': 0.001,
}

# Learning rates
learning_rates = [0.001, 0.01, 0.1]

# Prepare K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Train and validate models
results = {}

for model_size, hidden_layers in models.items():
    for reg_name, alpha in regularizations.items():
        for lr in learning_rates:
            model_name = f"{model_size} - {reg_name} - LR {lr}"
            model = create_pipeline(hidden_layers[0], alpha=alpha, learning_rate_init=lr)
            scores = cross_val_score(model, X_train_val, y_train_val, cv=kf)
            results[model_name] = scores

# Compare results
for name, scores in results.items():
    print(f"{name}: Mean accuracy = {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# Visualize results
plt.figure(figsize=(20, 10))
plt.boxplot(list(results.values()), labels=list(results.keys()))
plt.title('Model Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Choose the best model
best_model_name = max(results, key=lambda k: np.mean(results[k]))
print(f"\nBest model: {best_model_name}")

# Extract parameters from the best model name
# size, reg, lr = best_model_name.split(' - ')
# alpha = regularizations[reg.strip()]
# lr = float(lr.split()[-1])
# hidden_layers = models[size][0]

# Extract parameters from the best model name
parts = best_model_name.split(' - ')
size = parts[0]
lr = float(parts[-1].split()[-1])
reg = ' - '.join(parts[1:-1])

alpha = regularizations[reg.strip()]
hidden_layers = models[size][0]


# Create and train the best model
best_model = create_pipeline(hidden_layers, alpha=alpha, learning_rate_init=lr)
best_model.fit(X_train_val, y_train_val)

# Make predictions on the test set
y_pred = best_model.predict(X_test)




# Print the test results
print("\nTest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and print sensitivity (recall)
sensitivity = recall_score(y_test, y_pred)
print(f"\nSensitivity (Recall): {sensitivity:.4f}")

# Get predicted probabilities for the positive class
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Neural Network')
plt.legend(loc="lower right")
plt.show()

# Print AUC score
print(f"\nAUC Score: {roc_auc:.4f}")