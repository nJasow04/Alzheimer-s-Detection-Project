import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Prepare the features and target
    X = data.drop('Group', axis=1)
    y = data['Group']
    
    # Encode the target variable
    y = y.map({'Nondemented': 0, 'Demented': 1, 'Converted': 1})
    
    # Split the data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define numeric and categorical columns
    numeric_features = ['Age', 'EDUC', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
    categorical_features = ['M/F', 'SES']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    # Fit and transform the data
    X_train_val = preprocessor.fit_transform(X_train_val)
    X_test = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    numeric_features_out = numeric_features
    try:
        # For newer versions of scikit-learn
        categorical_features_out = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
    except AttributeError:
        # For older versions of scikit-learn
        categorical_features_out = preprocessor.named_transformers_['cat'].get_feature_names(categorical_features).tolist()
    feature_names = numeric_features_out + categorical_features_out
    
    return X_train_val, X_test, y_train_val, y_test, feature_names

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_model(input_dim, hidden_layers, nodes_per_layer, learning_rate, regularization):
    tf.keras.backend.clear_session()
    model = Sequential()
    
    # Input layer
    model.add(Dense(nodes_per_layer[0], activation='relu', input_dim=input_dim, 
                    kernel_regularizer=l2(regularization)))
    model.add(Dropout(0.2))
    
    # Hidden layers
    for i in range(1, hidden_layers):
        model.add(Dense(nodes_per_layer[i], activation='relu', 
                        kernel_regularizer=l2(regularization)))
        model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_models():
    models = {
        'Model1': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.01, 'regularization': 0},
        'Model2': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.005, 'regularization': 0},
        'Model3': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.001, 'regularization': 0},
        'Model4': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.01, 'regularization': 0},
        'Model5': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.005, 'regularization': 0},
        'Model6': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.001, 'regularization': 0},
        'Model7': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.01, 'regularization': 0},
        'Model8': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.005, 'regularization': 0},
        'Model9': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.001, 'regularization': 0},
        'Model10': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.01, 'regularization': 0.01},
        'Model11': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.005, 'regularization': 0.01},
        'Model12': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.001, 'regularization': 0.01},
        'Model13': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.01, 'regularization': 0.01},
        'Model14': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.005, 'regularization': 0.01},
        'Model15': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.001, 'regularization': 0.01},
        'Model16': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.01, 'regularization': 0.01},
        'Model17': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.005, 'regularization': 0.01},
        'Model18': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.001, 'regularization': 0.01},
        'Model19': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.01, 'regularization': 0.005},
        'Model20': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.005, 'regularization': 0.005},
        'Model21': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.001, 'regularization': 0.005},
        'Model22': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.01, 'regularization': 0.005},
        'Model23': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.005, 'regularization': 0.005},
        'Model24': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.001, 'regularization': 0.005},
        'Model25': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.01, 'regularization': 0.005},
        'Model26': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.005, 'regularization': 0.005},
        'Model27': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.001, 'regularization': 0.005},
        'Model28': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.01, 'regularization': 0.05},
        'Model29': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.005, 'regularization': 0.05},
        'Model30': {'hidden_layers': 2, 'nodes_per_layer': [64, 32], 'learning_rate': 0.001, 'regularization': 0.05},
        'Model31': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.01, 'regularization': 0.05},
        'Model32': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.005, 'regularization': 0.05},
        'Model33': {'hidden_layers': 3, 'nodes_per_layer': [64, 32, 16], 'learning_rate': 0.001, 'regularization': 0.05},
        'Model34': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.01, 'regularization': 0.05},
        'Model35': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.005, 'regularization': 0.05},
        'Model36': {'hidden_layers': 4, 'nodes_per_layer': [128, 64, 32, 16], 'learning_rate': 0.001, 'regularization': 0.05},
        
    }
    return models



import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

def plot_roc_curve(y_true, y_pred, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.show()
    
    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, fbeta_score

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    error_rate = 1 - accuracy
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    sensitivity = recall
    specificity = tn / (tn + fp)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'F2 Score': f2,
        'Error Rate': error_rate,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }

from tensorflow.keras.callbacks import EarlyStopping

def train_and_evaluate():
    # Load and preprocess data
    X_train_val, X_test, y_train_val, y_test, feature_names = load_and_preprocess_data('alzheimer.csv')
    
    # Create models
    models_config = create_models()
    
    # Prepare K-Fold cross-validation
    kf = KFold(n_splits=9, shuffle=True, random_state=42)
    
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    results = {}
    
    for name, config in models_config.items():
        fold_scores = []
        for train_index, val_index in kf.split(X_train_val):
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
            
            model = create_model(X_train.shape[1], **config)
            
            history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
            
            y_pred = (model.predict(X_val) > 0.5).astype(int)
            metrics = calculate_metrics(y_val, y_pred)
            fold_scores.append(metrics)
        
        # Calculate mean and std of each metric across folds
        mean_scores = {k: np.mean([fold[k] for fold in fold_scores]) for k in fold_scores[0]}
        std_scores = {k: np.std([fold[k] for fold in fold_scores]) for k in fold_scores[0]}
        
        results[name] = {'mean': mean_scores, 'std': std_scores}
    
    # Print results
    for name, scores in results.items():
        print(f"\n{name}:")
        print(f"Configuration: {models_config[name]}")
        for metric, value in scores['mean'].items():
            print(f"{metric}: {value:.4f} (+/- {scores['std'][metric]:.4f})")
    
    # Test best model
    best_model_name = max(results, key=lambda k: results[k]['mean']['F1 Score'])
    best_config = models_config[best_model_name]
    best_model = create_model(X_train_val.shape[1], **best_config)
    best_model.fit(X_train_val, y_train_val, epochs=100, batch_size=32, verbose=0)
    
    # Train best model with validation split and early stopping
    history = best_model.fit(X_train_val, y_train_val, epochs=100, batch_size=32, 
                             validation_split=0.2, verbose=0, callbacks=[early_stopping])
    
    
    # Save the best model
    best_model.save(f'best_model_{best_model_name}.h5')
    
    y_pred = (best_model.predict(X_test) > 0.5).astype(int)
    test_metrics = calculate_metrics(y_test, y_pred)
    
    print(f"\nBest Model ({best_model_name}) Test Results:")
    print(f"Best Model Configuration: {best_config}")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Plot ROC curve
    plot_roc_curve(y_test, best_model.predict(X_test).ravel(), best_model_name)
    
    # Feature importance (for the best model)
    feature_importance = np.abs(best_model.layers[0].get_weights()[0]).mean(axis=1)
    feature_importance = feature_importance / np.sum(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Item 6: Plot learning curve
    plot_learning_curve(history, best_model_name)
    
# Define the plot_learning_curve function outside of train_and_evaluate
def plot_learning_curve(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    train_and_evaluate()