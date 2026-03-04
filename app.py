from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import joblib
import os
import base64
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ================= LOAD MODEL & SCALER =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "final_model.pkl"))
except Exception as e:
    print(f"Error loading model: {e}. Please retrain the model with the current sklearn version.")
    model = None  # Set to None to handle downstream

X_mean = np.load(os.path.join(BASE_DIR, "X_mean.npy"))
X_std = np.load(os.path.join(BASE_DIR, "X_std.npy"))

# Load or prepare test data for metrics
try:
    df = pd.read_csv(os.path.join(BASE_DIR, "cardio_train.csv"), sep=';')
    df['age'] = (df['age'] / 365).astype(int)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    numeric_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
    categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    X_numeric = df[numeric_features]
    X_categorical = df[categorical_features]
    y = df['cardio']
    X_numeric_scaled = (X_numeric.values - X_mean[:6]) / X_std[:6]
    X_combined = np.concatenate([X_categorical.values, X_numeric_scaled], axis=1)
    from sklearn.model_selection import train_test_split
    X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(X_combined, y, test_size=0.2, random_state=42)
except Exception as e:
    print(f"Warning: Could not load test data: {e}")
    X_train_global = None
    X_test_global = None
    y_train_global = None
    y_test_global = None

# ================= HOME PAGE =================
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("home.html")

# ================= API ROUTES =================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        
        # Handle different input formats
        age = float(data.get("age") or data.get("age_year", 0))
        height = float(data.get("height", 0))
        weight = float(data.get("weight", 0))
        ap_hi = float(data.get("ap_hi", 0))
        ap_lo = float(data.get("ap_lo", 0))
        
        gender = int(data.get("gender", 0))
        cholesterol = int(data.get("cholesterol", 1))
        gluc = int(data.get("gluc", 1))
        smoke = int(data.get("smoke", 0))
        alco = int(data.get("alco", 0))
        active = int(data.get("active", 0))
        
        # Feature engineering
        bmi = weight / ((height / 100) ** 2) if height > 0 else 0
        
        # Scale numeric features
        numeric_features = np.array([[age, height, weight, ap_hi, ap_lo, bmi]])
        numeric_scaled = (numeric_features - X_mean[:6]) / X_std[:6]
        age_s, height_s, weight_s, ap_hi_s, ap_lo_s, bmi_s = numeric_scaled[0]
        
        # Final input
        final_input = np.array([[ 
            gender, cholesterol, gluc, smoke, alco, active,
            age_s, height_s, weight_s,
            ap_hi_s, ap_lo_s, bmi_s
        ]])
        
        # Prediction
        pred = model.predict(final_input)[0]
        pred_proba = model.predict_proba(final_input)[0]
        
        probability_no_disease = float(pred_proba[0])
        probability_disease = float(pred_proba[1])
        
        message = "⚠️ High Risk of Heart Disease" if pred == 1 else "✅ Low Risk of Heart Disease"
        
        return jsonify({
            "prediction": int(pred),
            "message": message,
            "probability_no_disease": probability_no_disease,
            "probability_disease": probability_disease
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ================= ROUTES =================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # ---------- FORM INPUT ----------
            age = float(request.form["age"])
            height = float(request.form["height"])
            weight = float(request.form["weight"])
            ap_hi = float(request.form["ap_hi"])
            ap_lo = float(request.form["ap_lo"])

            gender = int(request.form["gender"])
            cholesterol = int(request.form["cholesterol"])
            gluc = int(request.form["gluc"])
            smoke = int(request.form["smoke"])
            alco = int(request.form["alco"])
            active = int(request.form["active"])

            # ---------- FEATURE ENGINEERING ----------
            bmi = weight / ((height / 100) ** 2)

            # ---------- SCALE NUMERIC FEATURES ----------
            numeric_features = np.array([[age, height, weight, ap_hi, ap_lo, bmi]])
            numeric_scaled = (numeric_features - X_mean[:6]) / X_std[:6]
            age_s, height_s, weight_s, ap_hi_s, ap_lo_s, bmi_s = numeric_scaled[0]

            # ---------- FINAL INPUT ----------
            final_input = np.array([[ 
                gender, cholesterol, gluc, smoke, alco, active,
                age_s, height_s, weight_s,
                ap_hi_s, ap_lo_s, bmi_s
            ]])

            # ---------- PREDICTION ----------
            pred = model.predict(final_input)[0]
            result = "⚠️ High Risk of Heart Disease" if pred == 1 else "✅ Low Risk of Heart Disease"

            return render_template("result.html", prediction=result)

        except Exception as e:
            return render_template("result.html", prediction=f"Error: {e}")

    # If GET request, just show the form
    return render_template("predict.html")


@app.route("/metrics")
def metrics():
    return render_template("metrics.html")

@app.route("/api/graphs", methods=["GET"])
def api_graphs():
    try:
        if X_test_global is None or y_test_global is None:
            return jsonify({"error": "Test data not available"}), 500
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test_global)
        y_pred_proba = model.predict_proba(X_test_global)[:, 1]
        
        # Calculate metrics
        metrics_dict = {
            "accuracy": float(accuracy_score(y_test_global, y_pred)),
            "precision": float(precision_score(y_test_global, y_pred)),
            "recall": float(recall_score(y_test_global, y_pred)),
            "f1": float(f1_score(y_test_global, y_pred)),
            "roc_auc": float(roc_auc_score(y_test_global, y_pred_proba))
        }
        
        graphs = {}
        
        # Confusion Matrix
        cm = confusion_matrix(y_test_global, y_pred)
        plt.figure(figsize=(10, 8), facecolor='white')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                    xticklabels=['No Disease', 'Disease'], 
                    yticklabels=['No Disease', 'Disease'],
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': 14, 'fontweight': 'bold'},
                    linewidths=2, linecolor='black')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=100)
        buf.seek(0)
        graphs['confusion_matrix'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test_global, y_pred_proba)
        plt.figure(figsize=(10, 7), facecolor='white')
        plt.plot(fpr, tpr, linewidth=3, label='ROC Curve', color='#e74c3c')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - Model Performance', fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=100)
        buf.seek(0)
        graphs['roc_curve'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Feature Importance (if model has feature_importances_)
        if hasattr(model, 'feature_importances_'):
            feature_names = ['Active', 'Age', 'Alcohol', 'BMI', 'Cholesterol', 'Diastolic BP', 
                           'Gender', 'Glucose', 'Height', 'Smoke', 'Systolic BP', 'Weight']
            importances = model.feature_importances_
            plt.figure(figsize=(12, 7), facecolor='white')
            indices = np.argsort(importances)[::-1]
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importances)))
            bars = plt.bar(range(len(importances)), importances[indices], color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right', fontsize=11)
            plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
            plt.title('Feature Importance Analysis', fontsize=14, fontweight='bold', pad=20)
            plt.ylim([0, max(importances[indices]) * 1.15])
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=100)
            buf.seek(0)
            graphs['feature_importance'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        
        # Metrics Bar Chart
        metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [metrics_dict['accuracy'], metrics_dict['precision'], 
                 metrics_dict['recall'], metrics_dict['f1']]
        plt.figure(figsize=(10, 7), facecolor='white')
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        bars = plt.bar(metrics_list, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.ylim([0, 1.1])
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('Performance Metrics', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=0, fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=100)
        buf.seek(0)
        graphs['metrics_bar'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test_global, y_pred_proba)
        plt.figure(figsize=(10, 7), facecolor='white')
        plt.plot(recall, precision, linewidth=3, color='#9b59b6', label='PR Curve')
        plt.fill_between(recall, precision, alpha=0.3, color='#9b59b6')
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=100)
        buf.seek(0)
        graphs['pr_curve'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Prediction Distribution
        plt.figure(figsize=(10, 7), facecolor='white')
        plt.hist(y_pred_proba, bins=40, alpha=0.7, color='#3498db', edgecolor='black', linewidth=1.5)
        plt.xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold', pad=20)
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=100)
        buf.seek(0)
        graphs['pred_dist'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Classification Report
        report = classification_report(y_test_global, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_report.iloc[:-1, :].astype(float), annot=True, fmt='.3f', cmap='YlGnBu')
        plt.title('Classification Report')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        graphs['class_report'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Prepare feature importance data
        feature_importance_data = {}
        if hasattr(model, 'feature_importances_'):
            feature_names = ['Gender', 'Cholesterol', 'Glucose', 'Smoke', 'Alcohol', 'Active',
                           'Age', 'Height', 'Weight', 'Systolic BP', 'Diastolic BP', 'BMI']
            importances = model.feature_importances_
            for name, importance in zip(feature_names, importances):
                feature_importance_data[name] = float(importance)
        
        return jsonify({
            "metrics": metrics_dict,
            "graphs": graphs,
            "feature_importance": feature_importance_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Cache for comparison results
_comparison_cache = {"data": None}

@app.route("/api/comparison-graphs", methods=["GET"])
def api_comparison_graphs():
    try:
        if X_train_global is None or X_test_global is None or y_train_global is None or y_test_global is None:
            return jsonify({"error": "Training/test data not available"}), 500
        
        # Return cached results if available
        if _comparison_cache["data"] is not None:
            print("Returning cached comparison results...")
            return jsonify(_comparison_cache["data"])
        
        print("Generating fresh comparison results...")
        from sklearn.model_selection import train_test_split
        
        models_dict = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "BernoulliNB": BernoulliNB()
        }
        
        # Prepare data for all splits
        test_sizes = [0.2, 0.3]  # Reduced from 4 to 2 test sizes for faster loading
        comparison_data = {name: {} for name in models_dict.keys()}
        
        # Case 1: Train-Test Split with different test sizes (OPTIMIZED)
        print("Evaluating models with different test sizes...")
        for test_size in test_sizes:
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train_global, y_train_global, test_size=test_size, random_state=42
            )
            
            for name, model in models_dict.items():
                try:
                    # Train model
                    model.fit(X_train_split, y_train_split)
                    # Evaluate
                    y_pred = model.predict(X_test_split)
                    accuracy = accuracy_score(y_test_split, y_pred)
                    comparison_data[name][f'Case1_TestSize_{test_size}'] = accuracy
                except Exception as e:
                    print(f"Error training {name} with test_size={test_size}: {e}")
                    comparison_data[name][f'Case1_TestSize_{test_size}'] = None
        
        # Case 2: Cross-Validation (OPTIMIZED - reduced folds)
        print("Performing cross-validation...")
        for name, model in models_dict.items():
            try:
                cv_scores = cross_val_score(model, X_train_global, y_train_global, cv=3, scoring='accuracy', n_jobs=-1)
                comparison_data[name]['Case2_CV_Mean'] = float(cv_scores.mean())
                comparison_data[name]['Case2_CV_Std'] = float(cv_scores.std())
            except Exception as e:
                print(f"Error in CV for {name}: {e}")
                comparison_data[name]['Case2_CV_Mean'] = None
                comparison_data[name]['Case2_CV_Std'] = None
        
        # Case 3: Train on full training set and evaluate on test set (Best Accuracy)
        print("Evaluating on full test set...")
        for name, model in models_dict.items():
            try:
                model.fit(X_train_global, y_train_global)
                y_pred = model.predict(X_test_global)
                accuracy = accuracy_score(y_test_global, y_pred)
                comparison_data[name]['Case3_Best_Accuracy'] = accuracy
            except Exception as e:
                print(f"Error evaluating {name} on test set: {e}")
                comparison_data[name]['Case3_Best_Accuracy'] = None
        
        # Prepare table data
        table_data = []
        for name in models_dict.keys():
            row = {"Algorithm": name}
            for test_size in test_sizes:
                row[f'Case1_TestSize_{test_size}'] = comparison_data[name].get(f'Case1_TestSize_{test_size}')
            row['Case2_CV_Mean'] = comparison_data[name].get('Case2_CV_Mean')
            row['Case2_CV_Std'] = comparison_data[name].get('Case2_CV_Std')
            row['Case3_Best_Accuracy'] = comparison_data[name].get('Case3_Best_Accuracy')
            table_data.append(row)
        
        # Generate comparison graphs
        graphs = {}
        
        # Case 1: Train-Test Split Comparison
        plt.figure(figsize=(14, 8), facecolor='white')
        x = np.arange(len(models_dict.keys()))
        width = 0.35
        colors = ['#e74c3c', '#f39c12']
        
        for i, test_size in enumerate(test_sizes):
            accuracies = [comparison_data[name].get(f'Case1_TestSize_{test_size}', 0) or 0 for name in models_dict.keys()]
            bars = plt.bar(x + i*width, accuracies, width, label=f'Test Size {test_size}', 
                          color=colors[i], edgecolor='black', linewidth=1.5, alpha=0.85)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Case 1: Train-Test Split Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(x + width * 0.5, models_dict.keys(), rotation=45, ha='right', fontsize=10)
        plt.ylim([0, 1])
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=100)
        buf.seek(0)
        graphs['case1'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Case 2: Cross-Validation Comparison
        plt.figure(figsize=(14, 8), facecolor='white')
        algorithms = list(models_dict.keys())
        cv_means = [comparison_data[name].get('Case2_CV_Mean', 0) or 0 for name in algorithms]
        cv_stds = [comparison_data[name].get('Case2_CV_Std', 0) or 0 for name in algorithms]
        
        x_pos = np.arange(len(algorithms))
        bars = plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=8, alpha=0.85, 
                      color='#9b59b6', edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
        plt.ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
        plt.title('Case 2: Cross-Validation Comparison (3-Fold CV)', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(x_pos, algorithms, rotation=45, ha='right', fontsize=10)
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=100)
        buf.seek(0)
        graphs['case2'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Case 3: Best Accuracy Comparison
        plt.figure(figsize=(14, 8), facecolor='white')
        best_accuracies = [comparison_data[name].get('Case3_Best_Accuracy', 0) or 0 for name in algorithms]
        
        bars = plt.bar(x_pos, best_accuracies, alpha=0.85, color='#1abc9c', 
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Case 3: Best Accuracy on Test Set', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(x_pos, algorithms, rotation=45, ha='right', fontsize=10)
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=100)
        buf.seek(0)
        graphs['case3'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Cache the results
        result = {
            "graphs": graphs,
            "table": table_data
        }
        _comparison_cache["data"] = result
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/accuracy")
def accuracy():
    return render_template("accuracy.html")

@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/graphs")
def graphs():
    return render_template("graphs.html")

# ================= RUN APP =================
if __name__ == "__main__":
    print("Flask server running...")
    app.run(debug=True)
