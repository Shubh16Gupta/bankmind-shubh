# train.py - Complete with EDA and Model Training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from catboost import CatBoostClassifier
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BANKMIND - TRACK C: SYSTEM BUILDER")
print("="*60)

# ------------------------------
# 1. LOAD DATA
# ------------------------------
print("\n📂 Loading data...")
df = pd.read_csv('bank-full.csv', sep=';')
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA) - Track B Step 1
# ------------------------------
print("\n" + "="*60)
print("📊 EXPLORATORY DATA ANALYSIS")
print("="*60)

# 2.1 Class Distribution
print("\n📈 Class Distribution:")
class_dist = df['y'].value_counts(normalize=True)
print(class_dist)
print(f"✅ Positive class (subscribed): {class_dist['yes']*100:.2f}%")
print(f"✅ Negative class (not subscribed): {class_dist['no']*100:.2f}%")

# 2.2 Subscription Rate by Job
print("\n💼 Subscription Rate by Job:")
job_rate = df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean()).sort_values(ascending=False)
print(job_rate)
print(f"✅ Highest: {job_rate.index[0]} ({job_rate.iloc[0]*100:.1f}%)")
print(f"✅ Lowest: {job_rate.index[-1]} ({job_rate.iloc[-1]*100:.1f}%)")

# 2.3 Balance Analysis
print("\n💰 Balance Analysis:")
balance_yes = df[df['y'] == 'yes']['balance'].median()
balance_no = df[df['y'] == 'no']['balance'].median()
print(f"✅ Median balance (subscribed): €{balance_yes:,.0f}")
print(f"✅ Median balance (not subscribed): €{balance_no:,.0f}")

# 2.4 Age Group Analysis
print("\n👤 Age Group Analysis:")
age_bins = [18, 31, 46, 60, 100]
age_labels = ['18-30', '31-45', '46-60', '60+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
age_rate = df.groupby('age_group')['y'].apply(lambda x: (x == 'yes').mean())
print(age_rate)
print(f"✅ Highest subscription: {age_rate.idxmax()} ({age_rate.max()*100:.1f}%)")

# 2.5 Housing Loan Impact
print("\n🏠 Housing Loan Impact:")
housing_rate = df.groupby('housing')['y'].apply(lambda x: (x == 'yes').mean())
print(housing_rate)
print(f"✅ With housing loan: {housing_rate['yes']*100:.1f}%")
print(f"✅ Without housing loan: {housing_rate['no']*100:.1f}%")

# ------------------------------
# 3. VISUALIZATIONS (optional but helpful)
# ------------------------------
print("\n📊 Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Subscription rate by job
job_rate_sorted = job_rate.sort_values(ascending=True)
job_rate_sorted.plot(kind='barh', ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Subscription Rate by Job Type', fontsize=14)
axes[0, 0].set_xlabel('Subscription Rate')
axes[0, 0].set_ylabel('Job Type')
axes[0, 0].axvline(df['y'].map({'yes':1, 'no':0}).mean(), color='red', linestyle='--', label='Overall Average')
axes[0, 0].legend()

# Plot 2: Balance distribution by subscription
df_box = df.copy()
df_box['y_binary'] = df_box['y'].map({'yes': 1, 'no': 0})
df_box.boxplot(column='balance', by='y', ax=axes[0, 1])
axes[0, 1].set_title('Balance Distribution by Subscription', fontsize=14)
axes[0, 1].set_xlabel('Subscribed')
axes[0, 1].set_ylabel('Balance (€)')

# Plot 3: Subscription rate by age group
age_rate.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Subscription Rate by Age Group', fontsize=14)
axes[1, 0].set_xlabel('Age Group')
axes[1, 0].set_ylabel('Subscription Rate')
axes[1, 0].axhline(df['y'].map({'yes':1, 'no':0}).mean(), color='red', linestyle='--', label='Overall Average')
axes[1, 0].legend()

# Plot 4: Housing loan impact
housing_rate.plot(kind='bar', ax=axes[1, 1], color='coral')
axes[1, 1].set_title('Housing Loan Impact on Subscription', fontsize=14)
axes[1, 1].set_xlabel('Has Housing Loan')
axes[1, 1].set_ylabel('Subscription Rate')
axes[1, 1].axhline(df['y'].map({'yes':1, 'no':0}).mean(), color='red', linestyle='--', label='Overall Average')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
print("✅ Visualizations saved to 'eda_visualizations.png'")

# ------------------------------
# 4. PREPARE DATA FOR MODELING - Track B Step 2
# ------------------------------
print("\n" + "="*60)
print("🤖 MODEL TRAINING")
print("="*60)

print("\n🔄 Preparing data for modeling...")

# Drop duration (leakage) and prepare features
X = df.drop(['y', 'duration', 'age_group'], axis=1)
y = (df['y'] == 'yes').astype(int)

# Identify categorical columns
cat_features = X.select_dtypes(include='object').columns.tolist()
print(f"✅ Categorical features: {cat_features}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train set: {len(X_train)} samples")
print(f"✅ Test set: {len(X_test)} samples")

# ------------------------------
# 5. TRAIN LOGISTIC REGRESSION (Baseline)
# ------------------------------
print("\n📊 Training Logistic Regression (baseline)...")

# One-hot encode
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, join='left', axis=1, fill_value=0
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Train
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("\n" + "-"*40)
print("LOGISTIC REGRESSION RESULTS")
print("-"*40)
print(classification_report(y_test, y_pred_lr))
print(f"✅ ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")

# ------------------------------
# 6. TRAIN CATBOOST (Main Model)
# ------------------------------
print("\n📊 Training CatBoost (main model)...")

cb = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    cat_features=cat_features,
    verbose=50,
    scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
    random_state=42
)
cb.fit(X_train, y_train)
y_pred_cb = cb.predict(X_test)
y_proba_cb = cb.predict_proba(X_test)[:, 1]

print("\n" + "-"*40)
print("CATBOOST RESULTS")
print("-"*40)
print(classification_report(y_test, y_pred_cb))
print(f"✅ ROC-AUC: {roc_auc_score(y_test, y_proba_cb):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_cb)
print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives: {cm[0,0]}")
print(f"   False Positives: {cm[0,1]}")
print(f"   False Negatives: {cm[1,0]}")
print(f"   True Positives: {cm[1,1]}")

# ------------------------------
# 7. FEATURE IMPORTANCE - Track B Step 3
# ------------------------------
print("\n" + "="*60)
print("📊 FEATURE IMPORTANCE")
print("="*60)

feature_names = X_train.columns.tolist()
importances = cb.feature_importances_
feature_importance_dict = dict(zip(feature_names, importances))
sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\n📈 Top 10 Most Important Features:")
for i, (name, importance) in enumerate(sorted_importance[:10], 1):
    print(f"   {i:2d}. {name:15s}: {importance:.4f}")

print(f"\n✅ Most important feature: {sorted_importance[0][0]} ({sorted_importance[0][1]:.4f})")

# ------------------------------
# 8. SAMPLE PREDICTIONS - Track B Step 4
# ------------------------------
print("\n" + "="*60)
print("📋 SAMPLE PREDICTIONS (5 customers)")
print("="*60)

# Select 5 random customers from test set (at least 2 yes, 2 no)
yes_indices = np.where(y_test == 1)[0]
no_indices = np.where(y_test == 0)[0]

sample_indices = list(np.random.choice(yes_indices, size=min(3, len(yes_indices)), replace=False))
sample_indices.extend(list(np.random.choice(no_indices, size=min(3, len(no_indices)), replace=False)))
sample_indices = sample_indices[:5]

for i, idx in enumerate(sample_indices, 1):
    customer = X_test.iloc[idx]
    actual = y_test.iloc[idx]
    pred = y_pred_cb[idx]
    proba = y_proba_cb[idx]
    
    print(f"\n📌 Customer {i}:")
    print(f"   Age: {customer['age']}, Job: {customer['job']}, Balance: €{customer['balance']:,.0f}")
    print(f"   Housing Loan: {customer['housing']}, Personal Loan: {customer['loan']}")
    print(f"   ✅ Actual: {'Subscribed' if actual == 1 else 'Not Subscribed'}")
    print(f"   🤖 Prediction: {'Subscribed' if pred == 1 else 'Not Subscribed'} (Probability: {proba*100:.1f}%)")

# ------------------------------
# 9. SAVE MODEL AND ARTIFACTS
# ------------------------------
print("\n" + "="*60)
print("💾 SAVING MODEL AND ARTIFACTS")
print("="*60)

# Save model
joblib.dump(cb, 'model.pkl')
print("✅ Model saved as 'model.pkl'")

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ Feature names saved as 'feature_names.pkl'")

# Save feature importance
with open('feature_importance.pkl', 'wb') as f:
    pickle.dump(sorted_importance, f)
print("✅ Feature importance saved as 'feature_importance.pkl'")

# Save scaler (for Logistic Regression if needed)
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved as 'scaler.pkl'")

# Save encoder info
with open('cat_features.pkl', 'wb') as f:
    pickle.dump(cat_features, f)
print("✅ Categorical features saved as 'cat_features.pkl'")

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print("\n✅ All artifacts saved successfully!")
print("✅ You can now run: python3 -m uvicorn app:app --reload")
print("="*60)