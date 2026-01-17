import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from preprocessing import preprocess_data

# ============================================================
# ✅ CONFIGURATION
# ============================================================
FEATURES = [
    "Tenure",
    "Avg_Trans_no_month",
    "No_CurrentAccount",
    "Avg_CurrentAccount_Balance",
    "No_TermDeposit",
    "Avg_TermDeposit_Balance",
    "Avg_Loan_Balance",
    "Churn",
    # "Age",             # Excluded
    "No_Activity_Name",
    "Verify_method",
    "Client_gender",
    # "Staff",           # Excluded
    # "SMS",             # Excluded
    "EB_register_channel",
    "Type_Transactions",
    # "Max_CurrentAccount_Balance" # Excluded
    "No_Loan",
    "No_DC"
]

TARGET = "Avg_Trans_Amount"

def train_model():
    """
    Train a Gradient Boosting model to predict Avg_Trans_Amount.
    Uses specific feature subset and notebook hyperparameters.
    """
    print("=" * 60)
    print("Training Gradient Boosting Model for Avg_Trans_Amount Prediction")
    print("=" * 60)
    
    # Load training data
    print("\n1. Loading training data...")
    if os.path.exists('data/train.csv'):
        train_df = pd.read_csv('data/train.csv')
    else:
        # Fallback for local testing
        train_df = pd.read_csv('train.csv') if os.path.exists('train.csv') else pd.read_csv('data_storm_demand.csv')
        
    print(f"   - Loaded {len(train_df)} rows")
    
    # Separate features and target
    print("\n2. Preparing features and target...")
    
    if TARGET not in train_df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in training data.")
        
    y = train_df[TARGET].copy()
    
    # Filter for selected features only
    missing_features = [f for f in FEATURES if f not in train_df.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from input data: {missing_features}")
        
    X_raw = train_df[FEATURES].copy()
    print(f"   - Selected {len(FEATURES)} specific features")
    
    # Preprocess features
    print("\n3. Preprocessing features...")
    # fit_scaler=True because we are training
    X, scaler = preprocess_data(X_raw, fit_scaler=True)
    print(f"   - Feature shape: {X.shape}")
    print(f"   - Features: {X.columns.tolist()}")
    
    # Split data for validation
    print("\n4. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   - Train set: {X_train.shape}")
    print(f"   - Validation set: {X_val.shape}")
    
    # Initialize Model with Notebook Hyperparameters
    print("\n5. Initializing Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        subsample=0.8,
        n_estimators=1500,
        min_samples_leaf=15,
        max_depth=4,
        learning_rate=0.01,
        random_state=42
    )
    
    # Train Model
    print("\n6. Training model...")
    model.fit(X_train, y_train)
    print("   - Training complete")
    
    # Evaluate Model
    print("\n7. Evaluating model...")
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"   - RMSE: {rmse:,.4f}")
    print(f"   - MAE:  {mae:,.4f}")
    print(f"   - R2:   {r2:.4f}")
    
    # Save predictions (Validation set)
    print("\n8. Saving validation predictions...")
    predictions_df = pd.DataFrame({
        'actual': y_val,
        'prediction': y_pred
    })
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    predictions_path = 'output/validation_predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"   - Predictions saved to: {predictions_path}")
    
    # Save model and scaler
    print("\n9. Saving model and scaler...")
    os.makedirs('saved_models', exist_ok=True)
    
    model_path = 'saved_models/model.joblib'
    scaler_path = 'saved_models/scaler.joblib'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"   - Model saved to: {model_path}")
    print(f"   - Scaler saved to: {scaler_path}")
    
    # Also save the feature list so predict.py knows what to expect (optional helper)
    joblib.dump(FEATURES, 'saved_models/features.joblib')
    print(f"   - Feature list saved to: saved_models/features.joblib")
    
    print("\n" + "=" * 60)
    print("Training Pipeline Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    train_model()