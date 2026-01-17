import os
import pandas as pd
import joblib
import sys
# Assuming your directory structure requires 'src.' based on the provided file.
# If files are in the same directory, change to: from preprocessing import preprocess_data
try:
    from src.preprocessing import preprocess_data
except ImportError:
    from preprocessing import preprocess_data

# ============================================================
# ❌ DO NOT MODIFY
# These paths are fixed for competition evaluation.
# The grading server will mount data to these locations.
# ============================================================
INPUT_PATH = os.environ.get('INPUT_PATH', '/data/input')
OUTPUT_PATH = os.environ.get('OUTPUT_PATH', '/data/output')
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "scaler.joblib")


# ============================================================
# ❌ DO NOT MODIFY
# System path validation functions
# ============================================================
def validate_paths():
    """Validate input/output paths exist and create output directory if needed."""
    if not os.path.exists(INPUT_PATH):
        # Only strict check if we are running in the evaluation environment
        if os.environ.get('INPUT_PATH'):
            print(f"ERROR: Input directory not found: {INPUT_PATH}")
            print("Please ensure input data is mounted to /data/input")
            sys.exit(1)
        else:
            # Local fallback for testing
            os.makedirs(INPUT_PATH, exist_ok=True)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)


def get_input_files():
    """Get list of CSV files to process from INPUT_PATH."""
    try:
        files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".csv")]
    except FileNotFoundError:
        files = []
        
    if not files:
        print(f"WARNING: No CSV files found in {INPUT_PATH}")
        print("Inference completed with no files to process.")
    return files


# ============================================================
# ✅ MODIFIABLE
# Model loading - Custom logic
# ============================================================
def load_model():
    """Load the trained model and scaler."""
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        
        print(f"Loading scaler from {SCALER_PATH}...")
        scaler = joblib.load(SCALER_PATH)
        
        return {'model': model, 'scaler': scaler}
    except Exception as e:
        print(f"ERROR: Failed to load model/scaler: {e}")
        sys.exit(1)


def preprocess_input(df, artifacts):
    """
    Preprocess input data for prediction.
    
    Args:
        df: Input DataFrame
        artifacts: Dictionary containing loaded model and scaler
        
    Returns:
        X: Processed features ready for prediction
        ids: Series/List of identifiers (Customer_number) to map predictions back
    """
    # Extract Customer_number for the output file
    if 'Customer_number' in df.columns:
        customer_number = df['Customer_number']
    else:
        # Create dummy index if ID is missing
        customer_number = df.index
        
    # Use the shared preprocessing logic
    # fit_scaler=False because we use the pre-trained scaler
    X, _ = preprocess_data(df, scaler=artifacts['scaler'], fit_scaler=False)
    
    return X, customer_number


def make_predictions(artifacts, X):
    """
    Generate predictions using the loaded model.
    """
    model = artifacts['model']
    predictions = model.predict(X)
    return predictions


def save_predictions(predictions, customer_number, filename):
    """Save predictions to CSV in the required format."""
    results_df = pd.DataFrame({
        'Customer_number': customer_number,
        'prediction': predictions
    })
    
    # Construct output filename
    base_name = os.path.basename(filename)
    output_file = os.path.join(OUTPUT_PATH, f"pred_{base_name}")
    
    results_df.to_csv(output_file, index=False)
    print(f"  - Saved to {output_file}")


def process_single_file(artifacts, filename):
    """
    Pipeline for a single file: load, preprocess, predict, save.
    """
    file_path = os.path.join(INPUT_PATH, filename)
    try:
        print(f"Processing: {filename}")
        df = pd.read_csv(file_path)
        print(f"  - Loaded {len(df)} rows")
        
        # ✅ Modifiable: Preprocessing
        X, customer_number = preprocess_input(df, artifacts)
        
        # ✅ Modifiable: Prediction
        predictions = make_predictions(artifacts, X)
        
        # ❌ Fixed: Save output
        save_predictions(predictions, customer_number, filename)
        
        print(f"  ✓ Success\n")
        
    except Exception as e:
        print(f"  ✗ ERROR processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        print()


# ============================================================
# ❌ DO NOT MODIFY
# Main inference pipeline
# ============================================================
def run_inference():
    
    # Validate system paths
    validate_paths()
    
    # Load model
    artifacts = load_model()
    
    # Get input files
    files = get_input_files()
    if not files:
        return
    
    print(f"\nFound {len(files)} CSV file(s) to process: {files}\n")
    
    # Process each file
    for filename in files:
        process_single_file(artifacts, filename)
    
    print("=" * 60)
    print("Inference Completed")
    print("=" * 60)


if __name__ == "__main__":
    run_inference()