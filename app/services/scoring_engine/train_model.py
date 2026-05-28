import os
import json

try:
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    _can_train = True
except ImportError:
    _can_train = False
from app.utils.audio_processor import preprocess_audio
from app.services.asr_service import transcribe_audio

# Feature Extractors
from app.services.feature_extraction.fluency import extract_fluency_features
from app.services.feature_extraction.intelligibility import extract_intelligibility_features
from app.services.feature_extraction.language_control import extract_language_control_features
from app.services.feature_extraction.lexical_resource import extract_lexical_features
from app.services.feature_extraction.discourse import extract_discourse_features
from app.services.scoring_engine.feature_assembler import flatten_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DIMENSIONS = ["fluency", "intelligibility", "language_control", "lexical_resource", "discourse"]

def process_audio_for_features(file_path: str) -> list:
    """Runs the phase 3 pipeline to get flattened features for a single audio file."""
    prep_result = preprocess_audio(file_path)
    processed_file_path = prep_result["processed_file_path"]
    transcription_result = transcribe_audio(processed_file_path)
    
    duration_ms = prep_result["duration_ms"]
    text = transcription_result["text"]
    
    # Phase 3 feature extraction
    fluency = extract_fluency_features(transcription_result, duration_ms)
    intelligibility = extract_intelligibility_features(transcription_result)
    grammar = extract_language_control_features(text)
    lexical = extract_lexical_features(text)
    discourse = extract_discourse_features(text)
    
    features_dict = {
        "fluency": fluency,
        "intelligibility": intelligibility,
        "language_control": grammar,
        "lexical_resource": lexical,
        "discourse": discourse
    }
    
    # Clean up temp file
    if os.path.exists(processed_file_path):
        os.remove(processed_file_path)
        
    return flatten_features(features_dict)

def train_scoring_models(dataset_path: str):
    """
    Trains RandomForestRegressor models for the 5 scoring dimensions based on an uploaded dataset.
    The dataset_path must contain audio files and a labels.json mapping filenames to scores.
    """
    if not _can_train:
        raise RuntimeError("scikit-learn and joblib are not installed in this environment. Cannot train models.")

    labels_path = os.path.join(dataset_path, "labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.json not found in {dataset_path}")
        
    with open(labels_path, "r") as f:
        labels_dict = json.load(f)
        
    X = []
    y_dicts = []
    
    for filename, scores in labels_dict.items():
        file_path = os.path.join(dataset_path, filename)
        if not os.path.exists(file_path):
            print(f"Warning: Audio file {filename} not found in {dataset_path}, skipping.")
            continue
            
        print(f"Extracting features from {filename}...")
        try:
            features_array = process_audio_for_features(file_path)
            X.append(features_array)
            y_dicts.append(scores)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
            
    if not X:
        raise ValueError("No valid training data found.")
        
    # Train a model for each dimension
    results = {}
    for dim in DIMENSIONS:
        print(f"Training model for {dim}...")
        y_dim = [float(scores.get(dim, 0.0)) for scores in y_dicts]
        
        # Using RandomForestRegressor for robust non-linear feature mapping
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y_dim)
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"{dim}_model.pkl")
        joblib.dump(model, model_path)
        results[dim] = {"status": "trained", "model_path": model_path}
        
    print("Training complete for all dimensions!")
    return results
