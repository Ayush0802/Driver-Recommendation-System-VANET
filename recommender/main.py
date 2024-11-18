from train import SecureTelemetryTrainer, SecurePhysiologyTrainer
from predict import SecureDriverRecommender
import pandas as pd

def train_telematics_model():
    print("=== Training Telematics Model ===")
    trainer = SecureTelemetryTrainer()
    telematics_df = pd.read_csv('telematics.csv')
    X_telematics = telematics_df[['gps_speed', 'cTemp', 'rpm', 'eLoad', 'hard_brake', 'total_acceleration', 'angular_acceleration']]
    y_telematics = trainer.create_risk_labels(X_telematics)
    test_score = trainer.train_model(X_telematics, y_telematics)
    print(f"Telematics Testing Score: {test_score:.4f}")
    trainer.save_model()

def train_physiological_model():
    print("=== Training Physiological Model ===")
    trainer = SecurePhysiologyTrainer()
    physiological_df = pd.read_csv('physiological.csv')
    X_physiological = physiological_df[['Body_Temperature', 'Heart_Rate', 'SPO2']]
    y_physiological = trainer.create_risk_labels(physiological_df[['Driver_State']])
    test_score = trainer.train_model(X_physiological, y_physiological)
    print(f"Physiological Testing Score: {test_score:.4f}")
    trainer.save_model()

def test_prediction():
    print("\n=== Testing Prediction ===")
    recommender = SecureDriverRecommender()
    telemetry_input = {
        'gps_speed': 85.5,
        'cTemp': 91,
        'rpm': 2500,
        'eLoad': 70.2,
        'hard_brake': 0.8,
        'total_acceleration': 0.6,
        'angular_acceleration': 0.6
    }
    physiology_input = {
        'Body_Temperature': 98.6,
        'Heart_Rate': 88,
        'SPO2': 95,
    }
    prediction = recommender.predict(telemetry_input, physiology_input)
    
    print("=== Prediction Results ===")
    print(f"Risk Level: {prediction['risk_level']}")
    print("Recommendations:")
    for action in prediction['recommendations']:
        print(f"- {action}")
    print("\nEncrypted Telemetry Data:")
    for key, value in prediction['encrypted_telemetry_data'].items():
        print(f"{key}: {value}")
    print("\nEncrypted Physiology Data:")
    for key, value in prediction['encrypted_physiology_data'].items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    train_telematics_model()
    train_physiological_model()
    test_prediction()