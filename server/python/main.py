from train import SecureTelemetryTrainer, SecurePhysiologyTrainer
import pandas as pd

def train_telematics_model():
    print("=== Training Telematics Model ===")
    trainer = SecureTelemetryTrainer()
    telematics_df = pd.read_csv('telematics.csv')
    X_telematics = telematics_df[['gps_speed', 'cTemp', 'rpm', 'eLoad', 'hard_brake', 'total_acceleration', 'angular_acceleration']]
    y_telematics = trainer.create_risk_labels(X_telematics)
    train_score, test_score = trainer.train_model(X_telematics, y_telematics)
    print(f"Telematics Training Score: {train_score:.4f}")
    print(f"Telematics Testing Score: {test_score:.4f}")
    trainer.save_model()

def train_physiological_model():
    print("=== Training Physiological Model ===")
    trainer = SecurePhysiologyTrainer()
    physiological_df = pd.read_csv('physiological.csv')
    X_physiological = physiological_df[['Body_Temperature', 'Heart_Rate', 'SPO2']]
    y_physiological = trainer.create_risk_labels(physiological_df[['Driver_State']])
    train_score, test_score = trainer.train_model(X_physiological, y_physiological)
    print(f"Physiological Training Score: {train_score:.4f}")
    print(f"Physiological Testing Score: {test_score:.4f}")
    trainer.save_model()

if __name__ == "__main__":
    train_telematics_model()
    train_physiological_model()