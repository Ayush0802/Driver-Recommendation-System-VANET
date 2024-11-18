import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tenseal as ts
import joblib

class SecureTelemetryTrainer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.context = self.setup_encryption()
    
    def setup_encryption(self):
        """Setup the encryption context for secure computation"""
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context

    def preprocess_data(self, X):
        """Preprocess the dataset"""
        X = X.fillna(X.mean())
        return X

    def encrypt_data(self, data):
        """Encrypt input data using homomorphic encryption"""
        encrypted_data = []
        for row in data:
            encrypted_row = [ts.ckks_vector(self.context, [float(val)]) for val in row]
            encrypted_data.append(encrypted_row)
        return encrypted_data

    def train_model(self, X, y):
        """Train the telemetry model with encryption support"""
        X = self.preprocess_data(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encrypt training data
        encrypted_X_train = self.encrypt_data(X_train_scaled)
        
        self.model.fit(X_train_scaled, y_train)
        
        return self.model.score(X_train_scaled, y_train), self.model.score(X_test_scaled, y_test)

    def create_risk_labels(self, X):
        """Create 8-level risk labels based on telemetry data"""
        risk_scores = np.zeros(len(X))
        
        conditions = {
            'gps_speed': {
                'ranges': [
                    {'range': (0, 30), 'weight': 0},      # Very safe speed
                    {'range': (30, 50), 'weight': 0.5},   # Safe speed
                    {'range': (50, 70), 'weight': 1.0},   # Moderate speed
                    {'range': (70, 85), 'weight': 1.5},   # High speed
                    {'range': (85, float('inf')), 'weight': 2.0}  # Very high speed
                ]
            },
            'rpm': {
                'ranges': [
                    {'range': (0, 1500), 'weight': 0},    # Normal RPM
                    {'range': (1500, 2500), 'weight': 1.0},  # Moderate RPM
                    {'range': (2500, 3500), 'weight': 1.5},  # High RPM
                    {'range': (3500, float('inf')), 'weight': 2.0}  # Very high RPM
                ]
            },
            'hard_brake': {
                'ranges': [
                    {'range': (0, 0.3), 'weight': 0},     # Normal braking
                    {'range': (0.3, 0.6), 'weight': 1.0}, # Moderate braking
                    {'range': (0.6, 0.8), 'weight': 1.5}, # Hard braking
                    {'range': (0.8, 1.0), 'weight': 2.0}  # Very hard braking
                ]
            },
            'total_acceleration': {
                'ranges': [
                    {'range': (0, 0.3), 'weight': 0},     # Normal acceleration
                    {'range': (0.3, 0.6), 'weight': 1.0}, # Moderate acceleration
                    {'range': (0.6, 0.8), 'weight': 1.5}, # High acceleration
                    {'range': (0.8, 1.0), 'weight': 2.0}  # Very high acceleration
                ]
            },
            'angular_acceleration': {
                'ranges': [
                    {'range': (0, 0.3), 'weight': 0},     # Normal turning
                    {'range': (0.3, 0.6), 'weight': 1.0}, # Moderate turning
                    {'range': (0.6, 0.8), 'weight': 1.5}, # Sharp turning
                    {'range': (0.8, 1.0), 'weight': 2.0}  # Very sharp turning
                ]
            }
        }
        
        # Calculate risk scores based on conditions
        for feature, config in conditions.items():
            feature_values = X[feature].values
            for range_config in config['ranges']:
                mask = (feature_values > range_config['range'][0]) & (feature_values <= range_config['range'][1])
                risk_scores[mask] += range_config['weight']
        
        # Normalize risk scores to 0-7 range and create categories
        max_possible_score = sum(max(range_config['weight'] for range_config in config['ranges']) 
                               for config in conditions.values())
        normalized_scores = (risk_scores / max_possible_score) * 7
        
        # Create 8 risk categories (0-7)
        risk_categories = pd.cut(normalized_scores,
                               bins=[-np.inf, 1, 2, 3, 4, 5, 6, 7, np.inf],
                               labels=[0, 1, 2, 3, 4, 5, 6, 7])
        
        return risk_categories

    def save_model(self, model_path='secure_telemetry_model.joblib'):
        """Save the telemetry model and context"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, model_path)
            
            context_path = model_path.replace('.joblib', '_context.seal')
            with open(context_path, 'wb') as f:
                f.write(self.context.serialize())
                
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")

class SecurePhysiologyTrainer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.context = self.setup_encryption()
    
    def setup_encryption(self):
        """Setup the encryption context for secure computation"""
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context

    def preprocess_data(self, X):
        """Preprocess the dataset"""
        X = X.fillna(X.mean())
        return X
    
    def encrypt_data(self, data):
        """Encrypt input data using homomorphic encryption"""
        encrypted_data = []
        for row in data:
            encrypted_row = [ts.ckks_vector(self.context, [float(val)]) for val in row]
            encrypted_data.append(encrypted_row)
        return encrypted_data

    def train_model(self, X, y):
        """Train the physiology model with encryption support"""
        X = self.preprocess_data(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        encrypted_X_train = self.encrypt_data(X_train_scaled)
        
        self.model.fit(X_train_scaled, y_train)
        
        return self.model.score(X_train_scaled, y_train), self.model.score(X_test_scaled, y_test)

    def create_risk_labels(self, X):
        """Create 8-level risk labels based on driver state"""
        risk_scores = np.zeros(len(X))
        
        conditions = {
            'Driver_State': {
                'ranges': [
                    {'range': (0, 3), 'weight': 0},      # Alert
                    {'range': (3, 5), 'weight': 1.0},    # Mild fatigue
                    {'range': (5, 7), 'weight': 1.5},    # Moderate fatigue
                    {'range': (7, 10), 'weight': 2.0}    # Severe fatigue
                ]
            }
        }
        
        # Calculate risk scores based on driver state
        feature_values = X['Driver_State'].values
        for range_config in conditions['Driver_State']['ranges']:
            mask = (feature_values > range_config['range'][0]) & (feature_values <= range_config['range'][1])
            risk_scores[mask] += range_config['weight']
        
        # Normalize risk scores to 0-7 range and create categories
        max_possible_score = max(range_config['weight'] for range_config in conditions['Driver_State']['ranges'])
        normalized_scores = (risk_scores / max_possible_score) * 7
        
        # Create 8 risk categories (0-7)
        risk_categories = pd.cut(normalized_scores,
                               bins=[-np.inf, 1, 2, 3, 4, 5, 6, 7, np.inf],
                               labels=[0, 1, 2, 3, 4, 5, 6, 7])
        
        return risk_categories

    def save_model(self, model_path='secure_physiology_model.joblib'):
        """Save the physiology model and context"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, model_path)
            
            context_path = model_path.replace('.joblib', '_context.seal')
            with open(context_path, 'wb') as f:
                f.write(self.context.serialize())
                
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")