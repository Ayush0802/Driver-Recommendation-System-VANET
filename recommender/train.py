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
            encrypted_row = ts.ckks_vector(self.context, row.tolist())
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
        encrypted_X_test = self.encrypt_data(X_test_scaled)
    
        self.model.fit(X_train_scaled, y_train)
        # self.model.fit(encrypted_X_train, y_train)
        
        return self.model.score(X_train_scaled, y_train), self.model.score(X_test_scaled, y_test)
        # return self.model.score(encrypted_X_train, y_train), self.model.score(encrypted_X_test, y_test)

    def create_risk_labels(self, X):
        """Create 6-level risk labels based on telemetry data"""
        risk_scores = np.zeros(len(X))
        
        conditions = {
            'gps_speed': {
                'ranges': [
                    {'range': (0, 35), 'weight': 0},      # Very safe speed
                    {'range': (35, 60), 'weight': 0.6},   # Safe speed
                    {'range': (60, 80), 'weight': 1.2},   # Moderate speed
                    {'range': (80, float('inf')), 'weight': 2.0}  # High speed
                ]
            },
            'rpm': {
                'ranges': [
                    {'range': (0, 1800), 'weight': 0},    # Normal RPM
                    {'range': (1800, 2800), 'weight': 1.0},  # Moderate RPM
                    {'range': (2800, float('inf')), 'weight': 2.0}  # High RPM
                ]
            },
            'hard_brake': {
                'ranges': [
                    {'range': (0, 0.4), 'weight': 0},     # Normal braking
                    {'range': (0.4, 0.7), 'weight': 1.0}, # Moderate braking
                    {'range': (0.7, 1.0), 'weight': 2.0}  # Hard braking
                ]
            },
            'total_acceleration': {
                'ranges': [
                    {'range': (0, 0.4), 'weight': 0},     # Normal acceleration
                    {'range': (0.4, 0.7), 'weight': 1.0}, # Moderate acceleration
                    {'range': (0.7, 1.0), 'weight': 2.0}  # High acceleration
                ]
            },
            'angular_acceleration': {
                'ranges': [
                    {'range': (0, 0.4), 'weight': 0},     # Normal turning
                    {'range': (0.4, 0.7), 'weight': 1.0}, # Moderate turning
                    {'range': (0.7, 1.0), 'weight': 2.0}  # Sharp turning
                ]
            }
        }
        
        # Calculate risk scores based on conditions
        for feature, config in conditions.items():
            feature_values = X[feature].values
            for range_config in config['ranges']:
                mask = (feature_values > range_config['range'][0]) & (feature_values <= range_config['range'][1])
                risk_scores[mask] += range_config['weight']
        
        # Normalize risk scores to 0-5 range and create categories
        max_possible_score = sum(max(range_config['weight'] for range_config in config['ranges']) 
                               for config in conditions.values())
        normalized_scores = (risk_scores / max_possible_score) * 5
        
        # Create 6 risk categories (0-5)
        risk_categories = pd.cut(normalized_scores,
                               bins=[-np.inf, 1, 2, 3, 4, 5, np.inf],
                               labels=[0, 1, 2, 3, 4, 5])
        
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
            encrypted_row = ts.ckks_vector(self.context, row.tolist())
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
        """Create 6-level risk labels based on driver state"""
        risk_scores = np.zeros(len(X))
        
        conditions = {
            'Driver_State': {
                'ranges': [
                    {'range': (0, 2.5), 'weight': 0},      # Alert
                    {'range': (2.5, 5), 'weight': 1.0},    # Mild fatigue
                    {'range': (5, 7.5), 'weight': 1.5},    # Moderate fatigue
                    {'range': (7.5, 10), 'weight': 2.0}    # Severe fatigue
                ]
            }
        }
        
        # Calculate risk scores based on driver state
        feature_values = X['Driver_State'].values
        for range_config in conditions['Driver_State']['ranges']:
            mask = (feature_values > range_config['range'][0]) & (feature_values <= range_config['range'][1])
            risk_scores[mask] += range_config['weight']
        
        # Normalize risk scores to 0-5 range and create categories
        max_possible_score = max(range_config['weight'] for range_config in conditions['Driver_State']['ranges'])
        normalized_scores = (risk_scores / max_possible_score) * 5
        
        # Create 6 risk categories (0-5)
        risk_categories = pd.cut(normalized_scores,
                               bins=[-np.inf, 1, 2, 3, 4, 5, np.inf],
                               labels=[0, 1, 2, 3, 4, 5])
        
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