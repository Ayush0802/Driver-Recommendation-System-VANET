import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tenseal as ts
from sklearn.metrics import accuracy_score
import joblib


class SecureTelemetryTrainer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.context = self.setup_encryption()
        self.models = {}  

    def setup_encryption(self):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context

    def preprocess_data(self, X):
        X = X.fillna(X.mean())
        return X

    def encrypt_data(self, data):
        encrypted_data = []
        for row in data:
            encrypted_row = ts.ckks_vector(self.context, row.tolist())
            encrypted_data.append(encrypted_row)
        return encrypted_data

    def decrypt_prediction(self, encrypted_predictions):
        return [pred.decrypt() for pred in encrypted_predictions]

    def encrypted_logistic_regression(self, encrypted_X, y, lr=0.01, iterations=100):
        
        vector_size = len(encrypted_X[0].decrypt())
        weights = ts.ckks_vector(self.context, np.zeros(vector_size))
        bias = ts.ckks_vector(self.context, [0.0])
        
        decrypted_X = np.array([x.decrypt() for x in encrypted_X])
        encrypted_y = np.array([ts.ckks_vector(self.context, [float(label)]) for label in y])

        for _ in range(iterations):
            gradients = np.zeros(vector_size)
            bias_gradient = 0.0

            for x_decrypted, y_true_enc, x_enc in zip(decrypted_X, encrypted_y, encrypted_X):

                pred = x_enc.dot(weights) + bias
                error = pred - y_true_enc
                error_scalar = error.decrypt()[0]
                gradients += x_decrypted * error_scalar
                bias_gradient += error_scalar

            # Update weights and bias
            weights_update = ts.ckks_vector(self.context, -lr * gradients)
            bias_update = ts.ckks_vector(self.context, [-lr * bias_gradient])

            weights += weights_update
            bias += bias_update

        return weights, bias


    def predict(self, encrypted_X, weights, bias):
        encrypted_predictions = [x.dot(weights) + bias for x in encrypted_X]
        return encrypted_predictions

    def train_model(self, X, y, lr=0.01, iterations=100):
        X = self.preprocess_data(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Encrypt data
        encrypted_X_train = self.encrypt_data(X_train_scaled)
        encrypted_X_test = self.encrypt_data(X_test_scaled)

        # Multiclass classification
        for class_label in np.unique(y):
            y_train_binary = (y_train == class_label).astype(int)
            weights, bias = self.encrypted_logistic_regression(
                encrypted_X_train, y_train_binary, lr, iterations
            )
            self.models[class_label] = (weights, bias)

        # Predict and decrypt for evaluation
        decrypted_predictions = []
        for weights, bias in self.models.values():
            encrypted_predictions = self.predict(encrypted_X_test, weights, bias)
            decrypted_predictions.append(self.decrypt_prediction(encrypted_predictions))

        predictions = np.argmax(decrypted_predictions, axis=0)
        accuracy = accuracy_score(y_test, predictions)

        return accuracy

    def save_model(self, model_path="secure_telemetry_model.joblib"):
        model_data = {
            "models": self.models,
            "scaler": self.scaler,
        }
        joblib.dump(model_data, model_path)

        # Save the encryption context separately
        context_path = model_path.replace(".joblib", "_context.seal")
        with open(context_path, "wb") as f:
            f.write(self.context.serialize())


    def create_risk_labels(self, X):
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

class SecurePhysiologyTrainer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.context = self.setup_encryption()
        self.models = {}  # To store weights and biases for each class

    def setup_encryption(self):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context

    def preprocess_data(self, X):
        X = X.fillna(X.mean())
        return X

    def encrypt_data(self, data):
        encrypted_data = []
        for row in data:
            encrypted_row = ts.ckks_vector(self.context, row.tolist())
            encrypted_data.append(encrypted_row)
        return encrypted_data

    def decrypt_prediction(self, encrypted_predictions):
        return [pred.decrypt() for pred in encrypted_predictions]

    def encrypted_logistic_regression(self, encrypted_X, y, lr=0.01, iterations=100):
        vector_size = len(encrypted_X[0].decrypt())
        
        weights = ts.ckks_vector(self.context, np.zeros(vector_size))
        bias = ts.ckks_vector(self.context, [0.0])
        
        decrypted_X = np.array([x.decrypt() for x in encrypted_X])
        encrypted_y = np.array([ts.ckks_vector(self.context, [float(label)]) for label in y])

        for _ in range(iterations):
            gradients = np.zeros(vector_size)
            bias_gradient = 0.0

            for x_decrypted, y_true_enc, x_enc in zip(decrypted_X, encrypted_y, encrypted_X):

                pred = x_enc.dot(weights) + bias
                error = pred - y_true_enc
                error_scalar = error.decrypt()[0]
                gradients += x_decrypted * error_scalar
                bias_gradient += error_scalar

            # Update weights and bias
            weights_update = ts.ckks_vector(self.context, -lr * gradients)
            bias_update = ts.ckks_vector(self.context, [-lr * bias_gradient])

            weights += weights_update
            bias += bias_update

        return weights, bias

    def predict(self, encrypted_X, weights, bias):
        encrypted_predictions = [x.dot(weights) + bias for x in encrypted_X]
        return encrypted_predictions

    def train_model(self, X, y, lr=0.01, iterations=100):
        X = self.preprocess_data(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Encrypt data
        encrypted_X_train = self.encrypt_data(X_train_scaled)
        encrypted_X_test = self.encrypt_data(X_test_scaled)

        # Multiclass classification
        for class_label in np.unique(y):
            y_train_binary = (y_train == class_label).astype(int)
            weights, bias = self.encrypted_logistic_regression(
                encrypted_X_train, y_train_binary, lr, iterations
            )
            self.models[class_label] = (weights, bias)

        # Predict and decrypt for evaluation
        decrypted_predictions = []
        for weights, bias in self.models.values():
            encrypted_predictions = self.predict(encrypted_X_test, weights, bias)
            decrypted_predictions.append(self.decrypt_prediction(encrypted_predictions))

        predictions = np.argmax(decrypted_predictions, axis=0)
        accuracy = accuracy_score(y_test, predictions)

        return accuracy

    def save_model(self, model_path="secure_physiological_model.joblib"):
        model_data = {
            "models": self.models,
            "scaler": self.scaler,
        }
        joblib.dump(model_data, model_path)

        # Save the encryption context separately
        context_path = model_path.replace(".joblib", "_context.seal")
        with open(context_path, "wb") as f:
            f.write(self.context.serialize())

    def create_risk_labels(self, X):
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
        
        max_possible_score = max(range_config['weight'] for range_config in conditions['Driver_State']['ranges'])
        normalized_scores = (risk_scores / max_possible_score) * 5
        
        risk_categories = pd.cut(normalized_scores,
                               bins=[-np.inf, 1, 2, 3, 4, 5, np.inf],
                               labels=[0, 1, 2, 3, 4, 5])
        
        return risk_categories