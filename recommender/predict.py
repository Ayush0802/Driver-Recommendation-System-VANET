import pandas as pd
import tenseal as ts
import joblib

class SecureDriverRecommender:
    def __init__(self, telemetry_model_path='secure_telemetry_model.joblib', physiology_model_path='secure_physiology_model.joblib'):
        # Load models and contexts
        self.load_models(telemetry_model_path, physiology_model_path)
        self.setup_recommendations()
        
    def load_models(self, telemetry_path, physiology_path):
        # Load telemetry model
        telemetry_data = joblib.load(telemetry_path)
        self.telemetry_model = telemetry_data['model']
        self.telemetry_scaler = telemetry_data['scaler']
        
        # Load physiology model
        physiology_data = joblib.load(physiology_path)
        self.physiology_model = physiology_data['model']
        self.physiology_scaler = physiology_data['scaler']
        
        # Load encryption contexts
        with open(telemetry_path.replace('.joblib', '_context.seal'), 'rb') as f:
            self.context = ts.Context.load(f.read())
        
    def setup_recommendations(self):
        self.recommendations = {
            0: {
                'risk_level': 'Very Low',
                'actions': [
                    "Excellent driving conditions",
                    "Continue safe driving behavior",
                    "All metrics within optimal range"
                ]
            },
            1: {
                'risk_level': 'Low',
                'actions': [
                    "Good driving conditions",
                    "Maintain current driving behavior",
                    "Consider taking a break in the next 2 hours"
                ]
            },
            2: {
                'risk_level': 'Medium',
                'actions': [
                    "Moderate risk detected",
                    "Take a short break in the next 30 minutes",
                    "Ensure proper hydration",
                    "Adjust driving speed and style",
                    "Monitor fatigue levels"
                ]
            },
            3: {
                'risk_level': 'Medium-High',
                'actions': [
                    "Elevated risk detected",
                    "Plan for a break within 15 minutes",
                    "Reduce speed",
                    "Increase following distance",
                    "Check vital signs",
                    "Assess weather and road conditions"
                ]
            },
            4: {
                'risk_level': 'High',
                'actions': [
                    "CAUTION: High risk detected",
                    "Find safe location to pull over",
                    "Take a 20-minute rest break",
                    "Check vital signs thoroughly",
                    "Assess driver fatigue level",
                    "Review vehicle telemetry"
                ]
            },
            # 6: {
            #     'risk_level': 'Very High',
            #     'actions': [
            #         "WARNING: Very high risk detected",
            #         "Pull over at next safe location",
            #         "Minimum 30-minute rest required",
            #         "Check all vital signs",
            #         "Vehicle inspection recommended",
            #         "Consider alternate driver if available",
            #         "Reassess weather and road conditions"
            #     ]
            # },
            5: {
                'risk_level': 'Critical',
                'actions': [
                    "EMERGENCY: Critical risk level",
                    "Stop driving immediately",
                    "Seek immediate medical attention",
                    "Contact emergency services if needed",
                    "Do not resume driving",
                    "Comprehensive vehicle inspection required",
                    "Mandatory driver assessment needed"
                ]
            }
        }

    def encrypt_data(self, data):
        return {k: ts.ckks_vector(self.context, [float(v)]) 
                for k, v in data.items()}

    def predict(self, telemetry_data, physiology_data):
        # Ensure feature names and order match those used during training
        telemetry_input_data = {
            'gps_speed': telemetry_data['gps_speed'],
            'cTemp': telemetry_data['cTemp'],
            'rpm': telemetry_data['rpm'],
            'eLoad': telemetry_data['eLoad'],
            'hard_brake': telemetry_data['hard_brake'],
            'total_acceleration': telemetry_data['total_acceleration'],
            'angular_acceleration': telemetry_data['angular_acceleration']
        }
        
        physiology_input_data = {
            'Body_Temperature': physiology_data['Body_Temperature'],
            'Heart_Rate': physiology_data['Heart_Rate'],
            'SPO2': physiology_data['SPO2']
        }
        
        telemetry_input_df = pd.DataFrame([telemetry_input_data])
        physiology_input_df = pd.DataFrame([physiology_input_data])
        
        telemetry_input_scaled = self.telemetry_scaler.transform(telemetry_input_df)
        physiology_input_scaled = self.physiology_scaler.transform(physiology_input_df)
        
        encrypted_telemetry_input = self.encrypt_data(
            dict(zip(telemetry_input_data.keys(), telemetry_input_scaled[0]))
        )
        
        encrypted_physiology_input = self.encrypt_data(
            dict(zip(physiology_input_data.keys(), physiology_input_scaled[0]))
        )
        
        # Optimized risk level combination logic
        telemetry_risk_level = self.telemetry_model.predict(telemetry_input_scaled)[0]
        physiology_risk_level = self.physiology_model.predict(physiology_input_scaled)[0]

        combined_risk_level = max(telemetry_risk_level, physiology_risk_level)

        # Additional weight for combined risks
        if telemetry_risk_level >= 3 and physiology_risk_level >= 3:
            combined_risk_level = min(5, combined_risk_level + 1)
        
        recommendation = self.recommendations[combined_risk_level]
        
        return {
            'risk_level': recommendation['risk_level'],
            'recommendations': recommendation['actions'],
            'encrypted_telemetry_data': encrypted_telemetry_input,
            'encrypted_physiology_data': encrypted_physiology_input
        }