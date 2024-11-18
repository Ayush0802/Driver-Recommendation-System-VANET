import sys
import json
from predict import SecureDriverRecommender

def main():
    # Get input data from command line arguments
    telemetry_data = json.loads(sys.argv[1])
    physiology_data = json.loads(sys.argv[2])
    
    # Initialize recommender
    recommender = SecureDriverRecommender()
    
    # Get prediction
    prediction = recommender.predict(telemetry_data, physiology_data)
    
    # Convert prediction to JSON-serializable format
    result = {
        'risk_level': prediction['risk_level'],
        'recommendations': prediction['recommendations']
    }
    
    print(json.dumps(result))

if __name__ == "__main__":
    main()