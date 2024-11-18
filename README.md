# Driver Recommendation System using VANET

A secure driver recommendation system that analyzes telemetry and physiological data to provide real-time safety recommendations using VANET (Vehicular Ad-hoc Network) technology.

## Features

- Real-time driver risk assessment
- Encrypted data processing using TenSEAL
- Combined analysis of vehicle telemetry and driver physiological data
- 6-level risk classification system
- Secure machine learning models
- React-based user interface
- RESTful API backend

## Technology Stack

### Frontend
- React.js
- Material-UI (MUI)
- Axios

### Backend
- Node.js
- Express.js
- Python (Machine Learning)

### Machine Learning & Security
- scikit-learn
- TenSEAL (Homomorphic Encryption)
- Pandas
- NumPy

## System Architecture

The system consists of three main components:

1. **Machine Learning Models**
   - Telemetry Model: Analyzes vehicle data
   - Physiology Model: Processes driver's biological metrics

2. **Backend Server**
   - RESTful API endpoints
   - Python integration for ML predictions

3. **Frontend Interface**
   - Real-time data input
   - Recommendation display for Driver

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Driver-Recommendation-System-VANET.git
```

2. Install backend dependencies:
```bash
cd server
npm install
```

3. Install frontend dependencies:
```bash
cd client
npm install
```
 
4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the backend server:
```bash
cd server
npm start

2. Start the frontend application:
```bash
cd client
npm start
```

3. Access the application at `http://localhost:3000`

## Input Parameters

### Telemetry Data
- GPS Speed
- Coolant Temperature
- RPM
- Engine Load
- Hard Brake
- Total Acceleration
- Angular Acceleration

### Physiological Data
- Body Temperature
- Heart Rate
- SPO2 (Blood Oxygen Level)

## Risk Levels

The system classifies risk into six levels:
- Very Low
- Low
- Medium
- Medium-High
- High
- Critical

## Security Features

- Homomorphic encryption using TenSEAL
- Secure data transmission
- Encrypted model predictions
- Protected sensitive driver information
