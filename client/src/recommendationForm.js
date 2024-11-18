import React, { useState } from 'react';
import axios from 'axios';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  CircularProgress
} from '@mui/material';

const RecommendationForm = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  const [formData, setFormData] = useState({
    telemetry: {
      gps_speed: '',
      cTemp: '',
      rpm: '',
      eLoad: '',
      hard_brake: '',
      total_acceleration: '',
      angular_acceleration: ''
    },
    physiology: {
      Body_Temperature: '',
      Heart_Rate: '',
      SPO2: ''
    }
  });

  const handleInputChange = (category, field, value) => {
    setFormData(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [field]: value
      }
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('http://localhost:5000/api/recommendations/predict', {
        telemetryData: formData.telemetry,
        physiologyData: formData.physiology
      });
      
      setResult(response.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ p: 4, mt: 4 }}>
        <Typography variant="h4" gutterBottom>
          Driver Recommendation System
        </Typography>
        
        <form onSubmit={handleSubmit}>
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Telemetry Data
            </Typography>
            {Object.keys(formData.telemetry).map((field) => (
              <TextField
                key={field}
                label={field.replace(/_/g, ' ').toUpperCase()}
                fullWidth
                margin="normal"
                value={formData.telemetry[field]}
                onChange={(e) => handleInputChange('telemetry', field, e.target.value)}
                type="number"
                required
              />
            ))}
          </Box>

          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Physiological Data
            </Typography>
            {Object.keys(formData.physiology).map((field) => (
              <TextField
                key={field}
                label={field.replace(/_/g, ' ').toUpperCase()}
                fullWidth
                margin="normal"
                value={formData.physiology[field]}
                onChange={(e) => handleInputChange('physiology', field, e.target.value)}
                type="number"
                required
              />
            ))}
          </Box>

          <Button
            variant="contained"
            color="primary"
            type="submit"
            disabled={loading}
            fullWidth
          >
            {loading ? <CircularProgress size={24} /> : 'Get Recommendation'}
          </Button>
        </form>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {result && (
          <Box sx={{ mt: 4 }}>
            <Alert severity={
              result.risk_level === 'Very Low' ? 'success' :
              result.risk_level === 'Low' ? 'success' :
              result.risk_level === 'Low-Medium' ? 'info' :
              result.risk_level === 'Medium' ? 'warning' :
              result.risk_level === 'Medium-High' ? 'warning' :
              result.risk_level === 'High' ? 'error' :
              result.risk_level === 'Very High' ? 'error' :
              'error'
            }>
              <Typography variant="h6">
                Risk Level: {result.risk_level}
              </Typography>
            </Alert>
            
            <Typography variant="h6" sx={{ mt: 2 }}>
              Recommendations:
            </Typography>
            <ul>
              {result.recommendations.map((rec, index) => (
                <li key={index}>
                  <Typography>{rec}</Typography>
                </li>
              ))}
            </ul>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default RecommendationForm;