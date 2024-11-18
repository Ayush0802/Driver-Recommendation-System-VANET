const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const path = require('path');
const { spawn } = require('child_process');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Add logging middleware
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path}`);
  next();
});

app.post('/api/recommendations/predict', async (req, res) => {
    try {
        const { telemetryData, physiologyData } = req.body;
        console.log(telemetryData);

        // Spawn Python process
        const pythonProcess = spawn('python', [
            '-u',  // Equivalent to pythonOptions: ['-u'] - unbuffered output
            path.join(__dirname, './python/predict.py'),
            JSON.stringify(telemetryData),
            JSON.stringify(physiologyData)
        ]);

        let dataString = '';
        let errorString = '';

        // Collect data from script
        pythonProcess.stdout.on('data', (data) => {
            dataString += data.toString();
            try {
                // Try to parse the accumulated data as JSON
                const jsonData = JSON.parse(dataString);
                console.log('Parsed data:', jsonData);
                
                // Convert recommendations to array if it's not already
                const recommendations = Array.isArray(jsonData.actions) 
                    ? jsonData.actions 
                    : [jsonData.actions];

                // Send response with proper structure
                console.log(recommendations);
                res.json({
                    risk_level: jsonData.risk_level,
                    recommendations: jsonData.actions
                });
            } catch (parseError) {
                console.log(0);
            }
        });

    } catch (error) {
        console.error('Server Error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something broke!' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Python script path: ${path.join(__dirname, 'python')}`);
});