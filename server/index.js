const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const path = require('path');
const { spawn } = require('child_process');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5001;

app.use(cors());
app.use(express.json());

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
            '-u',  
            path.join(__dirname, './python/predict.py'),
            JSON.stringify(telemetryData),
            JSON.stringify(physiologyData)
        ]);

        let dataString = '';
        let errorString = '';

        pythonProcess.stdout.on('data', (data) => {
            dataString += data.toString();
            try {
                const jsonData = JSON.parse(dataString);
                console.log('Parsed data:', jsonData);
                
                const recommendations = Array.isArray(jsonData.actions) 
                    ? jsonData.actions 
                    : [jsonData.actions];

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

app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something broke!' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Python script path: ${path.join(__dirname, 'python')}`);
});