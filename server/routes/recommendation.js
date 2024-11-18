const express = require('express');
const router = express.Router();
const { PythonShell } = require('python-shell');
const path = require('path');

router.post('/predict', async (req, res) => {
  try {
    const { telemetryData, physiologyData } = req.body;
    
    console.log(telemetryData, physiologyData);
    let options = {
      mode: 'json',
      pythonPath: 'python3',
      pythonOptions: ['-u'],
      scriptPath: path.join(__dirname, '../python'),
      args: [JSON.stringify(telemetryData), JSON.stringify(physiologyData)]
    };

    PythonShell.run('predict.py', options, function (err, results) {
      if (err) throw err;
      res.json(results[0]);
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;