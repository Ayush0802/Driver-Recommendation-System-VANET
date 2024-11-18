import React from 'react';
import RecommendationForm from './recommendationForm.js';
import { createTheme, ThemeProvider, CssBaseline } from '@mui/material';

const theme = createTheme({
  palette: {
    mode: 'light',
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <RecommendationForm />
    </ThemeProvider>
  );
}

export default App;