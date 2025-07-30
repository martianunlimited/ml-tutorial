/*
 * Interactivity for the machine learning tutorial website.
 * This script implements two interactive demos:
 * 1. Classification vs Regression synthetic dataset visualization.
 * 2. Gradient descent training for linear regression.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Demo 1: Classification vs Regression
  const taskSelect = document.getElementById('task-select');
  const taskCanvas = document.getElementById('taskDemoCanvas');
  let taskChart;

  /**
   * Generate a synthetic dataset for classification or regression.
   * For classification, two clusters in 2D; for regression, points on a line with noise.
   * @param {string} type either 'classification' or 'regression'
   * @returns {Object} data with x, y and optional class labels
   */
  function generateDataset(type) {
    const points = [];
    if (type === 'classification') {
      // Two clusters separated along x axis
      for (let i = 0; i < 50; i++) {
        // cluster 1
        points.push({ x: Math.random() * 2 + 1, y: Math.random() * 2 + 1, label: 0 });
        // cluster 2
        points.push({ x: Math.random() * 2 + 4, y: Math.random() * 2 + 4, label: 1 });
      }
    } else {
      // Regression: line y = 2x + noise
      for (let i = 0; i < 100; i++) {
        const x = Math.random() * 5;
        const y = 2 * x + (Math.random() - 0.5) * 2;
        points.push({ x, y });
      }
    }
    return points;
  }

  /**
   * Compute a simple least squares linear regression line for the given points.
   * @param {Array} data points with properties x and y
   * @returns {{m: number, b: number}}
   */
  function computeLeastSquares(data) {
    const n = data.length;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumXX = 0;
    data.forEach(p => {
      sumX += p.x;
      sumY += p.y;
      sumXY += p.x * p.y;
      sumXX += p.x * p.x;
    });
    const m = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const b = (sumY - m * sumX) / n;
    return { m, b };
  }

  /**
   * Render the classification/regression chart using Chart.js.
   */
  function renderTaskChart() {
    const type = taskSelect.value;
    const data = generateDataset(type);
    // Prepare datasets for Chart.js
    let datasets;
    if (type === 'classification') {
      const class0 = data.filter(p => p.label === 0).map(p => ({ x: p.x, y: p.y }));
      const class1 = data.filter(p => p.label === 1).map(p => ({ x: p.x, y: p.y }));
      datasets = [
        {
          label: 'Class 0',
          data: class0,
          backgroundColor: 'rgba(66, 165, 245, 0.7)',
        },
        {
          label: 'Class 1',
          data: class1,
          backgroundColor: 'rgba(239, 83, 80, 0.7)',
        },
      ];
    } else {
      // Compute line of best fit
      const { m, b } = computeLeastSquares(data);
      // Create line samples for Chart.js line dataset
      const lineSamples = [];
      // Sort x for line plotting
      const sortedX = data.map(p => p.x).sort((a, b) => a - b);
      sortedX.forEach(x => {
        lineSamples.push({ x, y: m * x + b });
      });
      datasets = [
        {
          label: 'Data points',
          data: data.map(p => ({ x: p.x, y: p.y })),
          backgroundColor: 'rgba(255, 167, 38, 0.7)',
        },
        {
          label: 'Least squares fit',
          data: lineSamples,
          type: 'line',
          borderColor: 'rgba(102, 187, 106, 0.9)',
          borderWidth: 2,
          fill: false,
          pointRadius: 0,
        },
      ];
    }
    // Destroy existing chart if present
    if (taskChart) {
      taskChart.destroy();
    }
    taskChart = new Chart(taskCanvas.getContext('2d'), {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'linear',
            position: 'bottom',
            title: { display: true, text: 'x' },
            grid: { color: 'rgba(0,0,0,0.1)' },
          },
          y: {
            title: { display: true, text: 'y' },
            grid: { color: 'rgba(0,0,0,0.1)' },
          },
        },
        plugins: {
          legend: { display: true },
        },
      },
    });
  }

  // Initialize the first plot
  renderTaskChart();
  // Update chart when task selection changes
  taskSelect.addEventListener('change', renderTaskChart);

  // Demo 2: Linear regression gradient descent
  const lrCanvas = document.getElementById('linRegCanvas');
  const lrSlider = document.getElementById('lr-slider');
  const lrValueSpan = document.getElementById('lr-value');
  const trainButton = document.getElementById('train-button');
  let lrChart;
  // Generate training data for gradient descent
  const trainingData = [];
  for (let i = 0; i < 50; i++) {
    const x = Math.random() * 5;
    const y = 3 * x + 1 + (Math.random() - 0.5) * 4; // underlying slope 3, intercept 1 with noise
    trainingData.push({ x, y });
  }
  // Initialize model parameters
  let currentM = 0;
  let currentB = 0;

  /**
   * Perform one step of gradient descent on currentM and currentB.
   * @param {number} lr learning rate
   */
  function trainOneStep(lr) {
    let mGrad = 0;
    let bGrad = 0;
    const n = trainingData.length;
    trainingData.forEach(p => {
      const yPred = currentM * p.x + currentB;
      const error = yPred - p.y;
      mGrad += (2 / n) * error * p.x;
      bGrad += (2 / n) * error;
    });
    currentM -= lr * mGrad;
    currentB -= lr * bGrad;
  }

  /**
   * Render the linear regression chart including training data and current model line.
   */
  function renderLrChart() {
    // Prepare datasets
    const linePoints = [];
    const sortedX = trainingData.map(p => p.x).sort((a, b) => a - b);
    sortedX.forEach(x => {
      linePoints.push({ x, y: currentM * x + currentB });
    });
    const datasets = [
      {
        label: 'Training data',
        data: trainingData.map(p => ({ x: p.x, y: p.y })),
        backgroundColor: 'rgba(129, 212, 250, 0.7)',
      },
      {
        label: 'Model prediction',
        data: linePoints,
        type: 'line',
        borderColor: 'rgba(244, 143, 177, 0.9)',
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
      },
    ];
    if (lrChart) {
      lrChart.destroy();
    }
    lrChart = new Chart(lrCanvas.getContext('2d'), {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'linear',
            position: 'bottom',
            title: { display: true, text: 'x' },
            grid: { color: 'rgba(0,0,0,0.1)' },
          },
          y: {
            title: { display: true, text: 'y' },
            grid: { color: 'rgba(0,0,0,0.1)' },
          },
        },
        plugins: { legend: { display: true } },
      },
    });
  }
  // Initial render
  renderLrChart();

  // Update displayed learning rate when slider changes
  lrSlider.addEventListener('input', () => {
    lrValueSpan.textContent = parseFloat(lrSlider.value).toFixed(3);
  });
  // Train one step on button click
  trainButton.addEventListener('click', () => {
    const lr = parseFloat(lrSlider.value);
    trainOneStep(lr);
    renderLrChart();
  });
});