/*
 * Custom interactive demos for the machine learning primer.
 *
 * This script implements two demos without relying on any external
 * JavaScript libraries.  The first demo visualises linear regression
 * on a synthetic dataset.  Users can adjust the slope and intercept
 * of a line via sliders and immediately see how the fitted line
 * changes along with the mean squared error (MSE).  The second demo
 * illustrates gradient descent on the simple loss function J(w) = w².
 * The learning rate slider influences the size of the steps along the
 * descent path.  Both demos are drawn using the HTML Canvas API so
 * they work offline with no dependencies.
 */

document.addEventListener('DOMContentLoaded', () => {
    /* ------------------------------------------------------------------
     * Linear Regression Demo
     *
     * We generate a set of 30 noisy points sampled from a line with
     * true slope 1.5 and intercept -2.  Two sliders control the
     * slope (m) and intercept (b) of a candidate line.  On each
     * interaction we redraw the scatter plot, candidate line and
     * compute the mean squared error between the data and the line.
     */
    const regCanvas = document.getElementById('regressionChart');
    if (regCanvas) {
        const ctx = regCanvas.getContext('2d');
        // Generate synthetic data once
        const nPoints = 30;
        const xs = [];
        const ys = [];
        for (let i = 0; i < nPoints; i++) {
            const xVal = (i / (nPoints - 1)) * 10; // evenly spaced 0–10
            const trueY = 1.5 * xVal - 2;
            const noise = (Math.random() - 0.5) * 3; // small random noise
            xs.push(xVal);
            ys.push(trueY + noise);
        }

        // Slider elements
        const slopeSlider = document.getElementById('slopeSlider');
        const interceptSlider = document.getElementById('interceptSlider');
        const slopeValue = document.getElementById('slopeValue');
        const interceptValue = document.getElementById('interceptValue');
        const mseValue = document.getElementById('mseValue');

        // Compute mean squared error for a given line
        function computeMSE(m, b) {
            let err = 0;
            for (let i = 0; i < xs.length; i++) {
                const pred = m * xs[i] + b;
                const diff = pred - ys[i];
                err += diff * diff;
            }
            return err / xs.length;
        }

        // Draw the scatter plot and regression line
        function drawRegression(m, b) {
            const canvasWidth = regCanvas.width;
            const canvasHeight = regCanvas.height;
            // Clear canvas
            ctx.clearRect(0, 0, canvasWidth, canvasHeight);
            // Define margins around the plot area
            const margin = 40;
            const plotWidth = canvasWidth - margin * 2;
            const plotHeight = canvasHeight - margin * 2;
            // Compute data extents including line end points
            const xMin = Math.min(...xs);
            const xMax = Math.max(...xs);
            // Include predicted endpoints to keep line visible
            const yPredMin = m * xMin + b;
            const yPredMax = m * xMax + b;
            const yMinData = Math.min(...ys);
            const yMaxData = Math.max(...ys);
            // Add a small padding to the y-range for visual comfort
            const yMin = Math.min(yMinData, yPredMin) - 1;
            const yMax = Math.max(yMaxData, yPredMax) + 1;
            // Mapping functions
            const xScale = x => margin + ((x - xMin) / (xMax - xMin)) * plotWidth;
            const yScale = y => margin + plotHeight - ((y - yMin) / (yMax - yMin)) * plotHeight;
            // Draw axes
            ctx.strokeStyle = '#cccccc';
            ctx.lineWidth = 1;
            ctx.beginPath();
            // y-axis
            ctx.moveTo(margin, margin);
            ctx.lineTo(margin, margin + plotHeight);
            // x-axis
            ctx.lineTo(margin + plotWidth, margin + plotHeight);
            ctx.stroke();
            // Optional tick marks on axes
            const numTicks = 5;
            ctx.fillStyle = '#666666';
            ctx.font = '10px Arial';
            for (let i = 0; i <= numTicks; i++) {
                // X ticks
                const tx = xMin + (i / numTicks) * (xMax - xMin);
                const px = xScale(tx);
                ctx.beginPath();
                ctx.moveTo(px, margin + plotHeight);
                ctx.lineTo(px, margin + plotHeight + 5);
                ctx.stroke();
                ctx.fillText(tx.toFixed(1), px - 10, margin + plotHeight + 15);
                // Y ticks
                const ty = yMin + (i / numTicks) * (yMax - yMin);
                const py = yScale(ty);
                ctx.beginPath();
                ctx.moveTo(margin - 5, py);
                ctx.lineTo(margin, py);
                ctx.stroke();
                ctx.fillText(ty.toFixed(1), margin - 35, py + 3);
            }
            // Draw scatter points
            ctx.fillStyle = 'rgba(0, 123, 255, 0.8)';
            for (let i = 0; i < xs.length; i++) {
                const px = xScale(xs[i]);
                const py = yScale(ys[i]);
                ctx.beginPath();
                ctx.arc(px, py, 3, 0, 2 * Math.PI);
                ctx.fill();
            }
            // Draw regression line
            ctx.strokeStyle = 'rgba(220, 53, 69, 0.8)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            const x1 = xMin;
            const y1 = m * x1 + b;
            const x2 = xMax;
            const y2 = m * x2 + b;
            ctx.moveTo(xScale(x1), yScale(y1));
            ctx.lineTo(xScale(x2), yScale(y2));
            ctx.stroke();
        }

        // Update function when sliders change
        function updateRegression() {
            const m = parseFloat(slopeSlider.value);
            const b = parseFloat(interceptSlider.value);
            slopeValue.textContent = m.toFixed(2);
            interceptValue.textContent = b.toFixed(2);
            // Redraw plot
            drawRegression(m, b);
            // Compute and display MSE
            const mse = computeMSE(m, b);
            mseValue.textContent = mse.toFixed(3);
        }

        // Initial draw
        updateRegression();
        // Event listeners
        slopeSlider.addEventListener('input', updateRegression);
        interceptSlider.addEventListener('input', updateRegression);
    }

    /* ------------------------------------------------------------------
     * Gradient Descent Demo
     *
     * Visualises gradient descent on the simple quadratic function
     * J(w) = w².  A slider controls the learning rate alpha.  We start
     * from a fixed weight w0 = 3.5 and perform several descent steps.
     * Both the curve and the descent path are plotted on a canvas.
     */
    const gdCanvas = document.getElementById('gdChart');
    if (gdCanvas) {
        const gdCtx = gdCanvas.getContext('2d');
        const lrSlider = document.getElementById('lrSlider');
        const lrValue = document.getElementById('lrValue');
        // Precompute curve samples for J(w) = w^2
        const curveXs = [];
        const curveYs = [];
        for (let w = -4; w <= 4; w += 0.05) {
            curveXs.push(w);
            curveYs.push(w * w);
        }
        // Function to compute gradient descent path given a learning rate
        function computePath(alpha) {
            const path = [];
            let w = 3.5;
            const steps = 15;
            for (let i = 0; i < steps; i++) {
                const loss = w * w;
                path.push({ w, loss });
                const grad = 2 * w;
                w = w - alpha * grad;
                if (!isFinite(w)) break;
            }
            // Add final point
            path.push({ w, loss: w * w });
            return path;
        }
        // Draw gradient descent demo
        function drawGradientDescent(alpha) {
            const cw = gdCanvas.width;
            const ch = gdCanvas.height;
            // Clear canvas
            gdCtx.clearRect(0, 0, cw, ch);
            // Define plot area
            const margin = 40;
            const plotW = cw - margin * 2;
            const plotH = ch - margin * 2;
            // Range for w and J
            const xMin = -4;
            const xMax = 4;
            const yMin = 0;
            const yMax = 16;
            // Mapping functions
            const xScale = x => margin + ((x - xMin) / (xMax - xMin)) * plotW;
            const yScale = y => margin + plotH - ((y - yMin) / (yMax - yMin)) * plotH;
            // Draw axes
            gdCtx.strokeStyle = '#cccccc';
            gdCtx.lineWidth = 1;
            gdCtx.beginPath();
            // y-axis
            gdCtx.moveTo(margin, margin);
            gdCtx.lineTo(margin, margin + plotH);
            // x-axis
            gdCtx.lineTo(margin + plotW, margin + plotH);
            gdCtx.stroke();
            // Ticks and labels
            const ticks = 4;
            gdCtx.fillStyle = '#666666';
            gdCtx.font = '10px Arial';
            for (let i = 0; i <= ticks; i++) {
                const tx = xMin + (i / ticks) * (xMax - xMin);
                const px = xScale(tx);
                // x-axis ticks
                gdCtx.beginPath();
                gdCtx.moveTo(px, margin + plotH);
                gdCtx.lineTo(px, margin + plotH + 5);
                gdCtx.stroke();
                gdCtx.fillText(tx.toFixed(1), px - 10, margin + plotH + 15);
                // y-axis ticks
                const ty = yMin + (i / ticks) * (yMax - yMin);
                const py = yScale(ty);
                gdCtx.beginPath();
                gdCtx.moveTo(margin - 5, py);
                gdCtx.lineTo(margin, py);
                gdCtx.stroke();
                gdCtx.fillText(ty.toFixed(0), margin - 30, py + 3);
            }
            // Draw the curve J(w)=w^2
            gdCtx.strokeStyle = 'rgba(40, 167, 69, 0.9)';
            gdCtx.lineWidth = 2;
            gdCtx.beginPath();
            for (let i = 0; i < curveXs.length; i++) {
                const px = xScale(curveXs[i]);
                const py = yScale(curveYs[i]);
                if (i === 0) {
                    gdCtx.moveTo(px, py);
                } else {
                    gdCtx.lineTo(px, py);
                }
            }
            gdCtx.stroke();
            // Compute and draw descent path
            const path = computePath(alpha);
            // Draw path lines
            gdCtx.strokeStyle = 'rgba(255, 193, 7, 0.9)';
            gdCtx.lineWidth = 2;
            gdCtx.beginPath();
            for (let i = 0; i < path.length; i++) {
                const px = xScale(path[i].w);
                const py = yScale(path[i].loss);
                if (i === 0) {
                    gdCtx.moveTo(px, py);
                } else {
                    gdCtx.lineTo(px, py);
                }
            }
            gdCtx.stroke();
            // Draw points on the path
            gdCtx.fillStyle = 'rgba(255, 193, 7, 0.9)';
            for (const point of path) {
                const px = xScale(point.w);
                const py = yScale(point.loss);
                gdCtx.beginPath();
                gdCtx.arc(px, py, 3, 0, 2 * Math.PI);
                gdCtx.fill();
            }
        }
        // Update function when slider changes
        function updateGradient() {
            const alpha = parseFloat(lrSlider.value);
            lrValue.textContent = alpha.toFixed(2);
            drawGradientDescent(alpha);
        }
        // Initial draw
        updateGradient();
        // Event listener
        lrSlider.addEventListener('input', updateGradient);
    }
});
