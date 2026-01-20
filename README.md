# Linear Regression Visualization with Manim

This project contains a Manim animation `LinearRegressionScene` that visualizes the concepts of Ordinary Least Squares (OLS) and Total Least Squares (TLS) regression, including a generative process with errors-in-variables.

## Features

- **Generative Process**: Visualizes the creation of synthetic data with both vertical ($y$) and horizontal ($x$) noise added to a true underlying linear law.
- **OLS Estimation**: Visualizes Ordinary Least Squares regression by minimizing vertical residuals.
- **TLS Estimation**: Visualizes Total Least Squares regression by minimizing perpendicular residuals.
- **Quantitative Comparison**: A final comparison table showing the slope ($m$) and intercept ($b$) for the True Law, OLS Fit, and TLS Fit, highlighting attenuation bias.
- **Layout**: Optimized for 16:9 4K/1080p rendering with a clean split between the plot (left) and information (right).
- **Watermark**: "COGL - animations" watermark.

## Installation

1.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  Install Manim and dependencies:
    ```bash
    pip install manim
    ```
    *(Note: You may need system dependencies like `PROJ`, `cairo`, `pango`, `ffmpeg`. On macOS: `brew install pkg-config cairo pango ffmpeg`)*

## Usage

Activate your virtual environment:

```bash
source venv/bin/activate
```

### Rendering

**Preview (Low Quality 480p/720p):**
```bash
manim -qm regression/regressions.py LinearRegressionScene
```

**Full HD (1080p):**
```bash
manim -qh regression/regressions.py LinearRegressionScene
```

**4K Ultra HD:**
```bash
manim -qk regression/regressions.py LinearRegressionScene
```

### GPU Acceleration (Apple Silicon / OpenGL)

To speed up rendering on supported hardware (like Apple Silicon Macs), use the OpenGL renderer:

```bash
manim --renderer=opengl -qk regression/regressions.py LinearRegressionScene
```

## Project Structure

- `regression/regressions.py`: Main animation script containing `LinearRegressionScene`.
- `media/`: Output directory for generated videos.