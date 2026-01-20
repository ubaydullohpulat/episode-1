# Manim Animation Standards & Instructions

This document serves as the standard reference for creating Manim animations in this project. All future animations must adhere to these guidelines to ensure consistency in visualization, layout, and typography.

## 1. General Philosophy

-   **Visual Clarity**: Prioritize clear separation between data visualization (Left) and explanatory text (Right).
-   **Geometric Accuracy**: Always use 1:1 aspect ratios for axes when visualizing geometric concepts (e.g., perpendicular distances).
-   **Consistency**: Use standard fonts, colors, and positioning across all scenes.

## 2. Layout & Positioning

The screen is logically divided into two halves:
-   **Left Side**: Dedicated to graphs, axes, and geometric visualizations.
-   **Right Side**: Dedicated to titles, formulas, parameters, and explanations.

### Axes Configuration
-   **Position**: Shifted left and slightly down to center visually in the left half.
    ```python
    axes = Axes(
        ...,
        x_length=6, y_length=6, # Maintain 1:1 aspect ratio for equal ranges
    ).shift(LEFT * 3 + DOWN * 0.5)
    ```

### Text Alignment (Right Panel)
-   **Titles**: Anchored to the top-right with a standard buffer.
    ```python
    title.to_edge(RIGHT, buff=1.0).shift(UP * 3)
    ```
-   **Body Text / Lists**: Grouped using `VGroup` and arranged vertically with consistent padding.
    ```python
    info_group = VGroup(text1, text2, ...)
    info_group.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
    info_group.next_to(title, DOWN, buff=1.0).align_to(title, LEFT)
    ```

### Watermark
-   **Content**: "COGL - animations"
-   **Position**: Bottom-right corner.
-   **Style**: Small, gray, monospaced.
    ```python
    cogl_text = Text("COGL - animations", font_size=18, color=GRAY, font="Monospace")
    cogl_text.to_corner(DR, buff=0.2)
    self.add(cogl_text)
    ```

## 3. Typography

All text must use a **Monospace** font to ensure consistent character spacing and alignment, especially for numerical data and tables.

```python
font_style = {"font": "Monospace"}
Text("...", **font_style)
```

### Font Sizes
| Element Type | Size | Example Use |
| :--- | :--- | :--- |
| **Main Title** | 32 | "Generative Process" |
| **Section Title** | 28 | "OLS Estimation" |
| **Body / Formulas** | 24 | Explanations, Metrics ($m=...$) |
| **Axis Labels** | 24 | "x", "y" |
| **Small Labels** | 20 | "True Law" tags next to lines |
| **Watermark** | 18 | Footer watermark |

## 4. Color Palette

Use standard Manim colors for semantic consistency:

-   **Primary Text**: `WHITE`
-   **Titles**: `YELLOW`
-   **True Underlying Model**: `GREEN`
-   **OLS Model / Y-Noise**: `RED`
-   **TLS Model / X-Noise**: `PURPLE` / `BLUE` (Use consistently)
-   **Watermark**: `GRAY`

## 5. Rendering Standards

Always test and render with the following flags:

-   **4K (Ultra HD)**: `-qk`
-   **GPU Acceleration (OpenGL)**: `--renderer=opengl`
-   **Standard Command**:
    ```bash
    manim --renderer=opengl -qk <script.py> <SceneName>
    ```

## 6. Example Code Snippet

```python
from manim import *

class StandardScene(Scene):
    def construct(self):
        font_style = {"font": "Monospace"}
        
        # 1. Setup Axes (Left)
        axes = Axes(x_length=6, y_length=6).shift(LEFT * 3 + DOWN * 0.5)
        self.add(axes)
        
        # 2. Setup Title (Right)
        title = Text("Title", font_size=32, color=YELLOW, **font_style)
        title.to_edge(RIGHT, buff=1.0).shift(UP * 3)
        self.add(title)
        
        # 3. Add Info (Right)
        info = Text("Details...", font_size=24, **font_style)
        info.next_to(title, DOWN, buff=1.0).align_to(title, LEFT)
        self.add(info)
        
        # 4. Watermark
        wm = Text("COGL - animations", font_size=18, color=GRAY, **font_style).to_corner(DR, buff=0.2)
        self.add(wm)
```
