"""
TLS Angle Optimization Animation (Oblique Regression) - Multi-Scenario.

To render in 4K with GPU:
    manim --renderer=opengl -qk regression/tls_angle_optimization.py TLSAngleOptimizationScene
"""

from manim import *
import numpy as np

class TLSAngleOptimizationScene(Scene):
    def construct(self):
        # --- 1. Global Style & Layout Definitions (from instructions.md) ---
        self.font_style = {"font": "Monospace"}
        
        # Colors
        self.col_title = YELLOW
        self.col_true = GREEN
        self.col_ols = RED
        self.col_tls = PURPLE
        self.col_text = WHITE
        self.col_watermark = GRAY
        
        # Watermark
        cogl_text = Text("COGL - animations", font_size=18, color=self.col_watermark, **self.font_style)
        cogl_text.to_corner(DR, buff=0.2)
        self.add(cogl_text)

        # Title (Fixed)
        self.title = Text("Angle Optimization (Oblique Regression)", font_size=32, color=self.col_title, **self.font_style)
        self.title.to_edge(RIGHT, buff=1.0).shift(UP * 3)
        self.add(self.title)

        # --- 2. Define Scenarios ---
        # Format: (true_m, true_b, noise_scale, description)
        scenarios = [
            (1.0, 1.0, 0.6, "Scenario 1: Standard (m=1.0)"),
            (2.0, 0.5, 0.5, "Scenario 2: Steep Slope (m=2.0)"),
            (0.5, 2.0, 0.4, "Scenario 3: Shallow Slope (m=0.5)"),
            (-1.0, 8.0, 0.6, "Scenario 4: Negative Slope (m=-1.0)"),
            (0.0, 5.0, 0.3, "Scenario 5: Flat Line (m=0.0)"),
            (4.0, -10.0, 1.5, "Scenario 6: High Noise & Steep"), # Large offsets
            (-0.5, 6.0, 0.8, "Scenario 7: Negative Shallow"),
            (1.5, 0.0, 0.2, "Scenario 8: Low Noise"),
            (-2.0, 12.0, 1.0, "Scenario 9: Steep Negative"),
            (1.0, 4.0, 2.0, "Scenario 10: High Variance"),
        ]

        # --- 3. Run Scenarios ---
        for i, sc in enumerate(scenarios):
            self.animate_scenario(i, *sc)

    def animate_scenario(self, index, true_m, true_b, noise_scale, desc_text):
        # Clean previous mobjects if not first run
        # Note: We keep title and watermark
        
        # --- Setup Axes (Left Side) ---
        axes = Axes(
            x_range=[0, 12, 2],
            y_range=[0, 12, 2],
            x_length=6,
            y_length=6,
            axis_config={"color": BLUE, "include_numbers": False},
            tips=False
        ).shift(LEFT * 3 + DOWN * 0.5)
        
        # Manual Axes Labels (avoid LaTeX)
        x_label = Text("x", font_size=24, color=WHITE, **self.font_style).next_to(axes.x_axis, RIGHT)
        y_label = Text("y", font_size=24, color=WHITE, **self.font_style).next_to(axes.y_axis, UP)
        
        # Scenario Title
        scenario_title = Text(desc_text, font_size=24, color=WHITE, **self.font_style)
        scenario_title.next_to(self.title, DOWN, buff=0.5).align_to(self.title, LEFT)
        
        self.play(Create(axes), Write(x_label), Write(y_label), Write(scenario_title), run_time=1)
        
        # --- Data Generation ---
        np.random.seed(42 + index) 
        n_points = 20
        x_min, x_max = 1, 10
        x_true = np.linspace(x_min, x_max, n_points)
        y_true_vals = true_m * x_true + true_b
        
        # Add noise
        noise_x = np.random.normal(0, noise_scale, size=n_points)
        noise_y = np.random.normal(0, noise_scale, size=n_points)
        x_obs = x_true + noise_x
        y_obs = y_true_vals + noise_y
        
        # Calculate Stats for Regression
        x_mean = np.mean(x_obs)
        y_mean = np.mean(y_obs)
        Sxx = np.sum((x_obs - x_mean)**2)
        Syy = np.sum((y_obs - y_mean)**2)
        Sxy = np.sum((x_obs - x_mean) * (y_obs - y_mean))
        
        # Plot Dots
        dots = VGroup(*[Dot(axes.c2p(x, y), color=WHITE, radius=0.08) for x, y in zip(x_obs, y_obs)])
        
        self.play(FadeIn(dots), run_time=0.5)
        
        # True Line (Green)
        true_line = DashedVMobject(
            axes.plot(lambda x: true_m * x + true_b, color=self.col_true, x_range=[-2, 14]),
            num_dashes=20
        )
        p1 = axes.c2p(8, true_m*8+true_b)
         # Simple heuristic for label placement - if p1 is out of bounds, maybe clamp? 
         # We'll just rely on Manim not crashing.
        true_label = Text("True", font_size=16, color=self.col_true, **self.font_style).next_to(true_line, UP, buff=0.1)
        
        self.play(Create(true_line), FadeIn(true_label), run_time=0.5)
        
        # --- Optimization Setup ---
        
        # Optimization Plot (Right Bottom)
        opt_axes = Axes(
            x_range=[0, 180, 45],
            y_range=[0, 2.0, 0.5], 
            x_length=4,
            y_length=3,
            axis_config={"color": GREY, "include_numbers": False},
            tips=False
        ).to_corner(DR, buff=0.8) 
        
        # Manual Labels for Opt Axes
        opt_labels = VGroup()
        for x in [0, 90, 180]:
            t = Text(str(x), font_size=12, **self.font_style).next_to(opt_axes.c2p(x, 0), DOWN)
            opt_labels.add(t)
        
        opt_xlabel = Text("Angle", font_size=16, **self.font_style).next_to(opt_axes.x_axis, DOWN, buff=0.4)
        opt_ylabel = Text("Error", font_size=16, **self.font_style).next_to(opt_axes.y_axis, LEFT, buff=0.1).rotate(90*DEGREES)
        
        self.play(Create(opt_axes), FadeIn(opt_labels), Write(opt_xlabel), Write(opt_ylabel), run_time=0.5)
        
        # Calc Errors for Curve
        angles = np.linspace(1, 179, 100)
        errors = []
        
        def get_regression_params(angle_deg):
            theta = angle_deg * DEGREES
            if np.isclose(angle_deg % 180, 90, atol=1e-1): # OLS
                m = Sxy / Sxx if Sxx != 0 else 0
            elif np.isclose(angle_deg % 180, 0, atol=1e-1): # InvOLS
                m = Syy / Sxy if Sxy != 0 else 0
            else:
                k = np.tan(theta)
                denom = Sxy - k * Sxx
                if abs(denom) < 1e-6: m = 1.0 
                else: m = (Syy - k * Sxy) / denom
            b_val = y_mean - m * x_mean
            return m, b_val

        for a in angles:
            m_val, _ = get_regression_params(a)
            err = min(abs(m_val - true_m), 2.0) 
            errors.append(err)
            
        error_curve = opt_axes.plot_line_graph(
            angles, errors, add_vertex_dots=False, line_color=YELLOW
        )
        self.play(Create(error_curve), run_time=0.5)

        # Tracker
        angle_tracker = ValueTracker(90)
        
        # Dynamic Fit Line
        fit_line = always_redraw(lambda: 
            axes.plot(
                lambda x: get_regression_params(angle_tracker.get_value())[0] * x + 
                          get_regression_params(angle_tracker.get_value())[1],
                color=self.col_ols, 
                x_range=[-2, 14]
            )
        )
        self.add(fit_line)
        
        # Dynamic Dot on Plot
        track_dot = always_redraw(lambda: 
            Dot(opt_axes.c2p(angle_tracker.get_value(), 
                min(abs(get_regression_params(angle_tracker.get_value())[0] - true_m), 2.0)), 
                color=RED, radius=0.08)
        )
        self.add(track_dot)
        
        # Info Panel
        info_panel = always_redraw(lambda: VGroup(
            Text(f"Angle: {angle_tracker.get_value():.1f}", font_size=20, color=ORANGE, **self.font_style),
            Text(f"Slope: {get_regression_params(angle_tracker.get_value())[0]:.2f}", font_size=20, color=self.col_ols, **self.font_style)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(scenario_title, DOWN, buff=0.5).align_to(scenario_title, LEFT))
        self.add(info_panel)
        
        # --- Animation Sweep ---
        # Sweep 0 -> 180 faster
        self.play(angle_tracker.animate.set_value(179), run_time=1.5, rate_func=linear)
        self.play(angle_tracker.animate.set_value(1), run_time=1.5, rate_func=linear)
        
        # Find Best
        min_idx = np.argmin(errors)
        best_angle = angles[min_idx]
        self.play(angle_tracker.animate.set_value(best_angle), run_time=1.0)
        
        # Small Pause to show result
        self.wait(1.0)
        
        # --- Cleanup for Next Loop ---
        self.play(
            FadeOut(axes), FadeOut(x_label), FadeOut(y_label),
            FadeOut(dots), FadeOut(true_line), FadeOut(true_label),
            FadeOut(opt_axes), FadeOut(opt_labels), FadeOut(opt_xlabel), FadeOut(opt_ylabel),
            FadeOut(error_curve), FadeOut(fit_line), FadeOut(track_dot),
            FadeOut(info_panel), FadeOut(scenario_title),
            run_time=0.5
        )
