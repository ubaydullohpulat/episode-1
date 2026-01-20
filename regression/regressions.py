"""
Linear Regression Animation with Manim.

To render in 4K (Ultra HD), use the -qk flag:
    manim -qk regression/regressions.py LinearRegressionScene
    
To render in 1080p (Full HD), use the -qh flag:
    manim -qh regression/regressions.py LinearRegressionScene

To use GPU acceleration (OpenGL renderer), use --renderer=opengl:
    manim --renderer=opengl -qk regression/regressions.py LinearRegressionScene
"""
from manim import *
import numpy as np

class LinearRegressionScene(Scene):
    def construct(self):
        # Configuration
        # Font style for consistency
        font_style = {"font": "Monospace"} 
        
        # Axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": BLUE, "include_numbers": False},
            x_length=6,
            y_length=6,
            tips=False
        ).shift(LEFT * 3 + DOWN * 0.5)
        
        # Manual Labels
        x_label = Text("x", font_size=24, color=WHITE, **font_style).next_to(axes.x_axis, RIGHT)
        y_label = Text("y", font_size=24, color=WHITE, **font_style).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Watermark
        cogl_text = Text("COGL - animations", font_size=18, color=GRAY, **font_style).to_corner(DR, buff=0.2)
        self.add(cogl_text)
        
        # --- PART 1: THE GENERATIVE PROCESS ---
        
        # Title
        title = Text("Generative Process", font_size=32, color=YELLOW, **font_style)
        title.to_edge(RIGHT, buff=1.0).shift(UP * 3) # Anchor top-right
        self.play(Write(title))
        
        # 1. True Underlying Law
        true_m = 1.0
        true_b = 1.0
        
        true_line_func = lambda x: true_m * x + true_b
        true_line_graph = axes.plot(true_line_func, color=GREEN, x_range=[0, 8.5])
        true_label = Text("True Law", font_size=20, color=GREEN, **font_style).next_to(true_line_graph, UP, buff=0.1).shift(RIGHT*1)
        
        self.play(Create(true_line_graph), FadeIn(true_label))
        self.wait(1)
        
        # 2. Sample Points
        np.random.seed(60) 
        n_points = 15
        x_true = np.linspace(1, 8, n_points)
        y_true = true_m * x_true + true_b
        
        perfect_dots = VGroup(*[Dot(axes.c2p(x, y), color=WHITE, radius=0.06) for x, y in zip(x_true, y_true)])
        self.play(FadeIn(perfect_dots))
        
        # 3. Y-Noise
        noise_y = np.random.normal(0, 0.6, size=n_points)
        y_obs_temp = y_true + noise_y
        y_noise_dots = VGroup(*[Dot(axes.c2p(x, y), color=YELLOW, radius=0.08) for x, y in zip(x_true, y_obs_temp)])
        
        y_arrows = VGroup()
        for p_dot, y_dot in zip(perfect_dots, y_noise_dots):
            arrow = Arrow(p_dot.get_center(), y_dot.get_center(), color=RED, buff=0, stroke_width=2, tip_length=0.1)
            y_arrows.add(arrow)
            
        # Explanations Panel
        explanations = VGroup(
            Text("+ Y Noise (Vertical)", font_size=24, color=RED, **font_style),
            Text("+ X Noise (Horizontal)", font_size=24, color=BLUE, **font_style)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        
        explanations.next_to(title, DOWN, buff=1.0).align_to(title, LEFT)
        
        self.play(Write(explanations[0]))
        self.play(Transform(perfect_dots, y_noise_dots), Create(y_arrows), run_time=1.5)
        
        # 4. X-Noise
        noise_x = np.random.normal(0, 0.6, size=n_points)
        x_obs = x_true + noise_x
        y_obs = y_obs_temp 
        final_dots = VGroup(*[Dot(axes.c2p(x, y), color=WHITE, radius=0.08) for x, y in zip(x_obs, y_obs)])
        
        x_arrows = VGroup()
        for y_dot, final_dot in zip(y_noise_dots, final_dots): 
            arrow = Arrow(y_dot.get_center(), final_dot.get_center(), color=BLUE, buff=0, stroke_width=2, tip_length=0.1)
            x_arrows.add(arrow)

        self.play(Write(explanations[1]))
        self.play(Transform(perfect_dots, final_dots), Create(x_arrows), run_time=1.5)
        self.wait(1)
        
        # Cleanup
        self.play(
            FadeOut(true_line_graph),
            FadeOut(true_label),
            FadeOut(y_arrows),
            FadeOut(x_arrows),
            FadeOut(title),
            FadeOut(explanations)
        )
        dots = perfect_dots
        dots.set_color(WHITE)
        
        # --- PART 2: OLS ---
        
        ols_title = Text("OLS Estimation", font_size=28, color=RED, **font_style)
        ols_title.to_edge(RIGHT, buff=1.0).shift(UP * 2)
        
        self.play(Write(ols_title))
        
        m_ols_track = ValueTracker(0.5)
        b_ols_track = ValueTracker(3.0)
        
        ols_line = always_redraw(lambda: axes.plot(lambda x: m_ols_track.get_value() * x + b_ols_track.get_value(), color=RED, x_range=[0, 10]))
        self.play(Create(ols_line))
        
        # OLS Residuals
        def get_ols_residuals():
            m = m_ols_track.get_value()
            b = b_ols_track.get_value()
            lines = VGroup()
            for x, y in zip(x_obs, y_obs):
                y_pred = m * x + b
                p1 = axes.c2p(x, y)
                p2 = axes.c2p(x, y_pred)
                lines.add(Line(p1, p2, color=ORANGE, stroke_opacity=0.5))
            return lines
            
        ols_residuals_vis = always_redraw(get_ols_residuals)
        self.play(Create(ols_residuals_vis))
        
        # Calculate OLS
        A = np.vstack([x_obs, np.ones(len(x_obs))]).T
        m_ols_opt, b_ols_opt = np.linalg.lstsq(A, y_obs, rcond=None)[0]
        
        self.play(
            m_ols_track.animate.set_value(m_ols_opt),
            b_ols_track.animate.set_value(b_ols_opt),
            run_time=2,
            rate_func=smooth
        )
        
        # Show OLS result text
        ols_res_text = Text(f"Slope: {m_ols_opt:.2f}\nIntercept: {b_ols_opt:.2f}", font_size=24, color=RED, **font_style)
        ols_res_text.next_to(ols_title, DOWN, buff=0.5).align_to(ols_title, LEFT)
        
        self.play(Write(ols_res_text))
        self.wait(1)
        
        # Freeze OLS
        ols_line.clear_updaters()
        ols_residuals_vis.clear_updaters()
        self.play(FadeOut(ols_residuals_vis), FadeOut(ols_title), FadeOut(ols_res_text))
        
        # --- PART 3: TLS ---
        
        tls_title = Text("TLS Estimation", font_size=28, color=PURPLE, **font_style)
        tls_title.to_edge(RIGHT, buff=1.0).shift(UP * 2) # Same position as OLS title for continuity
        
        self.play(Write(tls_title))
        
        m_tls_track = ValueTracker(m_ols_opt)
        b_tls_track = ValueTracker(b_ols_opt)
        
        tls_line = always_redraw(lambda: axes.plot(lambda x: m_tls_track.get_value() * x + b_tls_track.get_value(), color=PURPLE, x_range=[0, 10]))
        self.play(Create(tls_line))
        
        # TLS Residuals
        def get_tls_residuals():
            m = m_tls_track.get_value()
            b = b_tls_track.get_value()
            lines = VGroup()
            for x0, y0 in zip(x_obs, y_obs):
                if m == 0:
                    xp, yp = x0, b
                else:
                    xp = (x0 + m*y0 - m*b) / (m**2 + 1)
                    yp = m * xp + b
                p_data = axes.c2p(x0, y0)
                p_proj = axes.c2p(xp, yp)
                lines.add(Line(p_data, p_proj, color=GREEN, stroke_opacity=0.6))
            return lines
            
        tls_residuals_vis = always_redraw(get_tls_residuals)
        self.play(Create(tls_residuals_vis))
        
        # Calculate TLS
        x_mean = np.mean(x_obs)
        y_mean = np.mean(y_obs)
        data_centered = np.vstack([x_obs - x_mean, y_obs - y_mean]).T
        U, S, Vt = np.linalg.svd(data_centered)
        v = Vt[0] 
        m_tls_opt = v[1] / v[0]
        b_tls_opt = y_mean - m_tls_opt * x_mean
        
        self.play(
            m_tls_track.animate.set_value(m_tls_opt),
            b_tls_track.animate.set_value(b_tls_opt),
            run_time=2
        )
        
        tls_res_text = Text(f"Slope: {m_tls_opt:.2f}\nIntercept: {b_tls_opt:.2f}", font_size=24, color=PURPLE, **font_style)
        tls_res_text.next_to(tls_title, DOWN, buff=0.5).align_to(tls_title, LEFT)
        
        self.play(Write(tls_res_text))
        self.wait(1)
        
        # Freeze TLS
        tls_line.clear_updaters()
        tls_residuals_vis.clear_updaters()
        self.play(FadeOut(tls_residuals_vis), FadeOut(tls_title), FadeOut(tls_res_text))
        
        # --- PART 4: COMPARISON ---
        
        comp_title = Text("Model Comparison", font_size=32, color=WHITE, **font_style)
        comp_title.to_edge(RIGHT, buff=1.0).shift(UP * 3) # Align with Generative Process title pos
        self.play(Write(comp_title))
        
        # True Line
        true_line_dashed = DashedVMobject(axes.plot(lambda x: true_m*x + true_b, color=GREEN, x_range=[0,10]), num_dashes=20)
        true_line_dashed.set_z_index(10) # Ensure visible
        self.play(Create(true_line_dashed))
        
        # Comparison Table
        # Header
        row_params = [
            ("True Law", true_m, true_b, GREEN),
            ("OLS Fit ", m_ols_opt, b_ols_opt, RED),
            ("TLS Fit ", m_tls_opt, b_tls_opt, PURPLE)
        ]
        
        res_group = VGroup()
        for label, m_val, b_val, col in row_params:
            # Padded string for alignment if Monospace
            # "True Law" is 8 chars. "OLS Fit " is 8 chars.
            txt = Text(f"{label}: m={m_val:.2f}, b={b_val:.2f}", font_size=24, color=col, **font_style)
            res_group.add(txt)
            
        res_group.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        res_group.next_to(comp_title, DOWN, buff=0.8).align_to(comp_title, LEFT)
        
        self.play(Write(res_group))
        
        self.wait(5)
