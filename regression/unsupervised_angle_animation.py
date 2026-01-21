"""
Unsupervised Angle Optimization - Full Explanation & Benchmark.

To render (Use Default Cairo Renderer, OpenGL may fail on Text triangulation):
    manim -qh regression/unsupervised_angle_animation.py UnsupervisedAngleScene
"""

from manim import *
import numpy as np

class UnsupervisedAngleScene(Scene):
    def construct(self):
        # --- Style Config ---
        self.font_style = {"font": "Monospace"}
        self.col_title = YELLOW
        self.col_true = GREEN
        self.col_ols = RED
        self.col_opt = PURPLE
        self.col_text = WHITE
        self.results = []
        
        cogl_text = Text("COGL - animations", font_size=18, color=GRAY, **self.font_style)
        cogl_text.to_corner(DR, buff=0.2)
        cogl_text.z_index = 1000 # Ensure it stays on top
        self.add(cogl_text)

        # Title
        self.main_title = Text("Unsupervised Angle Optimization", font_size=32, color=self.col_title, **self.font_style)
        self.main_title.to_edge(UP, buff=0.5)
        self.add(self.main_title)

        # --- 1. Detailed Scenario (Scenario 1) ---
        # Fixed seed for explanation consistency
        np.random.seed(42) 
        self.run_detailed_scenario(1, true_m=1.5, true_b=1.0)
        
        # --- 2. Fast Scenarios (2-10) ---
        # More random seeds
        scenarios = [
            (2, 0.5, 2.0),
            (3, -1.0, 5.0),
            (4, 2.0, 0.0),
            (5, -0.5, 4.0),
            (6, 1.0, 1.0),
            (7, -2.0, 8.0),
            (8, 0.0, 3.0),
            (9, 1.2, 0.5),
            (10, -1.5, 6.0)
        ]
        
        # Title update
        self.play(Transform(self.main_title, Text("Benchmarking 10 Scenarios...", font_size=32, color=self.col_title, **self.font_style).to_edge(UP, buff=0.5)))
        
        for idx, m, b in scenarios:
             self.run_fast_scenario(idx, m, b)
             
        # --- 3. Comparison Chart ---
        self.show_summary_chart()
        
    def run_detailed_scenario(self, index, true_m, true_b):
        # Setup Axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": BLUE, "include_numbers": False},
            tips=False
        ).shift(LEFT * 3 + DOWN * 0.5)
        
        x_label = Text("x", font_size=24, color=WHITE, **self.font_style).next_to(axes.x_axis, RIGHT)
        y_label = Text("y", font_size=24, color=WHITE, **self.font_style).next_to(axes.y_axis, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label), run_time=1)
        
        # Data Gen
        n_points = 25
        x_true = np.linspace(1, 8, n_points)
        y_true_vals = true_m * x_true + true_b
        x_obs = x_true + np.random.normal(0, 0.8, n_points)
        y_obs = y_true_vals + np.random.normal(0, 0.8, n_points)
        
        dots = VGroup(*[Dot(axes.c2p(x, y), color=WHITE, radius=0.08) for x, y in zip(x_obs, y_obs)])
        self.play(FadeIn(dots))
        
        # OLS
        A = np.vstack([x_obs, np.ones(len(x_obs))]).T
        m_ols, b_ols = np.linalg.lstsq(A, y_obs, rcond=None)[0]
        ols_line = axes.plot(lambda x: m_ols*x + b_ols, color=self.col_ols, x_range=[-1, 11])
        
        # Right Panel Text
        # Use VGroup to organize and scale
        info_panel = VGroup()
        
        t1 = Text("Scenario 1: Detailed Breakdown", font_size=24, color=ORANGE, **self.font_style)
        t2 = Text("1. Standard OLS (Red)", font_size=20, color=WHITE, **self.font_style)
        t3 = Text("   Biased by X-noise.", font_size=18, color=GRAY, **self.font_style)
        
        info_panel.add(t1, t2, t3)
        info_panel.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        info_panel.to_edge(RIGHT, buff=0.5).shift(UP*1.0)
        
        self.play(Write(info_panel), Create(ols_line))
        self.wait(1)
        
        # Sweep
        t4 = Text("2. Sweeping Angle...", font_size=20, color=WHITE, **self.font_style)
        info_panel.add(t4)
        info_panel.arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.5).shift(UP*1.0)
        self.play(Write(t4), FadeOut(ols_line)) # Hide OLS to show sweep
        
        # Sweep Logic
        x_mean = np.mean(x_obs)
        y_mean = np.mean(y_obs)
        Sxx = np.sum((x_obs - x_mean)**2)
        Syy = np.sum((y_obs - y_mean)**2)
        Sxy = np.sum((x_obs - x_mean) * (y_obs - y_mean))
        
        def get_fit(angle_deg):
            theta = angle_deg * DEGREES
            # Edge cases
            if np.isclose(angle_deg % 180, 90, atol=0.1): m = Sxy/Sxx
            elif np.isclose(angle_deg % 180, 0, atol=0.1): m = Syy/Sxy if Sxy!=0 else 0
            else:
                k = np.tan(theta)
                denom = Sxy - k * Sxx
                m = (Syy - k*Sxy)/denom if abs(denom)>1e-6 else 1.0
            b = y_mean - m*x_mean
            return m, b

        angle_tracker = ValueTracker(90)
        fit_line = always_redraw(lambda: axes.plot(lambda x: get_fit(angle_tracker.get_value())[0]*x + get_fit(angle_tracker.get_value())[1], color=self.col_opt, x_range=[-1,11]))
        self.add(fit_line)
        
        # Ortho Loss Plot (Small)
        loss_axes = Axes(x_range=[0,180,90], y_range=[0,1,1], x_length=2, y_length=1.5, axis_config={"color": GREY}).next_to(info_panel, DOWN, buff=0.5)
        # Precompute loss
        angs = np.linspace(1,179,50)
        losses = []
        for a in angs:
             m, _ = get_fit(a)
             v = np.array([1, m]); v/=np.linalg.norm(v)
             p = np.array([np.cos(a*DEGREES), np.sin(a*DEGREES)])
             losses.append((np.dot(v,p))**2)
        loss_curve = loss_axes.plot_line_graph(angs, losses, add_vertex_dots=False, line_color=YELLOW)
        self.play(Create(loss_axes), Create(loss_curve))
        
        # Animate Sweep
        self.play(angle_tracker.animate.set_value(179), run_time=1.5, rate_func=linear)
        self.play(angle_tracker.animate.set_value(1), run_time=1.5, rate_func=linear)
        
        # Find Best
        min_idx = np.argmin(losses)
        best_ang = angs[min_idx]
        m_opt, b_opt = get_fit(best_ang)
        
        self.play(angle_tracker.animate.set_value(best_ang), run_time=0.5)
        
        t5 = Text(f"3. Found: {best_ang:.1f} deg", font_size=20, color=YELLOW, **self.font_style)
        info_panel.add(t5)
        info_panel.arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_edge(RIGHT, buff=0.5).shift(UP*1.0)
        self.play(Write(t5))
        
        # Errors
        err_ols = abs(m_ols - true_m)
        err_opt = abs(m_opt - true_m)
        self.results.append((index, err_ols, err_opt))
        
        # Show True Line Comparison
        true_line = DashedVMobject(axes.plot(lambda x: true_m*x+true_b, color=self.col_true, x_range=[-1,11]), num_dashes=20)
        ols_line_static = axes.plot(lambda x: m_ols*x+b_ols, color=self.col_ols, x_range=[-1,11])
        
        self.play(Create(true_line), FadeIn(ols_line_static))
        
        # Final Text for this scene
        res_text = Text(f"True m: {true_m:.2f}\nOLS m: {m_ols:.2f}\nOPT m: {m_opt:.2f}", font_size=18, color=WHITE, **self.font_style).next_to(loss_axes, DOWN)
        self.play(Write(res_text))
        self.wait(2)
        
        # Cleanup
        self.play(
            FadeOut(axes), FadeOut(x_label), FadeOut(y_label), FadeOut(dots),
            FadeOut(true_line), FadeOut(ols_line_static), FadeOut(fit_line),
            FadeOut(info_panel), FadeOut(loss_axes), FadeOut(loss_curve), FadeOut(res_text)
        )

    def run_fast_scenario(self, index, true_m, true_b):
        # Condensed version
        
        # Title
        scen_text = Text(f"Scenario {index}", font_size=24, color=WHITE, **self.font_style).to_corner(UL, buff=1.0)
        self.add(scen_text)
        
        # Axes (Recreate)
        axes = Axes(x_range=[0,10,2], y_range=[0,10,2], x_length=4, y_length=4, axis_config={"color": BLUE}).shift(LEFT*2)
        self.play(Create(axes), run_time=0.2)
        
        # Data
        x_obs = np.linspace(1,8,20) + np.random.normal(0, 0.8, 20)
        y_obs = (true_m * np.linspace(1,8,20) + true_b) + np.random.normal(0, 0.8, 20)
        dots = VGroup(*[Dot(axes.c2p(x,y), radius=0.06, color=WHITE) for x,y in zip(x_obs,y_obs)])
        self.add(dots)
        
        # Math
        A = np.vstack([x_obs, np.ones(len(x_obs))]).T
        m_ols, _ = np.linalg.lstsq(A, y_obs, rcond=None)[0]
        
        x_mean = np.mean(x_obs); y_mean = np.mean(y_obs)
        Sxx = np.sum((x_obs-x_mean)**2); Syy = np.sum((y_obs-y_mean)**2); Sxy = np.sum((x_obs-x_mean)*(y_obs-y_mean))
        
        # Brute force best angle
        angs = np.linspace(1,179,100)
        best_loss = np.inf
        best_m = 0
        for a in angs:
            t = a*DEGREES
            k = np.tan(t)
            denom = Sxy-k*Sxx
            m = (Syy-k*Sxy)/denom if abs(denom)>1e-6 else 1.0
            v = np.array([1,m]); v/=np.linalg.norm(v)
            p = np.array([np.cos(t), np.sin(t)])
            loss = (np.dot(v,p))**2
            if loss < best_loss:
                best_loss = loss
                best_m = m
                
        # Show Results
        err_ols = abs(m_ols - true_m)
        err_opt = abs(best_m - true_m)
        self.results.append((index, err_ols, err_opt))
        
        # Visuals
        true_l = DashedVMobject(axes.plot(lambda x: true_m*x+true_b, color=self.col_true), num_dashes=10)
        ols_l = axes.plot(lambda x: m_ols*x+ (y_mean-m_ols*x_mean), color=self.col_ols)
        opt_l = axes.plot(lambda x: best_m*x+ (y_mean-best_m*x_mean), color=self.col_opt)
        
        self.add(true_l, ols_l, opt_l)
        
        # Text
        res_t = Text(f"True: {true_m:.2f}\nOLS: {m_ols:.2f}\nOPT: {best_m:.2f}", font_size=16, **self.font_style).next_to(axes, RIGHT)
        self.add(res_t)
        
        self.wait(0.5)
        
        # Clean
        self.remove(axes, dots, true_l, ols_l, opt_l, res_t, scen_text)

    def show_summary_chart(self):
        # Summary
        self.play(FadeOut(self.main_title))
        title = Text("Performance Summary (Mean Absolute Error)", font_size=32, color=YELLOW, **self.font_style).to_edge(UP)
        self.play(Write(title))
        
        # Prepare Data
        # results list of tuples (idx, ols_err, opt_err)
        ols_errs = [r[1] for r in self.results]
        opt_errs = [r[2] for r in self.results]
        
        avg_ols = np.mean(ols_errs)
        avg_opt = np.mean(opt_errs)
        
        # Bar Chart
        chart_axes = Axes(
            x_range=[0, 11, 1],
            y_range=[0, max(max(ols_errs), 1.0)*1.2, 0.5],
            x_length=8,
            y_length=4,
            axis_config={"include_numbers": False, "color": GREY}
        ).shift(DOWN*0.5)
        
        # Labels
        x_lab = Text("Scenario ID", font_size=20, **self.font_style).next_to(chart_axes.x_axis, DOWN)
        y_lab = Text("Slope Error", font_size=20, **self.font_style).next_to(chart_axes.y_axis, LEFT).rotate(90*DEGREES)
        self.play(Create(chart_axes), Write(x_lab), Write(y_lab))
        
        # Bars
        bars = VGroup()
        for i, (idx, eo, ep) in enumerate(self.results):
            # OLS Bar (Red)
            bar_o = Rectangle(
                width=0.3, 
                height=chart_axes.c2p(0, eo)[1] - chart_axes.c2p(0,0)[1],
                color=self.col_ols, fill_opacity=0.8
            ).move_to(chart_axes.c2p(idx - 0.2, eo/2))
            
            # OPT Bar (Purple)
            bar_p = Rectangle(
                width=0.3, 
                height=chart_axes.c2p(0, ep)[1] - chart_axes.c2p(0,0)[1],
                color=self.col_opt, fill_opacity=0.8
            ).move_to(chart_axes.c2p(idx + 0.2, ep/2))
            
            bars.add(bar_o, bar_p)
            
            # ID Label
            if i % 2 == 0: # label every other if crowded? or all
                lab = Text(str(idx), font_size=12, **self.font_style).next_to(chart_axes.c2p(idx, 0), DOWN)
                self.add(lab)

        self.play(Create(bars), run_time=2)
        
        # Avg Lines
        l_ols = DashedLine(chart_axes.c2p(0, avg_ols), chart_axes.c2p(11, avg_ols), color=self.col_ols)
        l_opt = DashedLine(chart_axes.c2p(0, avg_opt), chart_axes.c2p(11, avg_opt), color=self.col_opt)
        
        self.play(Create(l_ols), Create(l_opt))
        
        # Legend / Stats (Fixed Position to avoid overlap)
        stats_group = VGroup(
            Text(f"Avg OLS Error: {avg_ols:.3f}", font_size=20, color=self.col_ols, **self.font_style),
            Text(f"Avg OPT Error: {avg_opt:.3f}", font_size=20, color=self.col_opt, **self.font_style),
            Text(f"Improvement: {(avg_ols-avg_opt):.3f}", font_size=20, color=YELLOW, **self.font_style)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        
        # Position to the right of the chart or top right corner
        stats_group.next_to(chart_axes, RIGHT, buff=0.5).to_edge(RIGHT, buff=0.5).shift(UP*1)
        
        # Check if it fits, otherwise move to top right inside
        if stats_group.get_left()[0] > 7: # Screen width is approx 8
             stats_group.to_corner(UR, buff=0.5).shift(DOWN*1.0)
        
        self.play(Write(stats_group))
        
        self.wait(3)
