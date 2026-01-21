
import numpy as np
import scipy.optimize

class ObliqueRegressor:
    """
    Performs Oblique Regression where residuals are measured at a specific projection angle.
    """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.angle_degrees_ = None
        self.slope_vector_ = None
        self.projection_vector_ = None

    def fit(self, X, y, angle_degrees):
        """
        Fits the model by minimizing sum of squared oblique distances at 'angle_degrees'.
        """
        X = np.array(X)
        y = np.array(y)
        self.angle_degrees_ = angle_degrees
        
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        Sxx = np.sum((X - x_mean)**2)
        Syy = np.sum((y - y_mean)**2)
        Sxy = np.sum((X - x_mean) * (y - y_mean))
        
        angle_deg = angle_degrees % 180
        theta = np.deg2rad(angle_deg)
        
        # Vectors for logic checking
        self.projection_vector_ = np.array([np.cos(theta), np.sin(theta)])
        
        # Analytic Solution
        if np.isclose(angle_deg, 90, atol=1e-2): # OLS
            m = Sxy / Sxx if Sxx != 0 else 0.0
        elif np.isclose(angle_deg, 0, atol=1e-2) or np.isclose(angle_deg, 180, atol=1e-2): # InvOLS
            m = Syy / Sxy if Sxy != 0 else 0.0
        else:
            k = np.tan(theta)
            denom = Sxy - k * Sxx
            if abs(denom) < 1e-8:
                m = 1e9 
            else:
                m = (Syy - k * Sxy) / denom
                
        self.coef_ = m
        self.intercept_ = y_mean - m * x_mean
        
        # Vector along the line [1, m] normalized
        v_line = np.array([1, m])
        self.slope_vector_ = v_line / np.linalg.norm(v_line)
        
        return self

class BlindAngleOptimizer:
    """
    Finds the optimal projection angle WITHOUT knowing the true slope.
    Strategy: GEOMETRIC CONSISTENCY (Orthogonality).
    
    We search for theta such that the regression line for that angle 
    is PERPENDICULAR to the projection angle theta.
    This satisfies the Total Least Squares condition.
    
    We use a grid search to avoid local optima (singularities).
    """
    def __init__(self):
        self.best_angle_ = None
        self.best_slope_ = None
        
    def optimize(self, X, y):
        # Grid search over 0 to 180 degrees
        angles = np.linspace(0.1, 179.9, 1000)
        best_loss = np.inf
        
        reg = ObliqueRegressor()
        
        candidates = []
        
        for angle in angles:
            reg.fit(X, y, angle)
            m = reg.coef_
            
            # Geometric Consistency Loss: Deviation from Orthogonality
            # Line vector: [1, m]
            # Projection vector: [cos(theta), sin(theta)]
            theta_rad = np.deg2rad(angle)
            proj = np.array([np.cos(theta_rad), np.sin(theta_rad)])
            line_dir = np.array([1, m])
            
            # Normalize line direction to ensure dot product scale is consistent
            line_dir_norm = line_dir / np.linalg.norm(line_dir)
            
            # Dot product should be 0 if perpendicular
            ortho_loss = (np.dot(line_dir_norm, proj))**2
            
            # Secondary Loss: Sum of Squared Projected Differences (Cost Function)
            # This distinguishes between the "Best Fit" (TLS) and "Worst Fit" directions
            # The fit itself minimizes sum((y_i...)**2 / ...).
            # We want to pick the orthogonal solution that has minimal residuals.
            # But just minimizing ortho_loss finds the directions.
            if ortho_loss < 0.001: 
                # Candidate for orthogonality found
                candidates.append((angle, m))
                
        # If candidates form clusters, pick best ones. 
        # Actually, let's just minimize the ortho_loss directly but track the one with min Variance?
        # Standard TLS minimizes the sum of squared perpendicular distances.
        # Cost = Sum( d_i^2 )
        # d_i = |y - mx - b| / sqrt(1+m^2)
        
        # Robust approach: 
        # 1. Minimize Orthogonality Loss to find roots (valid TLS angles).
        # 2. Pick root with min residuals.
        
        # For simplicity in this script, let's just sweep and pick the argmin of Ortho Loss,
        # but weighting it? No.
        
        # Let's perform a brute force minimization of the TLS Cost Function itself?
        # User asked for "guessing angle and trying to find original". 
        # The "Angle" method implies minimizing orthogonality.
        
        # Let's scan for orthogonality.
        best_candidate = None
        min_residual_norm = np.inf
        
        for angle in angles:
            reg.fit(X, y, angle)
            m = reg.coef_
            b = reg.intercept_
            
            # Check orthogonality
            theta_rad = np.deg2rad(angle)
            proj = np.array([np.cos(theta_rad), np.sin(theta_rad)])
            line_dir = np.array([1, m])
            line_dir_norm = line_dir / np.linalg.norm(line_dir)
            ortho_loss = (np.dot(line_dir_norm, proj))**2
            
            # Check Residual Norm (TLS Cost)
            # Perpendicular distance sum of squares
            # D = sum ( (y - mx - b)^2 / (1+m^2) )
            y_pred = m * X + b
            residuals = y - y_pred
            ssr = np.sum(residuals**2) / (1 + m**2)
            
            # Combined metric? 
            # Ideally ortho_loss is 0. 
            # We filter for low ortho_loss, then pick min ssr.
            if ortho_loss < 0.0001:
                # Better solution?
                if ssr < min_residual_norm:
                    min_residual_norm = ssr
                    best_candidate = (angle, m)
                    
        # If grid was too coarse, run refinement on the best candidate
        if best_candidate:
             self.best_angle_, self.best_slope_ = best_candidate
        else:
            # Fallback to just min ortho loss if strict threshold failed
            # (Recalculating simpler version)
            best_ortho = np.inf
            for angle in angles:
                reg.fit(X, y, angle)
                m = reg.coef_
                theta_rad = np.deg2rad(angle)
                proj = np.array([np.cos(theta_rad), np.sin(theta_rad)])
                line_dir = np.array([1, m])
                line_dir /= np.linalg.norm(line_dir)
                ortho = (np.dot(line_dir, proj))**2
                if ortho < best_ortho:
                    best_ortho = ortho
                    self.best_angle_ = angle
                    self.best_slope_ = m

        return self.best_angle_

def main():
    print("--- Unsupervised Angle-Based Regression (10 Scenarios) ---")
    print("Optimization Goal: Find angle where residuals are orthogonal to fit (TLS Condition)\n")
    
    np.random.seed(100)
    
    results = []
    
    for i in range(1, 11):
        # Random Generative Parameters
        true_m = np.random.uniform(-3, 3)
        # Avoid 0 slope for clearer orthogonality visualization usually, but should work
        if abs(true_m) < 0.1: true_m += 0.5 
        
        true_b = np.random.uniform(-5, 5)
        n_points = 100
        
        # Isotropic Noise (Critical for Orthogonal solution to be optimal)
        # If noise is anisotropic (e.g. sigma_y >> sigma_x), orthogonal is NOT optimal
        # But here we assume standard TLS conditions.
        noise_scale = 1.0
        
        X = np.linspace(0, 10, n_points)
        X_obs = X + np.random.normal(0, noise_scale, n_points)
        y_obs = (true_m * X + true_b) + np.random.normal(0, noise_scale, n_points)
        
        # 1. Standard OLS
        ols = ObliqueRegressor().fit(X_obs, y_obs, 90)
        
        # 2. Unsupervised Optimizer
        optimizer = BlindAngleOptimizer()
        best_angle = optimizer.optimize(X_obs, y_obs)
        best_slope = optimizer.best_slope_
        
        # Errors
        err_ols = abs(ols.coef_ - true_m)
        err_opt = abs(best_slope - true_m)
        
        results.append({
            "id": i,
            "true_m": true_m,
            "ols_m": ols.coef_,
            "opt_m": best_slope,
            "opt_ang": best_angle,
            "improvement": err_ols - err_opt
        })
        
        print(f"Scenario {i:02d}: True m={true_m:6.3f} | OLS m={ols.coef_:6.3f} (Err: {err_ols:.3f}) | OPT m={best_slope:6.3f} (Err: {err_opt:.3f}) | Angle: {best_angle:5.1f} deg")

    # Summary
    avg_improv = np.mean([r['improvement'] for r in results])
    print(f"\nAverage Improvement in Slope Estimation: {avg_improv:.4f}")
    if avg_improv > 0:
        print("SUCCESS: Unsupervised Angle Optimization consistently outperformed OLS.")
    else:
        print("WARNING: OLS performed better on average (Check noise assumptions).")

if __name__ == "__main__":
    main()
