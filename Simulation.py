import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import odeint

# --- MISSION & COST CONSTANTS ---
MASS_BASELINE = 500.0  # kg
AREA_BASELINE = 2.0    # m^2
CD = 2.2               # Drag coefficient
LAUNCH_COST_PER_KG = 5000.0
BASE_MFG_COST = 15000.0
COST_PER_SQ_METER = 1000.0
SAIL_MASS_PER_SQ_METER = 0.15
DEPLOYER_MASS = 2.0
BASELINE_SAT_COST = 150000000.0 # $150 Million estimated mission cost

# --- PHYSICS CONSTANTS (SI Units) ---
R_EARTH_M = 6371000.0
R_EARTH_KM = 6371.0
MU_EARTH_M3_S2 = 3.986e14

def atmospheric_density(alt_km):
    rho0 = 4.0e-13
    h0 = 600.0
    H = 60.0
    if alt_km < 100:
        return 1.0e-7 
    return rho0 * np.exp(-(alt_km - h0) / H)

def orbital_decay_derivative(alt_km, t, mass, area, Cd):
    if alt_km <= 100:
        return 0.0
    rho = atmospheric_density(alt_km)
    radius_m = R_EARTH_M + (alt_km * 1000.0)
    da_dt_m_s = - (Cd * area / mass) * rho * np.sqrt(MU_EARTH_M3_S2 * radius_m)
    da_dt_km_yr = (da_dt_m_s / 1000.0) * (365.25 * 24 * 3600)
    return da_dt_km_yr

# --- SET UP THE GUI FIGURE ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 9))
fig.canvas.manager.set_window_title('AESS Challenge 4: Interactive Mission Simulator')

# Layout zones
ax_plot = fig.add_axes([0.05, 0.25, 0.60, 0.70]) 
ax_slider_alt = fig.add_axes([0.15, 0.12, 0.45, 0.03], facecolor='#333333')
ax_slider_area = fig.add_axes([0.15, 0.05, 0.45, 0.03], facecolor='#333333')
ax_viz = fig.add_axes([0.68, 0.50, 0.30, 0.45]) 
dash_ax = fig.add_axes([0.68, 0.05, 0.30, 0.40]) 
dash_ax.axis('off')

# --- INITIALIZE MAIN PLOT ---
time_years = np.linspace(0, 100, 1000)
line_base, = ax_plot.plot([], [], label='Baseline Satellite', color='#FF4444', linewidth=2.5)
line_sail, = ax_plot.plot([], [], label='Equipped with Drag Sail', color='#44FF44', linewidth=2.5, linestyle='--')
burn_up_line = ax_plot.axhline(100, color='white', linestyle=':', label='Burn-up Altitude (100 km)')

ax_plot.set_title('Post-Mission Orbital Lifetime Analysis', fontsize=16, fontweight='bold')
ax_plot.set_xlabel('Time (Years)', fontsize=13)
ax_plot.set_ylabel('Altitude (km)', fontsize=13)
ax_plot.grid(True, alpha=0.1)
ax_plot.legend(fontsize=11)

# --- INITIALIZE VISUALIZATION ---
ax_viz.set_title('Operational Behavior Visualization', fontsize=14, color='white')
ax_viz.set_aspect('equal') 
ax_viz.axis('off') 

earth = plt.Circle((0, 0), R_EARTH_KM, color='#223366') 
atmo_halo = plt.Circle((0, 0), R_EARTH_KM + 100.0, color='#88CCFF', alpha=0.2) 
ax_viz.add_artist(atmo_halo)
ax_viz.add_artist(earth)

sat_point, = ax_viz.plot([], [], 'gs', markersize=12, label='Active Mission')
viz_text = ax_viz.text(0, 0, '', color='white', ha='center', fontsize=12, fontweight='bold')

# Expanded view limit so the exaggerated movement fits on screen
view_limit = R_EARTH_KM + 3500.0
ax_viz.set_xlim(-view_limit, view_limit)
ax_viz.set_ylim(-view_limit, view_limit)

# --- INITIALIZE SLIDERS & DASHBOARD ---
s_alt = Slider(ax_slider_alt, 'Orbit Altitude (km)', 400.0, 800.0, valinit=600.0, valstep=10.0, color='#88CCFF')
s_area = Slider(ax_slider_area, 'Sail Area (m$^2$)', 5.0, 100.0, valinit=25.0, valstep=1.0, color='#44FF44')

dashboard_text = dash_ax.text(0.0, 0.95, '', transform=dash_ax.transAxes, fontsize=11, 
                              verticalalignment='top', family='monospace', color='white',
                              bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=1.0, pad=1))

def update(val=None):
    alt = s_alt.val
    area = s_area.val
    
    # 1. Update Visualization (Exaggerated scale for visual effect)
    visual_alt_multiplier = 4.0 # Makes the movement 4x more obvious on screen
    current_sat_radius_km = R_EARTH_KM + (alt * visual_alt_multiplier)
    sat_point.set_data([0], [current_sat_radius_km])
    viz_text.set_position((0, current_sat_radius_km + 300))
    viz_text.set_text(f"Alt: {alt:.0f} km")
    
    # 2. Update Main Simulation Plot
    mass_added = (area * SAIL_MASS_PER_SQ_METER) + DEPLOYER_MASS
    launch_cost = mass_added * LAUNCH_COST_PER_KG
    mfg_cost = BASE_MFG_COST + (area * COST_PER_SQ_METER)
    total_cost = launch_cost + mfg_cost
    mass_sail_total = MASS_BASELINE + mass_added
    
    alt_base_sim = odeint(orbital_decay_derivative, alt, time_years, args=(MASS_BASELINE, AREA_BASELINE, CD)).flatten()
    alt_sail_sim = odeint(orbital_decay_derivative, alt, time_years, args=(mass_sail_total, area, CD)).flatten()
    
    # Force lines to touch the 100km burn-up mark precisely
    base_deorbit_idx = np.where(alt_base_sim <= 100)[0]
    if len(base_deorbit_idx) > 0:
        idx = base_deorbit_idx[0]
        alt_base_sim[idx] = 100.0 
        alt_base_sim[idx+1:] = np.nan
        
    sail_deorbit_idx = np.where(alt_sail_sim <= 100)[0]
    if len(sail_deorbit_idx) > 0:
        idx = sail_deorbit_idx[0]
        alt_sail_sim[idx] = 100.0 
        alt_sail_sim[idx+1:] = np.nan
    
    line_base.set_data(time_years, alt_base_sim)
    line_sail.set_data(time_years, alt_sail_sim)
    
    deorbit_sail_time = time_years[sail_deorbit_idx[0]] if len(sail_deorbit_idx) > 0 else 100.0
    ax_plot.set_xlim(0, min(100, max(5, deorbit_sail_time * 1.3)))
    ax_plot.set_ylim(0, alt + 30)

    # 3. Update Dashboard Text
    deorbit_base = time_years[base_deorbit_idx[0]] if len(base_deorbit_idx) > 0 else 100.0
    base_str = f"{deorbit_base:.1f} Years" if deorbit_base < 100 else "> 100 Years"
    sail_str = f"{deorbit_sail_time:.1f} Years" if deorbit_sail_time < 100 else "> 100 Years"
    
    dash_content = (
        "MISSION PARAMETERS\n"
        "-------------------------------------\n"
        f"Initial Altitude: {alt:.0f} km\n"
        f"Baseline Mass:    {MASS_BASELINE:.0f} kg\n"
        f"Target Sail Area: {area:.0f} m^2\n\n"
        "SUSTAINABILITY KPIs\n"
        "-------------------------------------\n"
        f"Baseline Decay:   {base_str}\n"
        f"Drag Sail Decay:  {sail_str}\n\n"
        "UPGRADE ECONOMICS\n"
        "-------------------------------------\n"
        f"Base Mission:     ${BASELINE_SAT_COST/1000000:.0f} Million\n"
        f"Sail Hardware:    ${mfg_cost:,.0f}\n"
        f"Launch Penalty:   ${launch_cost:,.0f}\n"
        f"TOTAL SAIL COST:  ${total_cost:,.0f}\n"
        f"BUDGET IMPACT:    +{(total_cost / BASELINE_SAT_COST) * 100:.3f}%\n"
    )
    dashboard_text.set_text(dash_content)
    fig.canvas.draw_idle()

# --- ATTACH LOGIC AND START ---
s_alt.on_changed(update)
s_area.on_changed(update)
update() 
plt.show()