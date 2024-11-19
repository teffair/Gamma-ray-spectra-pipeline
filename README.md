# Gamma Ray Detector Calibration and Characterization pipeline

## Overview

This Python script provides a pipeline for analyzing gamma ray spectra obtained from three types of detectors commonly used in space based applications: NaI(Tl), BGO, and CdTe. The code is designed to facilitate the comparison of detector performance by calculating important characteristics such as energy resolution, efficiency (absolute and intrinsic), calibration, and off-axis response. These analyses highlight each detector's strengths and weaknesses, enabling comparisons to assess their suitability for space based gamma ray astronomy.

## Workflow

1. **Spectral Analysis:**
   - Spectra are processed for different sources and angles.
   - Gaussian fits are performed on photopeaks.

3. **Calibration:**
   - The channel-energy relation is determined by fitting photopeak means against known energies.

4. **Efficiency Analysis:**
   - Absolute and intrinsic efficiencies are calculated and plotted for each detector.

5. **Energy Resolution:**
   - Energy resolution is calculated and plotted as a function of energy, along with polynomial fits.

6. **Off-Axis Response:**
   - Spectra are visualized for sources at different angles, demostrating detector performance off-axis.

## How to Run

### Running the Script
1. Clone the repository containing the script and accompanying data.
2. Run the script:
   ```bash
   python spectrapipeline.py
   ```
3. Follow the interactive prompts to select detectors, sources, and analysis types.

### Input Files
- **Spectral Files:** `.Spe` and `.mca` files for each detector.
- **ROIs:** `ROIs.txt` file defining photopeaks for fitting.

### Note
This code was developed to generate plots and key data required for constructing a detailed report on gamma ray detector performance. The report focuses on comparing different detectors (NaI(Tl), BGO, CdTe) and their suitability for space based applications, particularly in the context of nanosatellite missions. 
