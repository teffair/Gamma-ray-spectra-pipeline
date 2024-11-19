# Estephany Barba Matta
# 18/11/2024
# Radioactive sources spectra pipeline
# Detectors: BGO, NaI(Tl), CdTe.
# Radioactive soruces: Barium, Cobalt, Caesium, Americium.

from scipy.optimize import curve_fit, OptimizeWarning
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import warnings
import argparse
import os

def read_spectrum_file(file_path):
    """
    Reads the .Spe and .mca files to extract spectrum data and metadata (observation date and time, and real time).
    """
    spectrum = []
    metadata = {}
    extension = os.path.splitext(file_path)[-1].lower()

    with open(file_path, 'r') as file:
        lines = file.readlines()

        if extension == '.spe':
            metadata['Observation date and time'] = lines[7].strip()
            metadata['Real time'] = float(lines[9].strip().split()[1])
            spectrum = [int(line.strip()) for line in lines[12:1035] if line.strip().isdigit()]

        elif extension == '.mca':
            metadata['Observation date and time'] = lines[9].strip().split(' - ')[-1]
            metadata['Real time'] = float(lines[8].strip().split()[-1])
            spectrum = [int(line.strip()) for line in lines[12:2059] if line.strip().isdigit()]

        else:
            raise ValueError(f"Unsupported file extension: {extension}")

    if metadata['Real time'] <= 0:
        raise ValueError(f"Invalid 'Real time' value: {metadata['Real time']} in file {file_path}")

    return np.array(spectrum, dtype=float), metadata

def convert_to_cps(spectrum, real_time):
    """
    Convert spectrum to counts per second (cps) using the provided real time.
    """
    if real_time > 0:
        return spectrum / real_time
    else:
        raise ValueError("Unvalid (<0 s) Real time.")

def load_background_cps_data(detector_path):
    """
    Loads and processes background cps data for each detector.
    """
    background_cps_data = {}

    for detector, folder_path in detector_path.items():
        #Build background file name based on the detector name
        if detector in ["NaI(Tl)", "BGO"]:
            background_file = f"{detector}_background.Spe"
        elif detector == "CdTe":
            background_file = f"{detector}_background.mca"
        else:
            raise ValueError(f"Unsupported detector type: {detector}")
    
        background_path = os.path.join(folder_path, background_file) #Builds background file path, example: "./data/NaI(Tl)/NaI(Tl)_background".
    
        if not os.path.isfile(background_path):
            raise FileNotFoundError(
            f"Error: Background file '{background_file}' not found for {detector} in {folder_path}. "
            f"Please ensure that the detector '{detector}' has a background file with the correct naming convention '{detector}_background' in the specified folder '{folder_path}'."
            )

        background_spectrum, metadata = read_spectrum_file(background_path)
        background_cps_data[detector] = convert_to_cps(background_spectrum, metadata['Real time'])

    return background_cps_data

def extract_file_info(file_name, detector_mapping, source_mapping):
    """
    Extract file information such as detector, source, and source angle from the file name.
    """
    #Removing file extensions to avoid including .Spe or .mca in titles
    base_name = file_name.replace(".Spe", "").replace(".mca", "").replace(".spe", "").replace(".MCA", "")

    if "background" in base_name.lower():
        detector_name = base_name.split("_")[0]
        return {"Detector": detector_mapping.get(detector_name.lower(), detector_name), 
                "Source": "Background", 
                "Source angle": 0
        }
    
    parts = base_name.split("_")
    if len(parts) == 3:
        detector_name = detector_mapping.get(parts[0].lower(), parts[0])
        source_name = source_mapping.get(parts[1].upper(), parts[1])
        source_angle = int(parts[2])
        
        return {
            "Detector": detector_name,
            "Source": source_name,
            "Source angle": source_angle
        }
    else:
        raise ValueError(f"Error: Unexpected file name format for '{file_name}'. Expected format is '<Detector>_<Source>_<Angle>'.")

def load_spectra_from_folder(folder_path, background_cps):
    """
    Loads and processes spectra data from a specified folder.
    """
    
    spectra_data = {}

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith((".spe", ".mca")):
            file_path = os.path.join(folder_path, file_name)
            
            spectrum, metadata = read_spectrum_file(file_path)
            file_info = extract_file_info(file_name, detector_mapping, source_mapping)
            is_background_file = "background" in file_name.lower()

            #Include metadata in file_info
            file_info['Real time'] = metadata['Real time']
            file_info['Observation date and time'] = metadata['Observation date and time']

            #Assign background cps only if not a background file
            file_background_cps = None if is_background_file else background_cps
            
            spectra_data[file_name] = {
                "spectrum": spectrum,
                "metadata": metadata,
                "file_info": file_info,
                "background_cps": file_background_cps
                }
        else:
            raise ValueError(f"Unsupported file type encountered: {file_name}")
    return spectra_data

def plot_spectrum_base(data, file_info, ylabel, color, background_cps, label='Spectrum', plot_type='raw'):
    """
    Plots a basic spectrum graph with customizable options.
    """
    plt.figure(figsize=(9, 6))
    plt.plot(data, drawstyle='steps-mid', color=color, label=label)

    if file_info.get('Source', '').lower() == 'background':
        title = f"{file_info['Detector']} detector - Background"
    else:
        title = f"{file_info['Detector']} detector - {file_info['Source']}"
        if file_info.get('Source angle', 0) == 0:
            title += " at 0°"
        elif file_info.get('Source angle', 0) != 0:
            title += f" at {file_info['Source angle']}°"

    if plot_type == 'cps_subtracted' and background_cps is not None:
        title += " (Background Subtracted)"

    plt.title(title)
    plt.xlabel('Channel')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_cps_subtracted_spectrum(spectrum, metadata, file_info, background_cps):
    """
    Plots the spectrum as cps with background subtracted.
    """
    spectrum_cps = convert_to_cps(spectrum, metadata['Real time'])
    net_spectrum = spectrum_cps - background_cps
    net_spectrum[net_spectrum < 0] = 0
    plot_spectrum_base(net_spectrum, file_info, ylabel='Counts / Second', color='purple', background_cps=background_cps, plot_type='cps_subtracted')
    
def plot_cps_spectrum(spectrum, metadata, file_info):
    """
    Plots the spectrum as cps.
    """
    spectrum_cps = convert_to_cps(spectrum, metadata['Real time'])
    plot_spectrum_base(spectrum_cps, file_info, ylabel='Counts / Second', color='green', background_cps=None, plot_type='cps')

def plot_raw_spectrum(spectrum, file_info):
    """
    Plots the raw spectrum data.
    """
    plot_spectrum_base(spectrum, file_info, ylabel='Counts', color='blue', background_cps=None, plot_type='raw')

def plot_spectrum(spectrum, metadata, file_info, plot_type, background_cps=None):
    """
    Calls the appropriate plotting function depending on plot_type, which is chosen by the user.
    """    
    if plot_type == 'cps':
        plot_cps_spectrum(spectrum, metadata, file_info)
            
    elif plot_type == 'cps_subtracted':
        plot_cps_subtracted_spectrum(spectrum, metadata, file_info, background_cps)
            
    elif plot_type == 'raw':
        plot_raw_spectrum(spectrum, file_info)
            
    else:
        print(f"Invalid plot type specified.")

def load_and_plot_spectra(detector, folder_path, background_cps_data, plot_type):
    """
    Loads spectra data from its folder, assignates its background cps data, and plots each spectrum based on the plot type.
    """
    background_cps = background_cps_data.get(detector, None)
    spectra_data = load_spectra_from_folder(folder_path, background_cps=background_cps)

    for file_name, data in spectra_data.items():
        if data["file_info"]["Source"].lower() == "background" and plot_type == 'cps_subtracted':
            continue 
            
        plot_spectrum(
            spectrum=data["spectrum"],
            metadata=data["metadata"],
            file_info=data["file_info"],
            plot_type=plot_type,
            background_cps=data["background_cps"]
        )

def parse_rois(file_path):
    """
    Parses the ROIs from ROIs.txt. Returns a dictionary with detector names as keys, each containing a dictionary of sources and their ROIs.
    """
    rois = {}
    with open(file_path, 'r') as file:
        current_detector = None
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            if line.replace("(", "").replace(")", "").isalpha():  #Check if the line is a detector name by removing parentheses and verifying it contains only letters

                current_detector = detector_mapping.get(line.lower(), line)
                if current_detector not in rois:
                    rois[current_detector] = {}
            else:
                parts = line.split()
                source = parts[0]
                rois[current_detector][source] = [
                    (int(parts[i]), int(parts[i + 1])) for i in range(1, len(parts), 2)
                ]
    return rois

def gaussian(x, A, mu, sigma):
    """
    Gaussian function.
    """
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def fit_gaussian_to_roi(spectrum, roi):
    """
    Fits a Gaussian curve to the ROI in a spectrum.
    """
    start_channel, end_channel = roi
    roi_channels = np.arange(start_channel, end_channel + 1)
    roi_counts = spectrum[start_channel:end_channel + 1]
    initial_guess = [np.max(roi_counts), (start_channel + end_channel) / 2, (end_channel - start_channel) / 4]
    try:
        # Attempt Gaussian fit
        popt, pcov = curve_fit(gaussian, roi_channels, roi_counts, p0=initial_guess, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))  # Parameter errors
        return popt, perr, roi_channels, roi_counts
    except Exception as e:
        # Catch any exception (e.g., RuntimeError, ValueError) and silently return None
        return None, None, roi_channels, roi_counts

def subtract_background(spectrum_cps, background_cps):
    """
    Subtracts the background cps from the spectrum cps.
    """
    return np.maximum(spectrum_cps - background_cps, 0)  #For no negative counts after subtraction

def process_cps_subtracted_with_rois(detector_name, source_name, spectrum, background_spectrum, real_time, rois):
    """
    Processes the spectrum using ROIs for the specified detector and source.
    """
    detector_name = detector_mapping.get(detector_name.lower(), detector_name)
    source_name = source_name.upper()

    spectrum_cps = convert_to_cps(spectrum, real_time)
    background_cps = convert_to_cps(background_spectrum, real_time)
    net_spectrum_cps = subtract_background(spectrum_cps, background_cps)
    if detector_name not in rois or source_name not in rois[detector_name]:
        print(f"No photopeaks identified for '{source_name}' for detector '{detector_name}'.")
        return
    source_rois = rois[detector_name][source_name]

    #Determine the plot's x-axis limits based on the ROIs, for a good view of the photopeaks
    roi_channels = [channel for roi in source_rois for channel in roi]
    x_min = max(0, min(roi_channels) - 50)
    x_max = max(roi_channels) + 50
    
    #Plot CPS-subtracted spectrum in grey
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(net_spectrum_cps, label='Background subtracted spectrum', color='gray', alpha=0.5)
    table_data = []

    global_max_counts = 0 #For y-axis scaling

    for i, roi in enumerate(source_rois):
        fit_params, fit_errors, roi_channels, roi_counts = fit_gaussian_to_roi(net_spectrum_cps, roi)
        
        if fit_params is not None:
            A, mu, sigma = fit_params
            dA, dmu, dsigma = fit_errors
            #Table of parameters and their errors
            table_data.append([f'Peak {i+1}', f'{A:.2f} ± {dA:.2f}', f'{mu:.2f} ± {dmu:.2f}', f'{sigma:.2f} ± {dsigma:.2f}'])
            
            global_max_counts = max(global_max_counts, np.max(gaussian(roi_channels, *fit_params))) #counts for y-axis scaling

            full_range_gaussian = gaussian(np.arange(len(net_spectrum_cps)), *fit_params)
            ax.plot(full_range_gaussian, linestyle='--', label=f'Peak {i+1} Gaussian fit')
            
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, global_max_counts * 1.2) #Scaling y-axis, for a good view of the photopeaks
    ax.set_xlabel('Channel')
    ax.set_ylabel('Counts / Second')
    display_detector_name = "NaI(Tl)" if detector_name == "NaI(Tl)" else detector_name
    ax.set_title(f"{display_detector_name} detector - {source_mapping.get(source_name, source_name)} spectrum at 0°")
    ax.legend()

    #Table below the plot for fit parameters
    column_labels = ["Peak", "Amplitude (A)", "Mean (μ)", "Std Dev (σ)"]
    table = plt.table(cellText=table_data, colLabels=column_labels, cellLoc='center', loc='bottom', bbox=[0, -0.35, 1, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.subplots_adjust(bottom=0.35)
    plt.show()

def calculate_R(energies, delta_E):
    """
    Calculates the energy resolution (R).
    """
    return delta_E / np.array(energies)

def energy_res_line(x, a, b, c):
    """
    Defines the fitting function for energy resolution.
    """
    return np.sqrt(a/x**2 + b/x + c)

def plot_energy_resolution_with_fit(energies, R_values, detector_name):
    """
    Plots the energy resolution with a fitted polynomial curve.
    """
    with warnings.catch_warnings(): #Supress warning for curve fitting of CdTe, since it is expected due to insufficient data
        warnings.simplefilter("ignore", OptimizeWarning)
        
        initial_guess = [1e-3, 1e-3, 1e-3]
        popt, pcov = curve_fit(energy_res_line, energies, R_values, p0=initial_guess, maxfev=10000)

    x_fit = np.linspace(min(energies), max(energies), 500)
    y_fit = energy_res_line(x_fit, *popt)
    
    plt.figure(figsize=(9, 6))
    plt.plot(energies, R_values, 'o', label=f'{detector_name} Energy Resolution Data')
    plt.plot(x_fit, y_fit, '-', label=f'Fit: a={popt[0]:.2g}, b={popt[1]:.2g}, c={popt[2]:.2g}')
    plt.xscale('log')
    plt.yscale('log')  
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Energy Resolution (R) (%)')
    plt.title(f'Energy Resolution for {detector_name} Detector with Fitted Polynomial')
    plt.grid(True)
    plt.legend()
    plt.show()

def calculate_energy_resolution(gaussian_results_by_detector, rois):
    """
    Calculates the energy resolution (R) for each detector using the standar dev. (sigma) parameters of the ROIs Gaussian fits.
    """
    R_values = {}

    for detector_name, detector_rois in rois.items():
        if detector_name not in gaussian_results_by_detector:
            continue 
            
        #Extract the sigma (standard deviation) values from Gaussian results for the current detector
        gaussian_sigmas = np.array(gaussian_results_by_detector[detector_name]["sigmas"])
        #Calculate the FWHM (delta_E) using sigma
        delta_E = gaussian_sigmas * 2 * np.sqrt(2 * np.log(2))  

        energies = []
        for source_name, roi_list in detector_rois.items():
            for roi in roi_list:
                peak_index = len(energies) 
                if peak_index < len(gaussian_results_by_detector[detector_name]["means"]):
                    energies.append(gaussian_results_by_detector[detector_name]["means"][peak_index])

        R = calculate_R(energies, delta_E[:len(energies)])
        R_values[detector_name] = R
    return R_values

def time_difference(measurement_date, lab_date):
    """
    Used for calculating difference in years for the source activity calculations.
    """
    measurement_date = datetime.strptime(measurement_date, '%Y-%m-%d %H:%M:%S %Z')
    lab_date = datetime.strptime(lab_date, '%Y-%m-%d %H:%M:%S %Z')

    difference_in_days = (lab_date - measurement_date).days
    years_difference = difference_in_days / 365.2422 
    return years_difference
measurement_date = '1979-02-01 12:00:00 GMT'
lab_date = '2024-10-01 12:00:00 GMT'

#years_difference = time_difference(measurement_date, lab_date)
#print(f'The number of years between the measurement date and the lab date is {years_difference:.2f}')

def extract_gaussian_params_by_detector(detector_path, rois, background_cps_data):
    """
    Runs through all available spectra for each detector, fits Gaussians on defined ROIs, and stores the mean (μ) values for each detector.
    """
    #Current (01/10/2024) radioactive source activities in Bq. Calculated with the radioactve decay formula
    source_activities = {
        "AM": 409891.148,
        "BA": 19846.755,
        "CS": 160352.672,
        "CO": 1032.773
    }

    gaussian_results = {
        "BGO": {"means": [], "sigmas": [], "amplitudes": [], "x_values": [], "y_values": [], "count_rates": [], "abs_efficiency": []},
        "CdTe": {"means": [], "sigmas": [], "amplitudes": [], "x_values": [], "y_values": [], "count_rates": [], "abs_efficiency": []},
        "NaI(Tl)": {"means": [], "sigmas": [], "amplitudes": [], "x_values": [], "y_values": [], "count_rates": [], "abs_efficiency": []}
    }

    for detector_name, sources  in rois.items():
        if detector_name not in detector_path:
            continue
            
        background_cps = background_cps_data.get(detector_name, None)

        folder_path = detector_path.get(detector_name, None)
        if not folder_path:
            print(f"Folder path for detector {detector_name} not found.")
            continue

        for source_name, roi_list in sources.items():
            #Construct a spectrum file path based on the detector and source
            spectrum_file_path = f"./data/{detector_name}/{detector_name}_{source_name}_0.Spe" if detector_name in ["NaI(Tl)", "BGO"] else f"./data/{detector_name}/{detector_name}_{source_name}_0.mca"

            if not os.path.exists(spectrum_file_path):
                print(f"Skipping {spectrum_file_path} - file not found.")
                continue
            
            spectrum, metadata = read_spectrum_file(spectrum_file_path)
            real_time = metadata.get('Real time', None)
            
            if real_time is None or real_time <= 0:
                print(f"Skipping {spectrum_file_path} due to invalid real time.")
                continue
                
            spectrum_cps = convert_to_cps(spectrum, real_time)
            net_spectrum_cps = subtract_background(spectrum_cps, background_cps)
            
            for roi in roi_list:
                fit_params, gaus_y_fit, gaus_x_fit, _ = fit_gaussian_to_roi(net_spectrum_cps, roi)
                if fit_params is not None:
                    amplitude = fit_params[0]  
                    mu = fit_params[1]  
                    sigma = fit_params[2]  

                    #Calculate count rate by summing
                    count_rate = np.sum(gaus_y_fit)
                    
                    gaussian_results[detector_name]["amplitudes"].append(amplitude)
                    gaussian_results[detector_name]["means"].append(mu)
                    gaussian_results[detector_name]["sigmas"].append(sigma)
                    gaussian_results[detector_name]["x_values"].append(gaus_x_fit)
                    gaussian_results[detector_name]["y_values"].append(gaus_y_fit)
                    gaussian_results[detector_name]["count_rates"].append(count_rate)
                    
                    source_key = source_name.upper()
                    if source_key in source_activities:
                        abs_efficiency = count_rate / source_activities[source_key]
                        gaussian_results[detector_name]["abs_efficiency"].append((source_name, abs_efficiency))
                    else:
                        gaussian_results[detector_name]["abs_efficiency"].append((source_name, None))

                else:
                    print(f"Failed to fit Gaussian for {detector_name} {source_name} ROI {roi}")

    return gaussian_results

def plot_quadratic_calibration_curve(gaussian_results, energies, detector_name):
    """
    Plots a quadratic (or linear if insufficient data points) calibration curve for a detector.
    """
    gaussian_means = np.array(gaussian_results["means"])
    energies = np.array(energies)

    if len(gaussian_means) < 4:
        print(f"Insufficient data points for quadratic fit for {detector_name}. Fitting a linear polynomial instead.")
        degree = 1  
    else:
        degree = 2 

    coeffs, cov_matrix = np.polyfit(gaussian_means, energies, degree, cov=True)
    poly_fit = np.poly1d(coeffs)
    
    errors = np.sqrt(np.diag(cov_matrix))
    
    x_fit = np.linspace(min(gaussian_means), max(gaussian_means), 100)
    y_fit = poly_fit(x_fit)

    plt.figure(figsize=(9, 7))
    plt.scatter(gaussian_means, energies, color='blue', label='Peak channel (μ)')

    if degree == 2:
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Calibration function: y = {coeffs[0]:.2g}x² + {coeffs[1]:.2g}x + {coeffs[2]:.2g}')
    else:
        plt.plot(x_fit, y_fit, color='red', linestyle='--', label=f'Calibration function: y = {coeffs[0]:.2g}x + {coeffs[1]:.2f}')

    plt.xlabel('Channel')
    plt.ylabel('Energy (keV)')
    plt.title(f'{detector_name} - Calibration curve')
    plt.legend()
    plt.grid(True)
    
    table_data = [["Coefficient", "Value ± Error"]]
    if degree == 2:
        table_data.append(["a", f"{coeffs[0]:.2g} ± {errors[0]:.2g}"])
        table_data.append(["b", f"{coeffs[1]:.2g} ± {errors[1]:.2g}"])
        table_data.append(["c", f"{coeffs[2]:.2g} ± {errors[2]:.2g}"])
    else:
        table_data.append(["a", f"{coeffs[0]:.2g} ± {errors[0]:.2g}"])
        table_data.append(["b", f"{coeffs[1]:.2g} ± {errors[1]:.2g}"])
   
    
    table = plt.table(cellText=table_data, colLabels=None, cellLoc='center', loc='bottom', bbox=[0, -0.5, 1, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.subplots_adjust(bottom=0.35)
    plt.show()


def efficiencies(gaussian_results_by_detector, energies_nai, energies_cdte, energies_bgo):
    """
    Calculates the intrinsic efficiency for each detector based on absolute efficiency values and the geometry factor (G factor).
    """
    detectors = {
        "NaI(Tl)": (energies_nai, 0.011449093),
        "CdTe": (energies_cdte, 8.84194E-05),
        "BGO": (energies_bgo, 0.009127147)
    }


    intrinsic_efficiency_by_detector = {}
    
    for detector, (energies, g_factor) in detectors.items():
        abs_efficiency_values = [eff[1] for eff in gaussian_results_by_detector[detector]["abs_efficiency"] if eff[1] is not None]

        if len(abs_efficiency_values) != len(energies):
            print(f"Warning: Mismatch in number of absolute efficiency values and energies for {detector}.")

        intrinsic_efficiency_values = [eff / g_factor for eff in abs_efficiency_values]
        intrinsic_efficiency_by_detector[detector] = intrinsic_efficiency_values
    return intrinsic_efficiency_by_detector

def fit_log_polynomial(energies, intrinsic_efficiencies, detector_name):
    """
    Fits a polynomial curve to the intrinsic efficiency data and plots it.
    """
    log_energies = np.log(energies)
    log_intrinsic_efficiencies = np.log(intrinsic_efficiencies)

    coeffs, cov_matrix = np.polyfit(log_energies, log_intrinsic_efficiencies, 2, cov=True)
    poly_fit = np.poly1d(coeffs)

    errors = np.sqrt(np.diag(cov_matrix))

    log_energies_fit = np.linspace(min(log_energies), max(log_energies), 100)
    log_intrinsic_fit = poly_fit(log_energies_fit)

    energies_fit = np.exp(log_energies_fit)
    intrinsic_fit = np.exp(log_intrinsic_fit)

    plt.figure(figsize=(8, 8))
    plt.scatter(energies, intrinsic_efficiencies, marker='o', linestyle='-', label=f'{detector_name} Intrinsic Efficiency')

    plt.plot(energies_fit, intrinsic_fit, color='red', linestyle='--',
             label=f'Fit: $\\ln \\epsilon_p = {coeffs[2]:.2f} + {coeffs[1]:.2f} \\ln E + {coeffs[0]:.2f} (\\ln E)^2$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Intrinsic Efficiency (%)')
    plt.title(f'{detector_name} - Intrinsic Efficiency with Polynomial Fit')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()
    
    #Table for fit parameters and their errors
    table_data = [
        ["Parameter", "Value ± Error"],
        ["a (constant)", f"{coeffs[2]:.2g} ± {errors[2]:.2g}"],
        ["b (linear term)", f"{coeffs[1]:.2g} ± {errors[1]:.2g}"],
        ["c (quadratic term)", f"{coeffs[0]:.2g} ± {errors[0]:.2g}"]
    ]
    plt.table(cellText=table_data, colLabels=None, cellLoc='center', loc='bottom', bbox=[0, -0.3, 1, 0.2])
    plt.subplots_adjust(bottom=0.3) 
    plt.show()

def off_axis_response(detector_name, source_name, folder_path, background_cps_data, rois):
    """
    Plots all spectra for a given detector and source at different angles.
    
    Gaussian fittings are done again (only if user wants off-axis plots) for each angle instead of precomputing for all angles.
    Precomputing for all angles was unnecessary earlier as only on-axis was needed for previous analyses.
    Now, all angles are considered for visualization.
    """
    detector_name = detector_mapping.get(detector_name.lower(), detector_name)
    source_name = source_name.upper()
    background_cps = background_cps_data[detector_name]

    source_rois = rois[detector_name][source_name]
    spectra_data = load_spectra_from_folder(folder_path, background_cps=background_cps)

    plt.figure(figsize=(10, 7))
    
    global_max_count = 0 #For y-axis scaling
    global_x_min = float('inf') #For x-axis scaling
    global_x_max = float('-inf')

    angles = sorted({data['file_info']['Source angle'] for data in spectra_data.values()})

    #Colourmap for the angles for a good visualization, form blue to green to red.
    cmap = mcolors.LinearSegmentedColormap.from_list("angle_colormap", ["blue", "green", "red"], len(angles))
    angle_color_map = {angle: cmap(i / (len(angles) - 1)) for i, angle in enumerate(angles)}

    for data in spectra_data.values():
        file_info = data['file_info']
        angle = int(file_info['Source angle'])

        if file_info['Source'] == source_mapping[source_name]:
            spectrum = data['spectrum']
            metadata = data['metadata']
            color = angle_color_map[angle]

            spectrum_cps = convert_to_cps(spectrum, metadata['Real time'])
            if background_cps is not None:
                spectrum_cps = subtract_background(spectrum_cps, background_cps)

            for roi in source_rois:
                start_channel, end_channel = roi
                fit_params, _, _, _ = fit_gaussian_to_roi(spectrum_cps, roi)
                if fit_params is not None:
                    full_range_gaussian = gaussian(np.arange(len(spectrum_cps)), *fit_params)
                    plt.plot(np.arange(len(spectrum_cps)), full_range_gaussian, color=color, label=f"{angle}°")

                    global_max_count = max(global_max_count, np.max(full_range_gaussian[start_channel:end_channel + 1]))
                    global_x_min = min(global_x_min, max(0, start_channel - 20))
                    global_x_max = max(global_x_max, end_channel + 10)

    plt.xlim(global_x_min, global_x_max)
    plt.ylim(0, global_max_count * 1.2)
    plt.xlabel('Channel')
    plt.ylabel('Counts / Second')
    plt.title(f"Off-Axis Response for {detector_name} - {source_mapping[source_name]} (ROIs with Gaussian Fits)")
    plt.legend(title='Angle', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def display_menu():
    """
    Displays a menu for the spectra analysis script
    """
    print("=" * 60)
    print(" Welcome to the Spectra Analysis Script ".center(60, "="))
    print("=" * 60)
    print("\nThis script will guide you through multiple steps for analyzing gamma ray spectra.\n")
    print("Steps include:")
    print("1. Basic graph plotting (raw, counts per second (CPS), or CPS with background subtracted).")
    print("2. Gaussian curve fitting for photopeaks.")
    print("3. Detector calibration plots.")
    print("4. Efficiency curve plots (absolute and intrinsic).")
    print("5. Energy resolution plots.")
    print("6. Off-axis response.\n")
    print("Note:")
    print("- For each step, you'll be prompted for input.")
    print("- If you wish to skip a step, simply press 'Enter' without typing anything.\n")
    print("=" * 60)
    print()


detector_path = {
    "NaI(Tl)": "./data/NaI(Tl)",
    "CdTe": "./data/CdTe",
    "BGO": "./data/BGO"
}

source_mapping = {
    "AM": "$^{241}$Am",
    "CS": "$^{137}$Cs",
    "BA": "$^{133}$Ba",
    "CO": "$^{60}$Co"
}

detector_mapping = {
    "nai(tl)": "NaI(Tl)",
    "cdte": "CdTe",
    "bgo": "BGO"
}

rois_file_path = "./data/ROIs.txt"

def main():
    """
    Main function to run the spectra analysis.
    """
    display_menu()

    background_cps_data = load_background_cps_data(detector_path)
    rois = parse_rois(rois_file_path)
    gaussian_results_by_detector = extract_gaussian_params_by_detector(detector_path, rois, background_cps_data)

    energies_nai = [80.9979, 356.0129, 26.3446, 59.5409, 661.657]
    energies_cdte = [356.0129, 59.5409, 661.657]
    energies_bgo = [80.9979, 302.8508, 356.0129, 59.5409, 661.657, 1173.228, 1332.492]

    
    detector_calibration_mapping = {
        "NaI(Tl)": (gaussian_results_by_detector["NaI(Tl)"], energies_nai),
        "CdTe": (gaussian_results_by_detector["CdTe"], energies_cdte),
        "BGO": (gaussian_results_by_detector["BGO"], energies_bgo)
    }

    intrinsic_efficiency_results = efficiencies(gaussian_results_by_detector, energies_nai, energies_cdte, energies_bgo)

    # Step 1: Basic graph plotting
    print("Step 1: Choose a detector and plot the graphs (raw, cps, or background-subtracted).")

    user_input_detector = input("Which detector spectra would you like to plot? Enter 'BGO', 'NaI(Tl)', or 'CdTe': ").strip().lower()
    detector = detector_mapping.get(user_input_detector, user_input_detector)
    
    if detector not in detector_path:
        print(f"Invalid detector selection: '{user_input_detector}'. Please enter 'BGO', 'NaI(Tl)', or 'CdTe'.")
    else:
        user_input_source = input("Enter the source name ('AM' for Americium, 'CO' for Cobalt, 'CS' for Caesium, 'BA' for Barium): ").strip().upper()

        if user_input_source not in source_mapping:
            print(f"Invalid source selection: '{user_input_source}'. Please enter a valid source code ('AM', 'CO', 'CS', or 'BA'")
        else:
            # Prompt user for plot type
            plot_type = input("Enter 'raw', 'cps', or 'cps_subtracted' for the plot type: ").strip()
            if plot_type not in ['raw', 'cps', 'cps_subtracted']:
                print(f"Invalid plot type specified: '{plot_type}'. Please enter 'raw', 'cps', or 'cps_subtracted'.")
            else:
                # Load and plot spectra for the chosen detector and source
                folder_path = detector_path[detector]
                spectra_data = load_spectra_from_folder(folder_path, background_cps_data.get(detector))

                # Filter spectra by source
                for file_name, data in spectra_data.items():
                    if data["file_info"]["Source"] == source_mapping[user_input_source]:
                        plot_spectrum(
                            spectrum=data["spectrum"],
                            metadata=data["metadata"],
                            file_info=data["file_info"],
                            plot_type=plot_type,
                            background_cps=data["background_cps"]
                        )

    #Step 2: Advanced ROI Analysis
    print("\nStep 2: Gaussian curve fitting for photopeaks.")
    
    user_input_detector = input("Enter detector name (e.g., 'CdTe', 'NaI(Tl)', 'BGO'): ").strip().lower()
    detector_name = detector_mapping.get(user_input_detector.lower(), user_input_detector)
    source_name = input("Enter the source name ('AM' for Americium, 'CO' for Cobalt, 'CS' for Caesium, 'BA' for Barium): ").strip().upper()
    
    if detector_name not in rois:
        print(f"Invalid detector selection: '{user_input_detector}'.Please enter 'BGO', 'NaI(Tl)', or 'CdTe'.")
    else:
        file_extension = ".mca" if detector_name == "CdTe" else ".Spe" if detector_name in ["NaI(Tl)", "BGO"] else None
        if file_extension is None:
            print(f"Unknown file type for detector '{detector_name}'.")
        else:
            #Construct file paths based on user input and file extension
            spectrum_file_path = f"./data/{detector_name}/{detector_name}_{source_name}_0{file_extension}"
            background_file_path = os.path.join(detector_path[detector_name], f"{detector_name}_background.Spe" if detector_name in ["NaI(Tl)", "BGO"] else f"{detector_name}_background.mca")
    
            if not os.path.exists(spectrum_file_path):
                print(f"Spectrum file not found at path: {spectrum_file_path}")
            elif not os.path.exists(background_file_path):
                print(f"Background file not found at path: {background_file_path}")
            else:
                spectrum, metadata = read_spectrum_file(spectrum_file_path)
                background_spectrum, _ = read_spectrum_file(background_file_path)

                real_time = metadata.get('Real time', None)

                process_cps_subtracted_with_rois(detector_name, source_name, spectrum, background_spectrum, real_time, rois)
    
    #Step 3: Calibration curve
    print("\nStep 3: Detector calibration plots.")
    user_input_detector = input("Enter detector name (e.g., 'CdTe', 'NaI(Tl)', 'BGO'): ").strip().lower()
    
    detector_calibration_mapping_lowercase = {
        "nai(tl)": ("NaI(Tl)", gaussian_results_by_detector["NaI(Tl)"], energies_nai),
        "cdte": ("CdTe", gaussian_results_by_detector["CdTe"], energies_cdte),
        "bgo": ("BGO", gaussian_results_by_detector["BGO"], energies_bgo)
    } 

    
    if user_input_detector in detector_calibration_mapping_lowercase:
        detector_name, gaussian_results, energies = detector_calibration_mapping_lowercase[user_input_detector]
        plot_quadratic_calibration_curve(gaussian_results, energies, detector_name)
    else:
        print(f"Invalid detector selection: '{user_input_detector}'.Please enter 'BGO', 'NaI(Tl)', or 'CdTe'.")

    # Step 4: Efficiency Plots
    print("\nStep 4: Efficiency curve plots (absolute and intrinsic).")
    user_input_detector = input("Enter detector name (e.g., 'CdTe', 'NaI(Tl)', 'BGO'): ").strip().lower()
    detector_name = detector_mapping.get(user_input_detector, None) #estefue el primero

    if detector_name == "CdTe":
        #Plot only the Absolute Efficiency for CdTe
        abs_efficiency_values = [eff[1] for eff in gaussian_results_by_detector["CdTe"]["abs_efficiency"] if eff[1] is not None]
        plt.figure(figsize=(9, 6))
        plt.scatter(energies_cdte, abs_efficiency_values, marker='o', linestyle='-', label='CdTe Absolute Efficiency')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Absolute Efficiency (%)')
        plt.title('CdTe - Absolute Efficiency')
        plt.grid(True)
        plt.legend()
        plt.show()

    elif detector_name in ["NaI(Tl)", "BGO"]:
        abs_efficiency_values = [eff[1] for eff in gaussian_results_by_detector[detector_name]["abs_efficiency"] if eff[1] is not None]
        intrinsic_efficiency_values = intrinsic_efficiency_results[detector_name]

        #Absolute Efficiency 
        plt.figure(figsize=(9, 6))
        energies = energies_nai if detector_name == "NaI(Tl)" else energies_bgo
        plt.scatter(energies, abs_efficiency_values, marker='o', linestyle='-', label=f'{detector_name} Absolute Efficiency')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Absolute Efficiency (%)')
        plt.title(f'{detector_name} - Absolute Efficiency')
        plt.grid(True)
        plt.legend()
        plt.show()

        #Intrinsic Efficiency
        fit_log_polynomial(energies, intrinsic_efficiency_values, detector_name)

    else:
        print(f"Invalid detector selection: '{user_input_detector}'.Please enter 'BGO', 'NaI(Tl)', or 'CdTe'.")

    #Step 5: Energy resolution
    print("\nStep 5: Energy Resolution plots")
    user_input_detector = input("Enter detector name (e.g., 'CdTe', 'NaI(Tl)', 'BGO'): ").strip().lower()
    detector_name = detector_mapping.get(user_input_detector, None)
    
    R_values = calculate_energy_resolution(gaussian_results_by_detector, parse_rois(rois_file_path))

    if detector_name in R_values:
        energies = gaussian_results_by_detector[detector_name]["means"][:len(R_values[detector_name])]
        plot_energy_resolution_with_fit(energies, R_values[detector_name], detector_name)
    else:
        print(f"Invalid detector selection: '{user_input_detector}'.Please enter 'BGO', 'NaI(Tl)', or 'CdTe'.")

    #Step 6: Off axis reponse
    print("\nStep 6: Off-Axis Response.")
    user_input_detector = input("Enter detector name (e.g., 'CdTe', 'NaI(Tl)', 'BGO'): ").strip().lower()
    detector_name = detector_mapping.get(user_input_detector, None)
    source_choice = input("Enter the source ('AM' for Americium, 'CO' for Cobalt, 'CS' for Caesium, 'BA' for Barium): ").strip().upper()
    
    if detector_name not in rois:
        print(f"Invalid detector selection: '{user_input_detector}'.Please enter 'BGO', 'NaI(Tl)', or 'CdTe'.")
        return

    if source_choice not in rois[detector_name]:
        print(f"No photopeaks identified for '{source_choice}' for detector '{user_input_detector}'.")
        return

    folder_path = detector_path[detector_name]
    off_axis_response(detector_name, source_choice, folder_path, background_cps_data, rois)

#Run main function
if __name__ == "__main__":
    main()
