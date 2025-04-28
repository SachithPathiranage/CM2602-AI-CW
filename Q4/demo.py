import numpy as np
import matplotlib.pyplot as plt
from Anomaly_detector import FuzzyGridAnomalyDetector

def run_simulation_scenario(detector, scenario_name, voltage_range, frequency_range, load_range, num_steps=10):
    """Run a simulation scenario with varying parameters"""
    print(f"\n=== Scenario: {scenario_name} ===")
    
    results = []
    
    # Generate parameter values
    voltage_values = np.linspace(voltage_range[0], voltage_range[1], num_steps)
    frequency_values = np.linspace(frequency_range[0], frequency_range[1], num_steps)
    load_values = np.linspace(load_range[0], load_range[1], num_steps)
    
    for i in range(num_steps):
        # Get current values
        voltage = voltage_values[i]
        frequency = frequency_values[i]
        load = load_values[i]
        
        # Detect and correct anomalies
        result = detector.detect_and_correct(voltage, frequency, load)
        results.append(result)
        
        # Print results
        print(f"\nStep {i+1}:")
        print(f"  Inputs: Voltage Deviation = {voltage:.2f}%, Frequency Variation = {frequency:.2f}Hz, Load Imbalance = {load:.2f}%")
        print(f"  Anomaly Severity: {result['anomaly_severity']:.2f}")
        print(f"  Action Intensity: {result['action_intensity']:.2f}")
        print(f"  Recommended Action: {result['corrective_actions']['action_type']} - {result['corrective_actions']['description']}")
    
    return results

def main():
    # Create the anomaly detector
    detector = FuzzyGridAnomalyDetector()
    
    # Plot detector's membership functions
    detector.plot_membership_functions()
    
    # Run different scenarios
    
    # Scenario 1: Gradually increasing voltage deviation
    run_simulation_scenario(
        detector,
        "Increasing Voltage Deviation",
        voltage_range=(-2, 12),
        frequency_range=(0, 0),
        load_range=(5, 5)
    )
    
    # Scenario 2: Gradually worsening frequency instability
    run_simulation_scenario(
        detector,
        "Worsening Frequency Instability",
        voltage_range=(0, 0),
        frequency_range=(-0.1, 0.8),
        load_range=(5, 5)
    )
    
    # Scenario 3: Multiple parameters changing
    run_simulation_scenario(
        detector,
        "Multiple Parameter Anomaly",
        voltage_range=(0, 8),
        frequency_range=(0, 0.5),
        load_range=(5, 40)
    )
    
    # Test specific anomaly cases
    print("\n=== Testing Specific Anomaly Cases ===")
    
    # Case 1: Normal operation
    result = detector.detect_and_correct(0, 0, 5)
    print("\nCase 1: Normal Operation")
    print(f"  Inputs: Voltage Deviation = 0%, Frequency Variation = 0Hz, Load Imbalance = 5%")
    print(f"  Anomaly Severity: {result['anomaly_severity']:.2f}")
    print(f"  Recommended Action: {result['corrective_actions']['action_type']} - {result['corrective_actions']['description']}")
    
    # Case 2: High voltage anomaly
    result = detector.detect_and_correct(12, 0, 5)
    print("\nCase 2: High Voltage Anomaly")
    print(f"  Inputs: Voltage Deviation = 12%, Frequency Variation = 0Hz, Load Imbalance = 5%")
    print(f"  Anomaly Severity: {result['anomaly_severity']:.2f}")
    print(f"  Recommended Action: {result['corrective_actions']['action_type']} - {result['corrective_actions']['description']}")
    
    # Case 3: Critical anomaly (all parameters affected)
    result = detector.detect_and_correct(10, 0.7, 60)
    print("\nCase 3: Critical Anomaly (All Parameters)")
    print(f"  Inputs: Voltage Deviation = 10%, Frequency Variation = 0.7Hz, Load Imbalance = 60%")
    print(f"  Anomaly Severity: {result['anomaly_severity']:.2f}")
    print(f"  Recommended Action: {result['corrective_actions']['action_type']} - {result['corrective_actions']['description']}")

if __name__ == "__main__":
    main()