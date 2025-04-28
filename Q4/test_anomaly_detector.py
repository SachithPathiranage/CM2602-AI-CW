import numpy as np
import matplotlib.pyplot as plt
from Anomaly_detector import FuzzyGridAnomalyDetector
import pandas as pd
from datetime import datetime

def generate_test_dataset():
    """
    Generate a test dataset with simulated grid anomalies
    """
    # Create empty lists to store data
    timestamps = []
    voltage_deviations = []
    frequency_variations = []
    load_imbalances = []
    anomaly_present = []
    
    # Generate normal operating conditions
    normal_samples = 20
    for i in range(normal_samples):
        timestamps.append(i)
        voltage_deviations.append(np.random.normal(0, 1))
        frequency_variations.append(np.random.normal(0, 0.05))
        load_imbalances.append(np.random.normal(5, 2))
        anomaly_present.append(False)
    
    # Generate voltage drop anomaly
    voltage_drop_samples = 5
    for i in range(voltage_drop_samples):
        timestamps.append(normal_samples + i)
        voltage_deviations.append(np.random.normal(-12, 1))
        frequency_variations.append(np.random.normal(0, 0.1))
        load_imbalances.append(np.random.normal(8, 3))
        anomaly_present.append(True)
    
    # Generate voltage spike anomaly
    voltage_spike_samples = 5
    for i in range(voltage_spike_samples):
        timestamps.append(normal_samples + voltage_drop_samples + i)
        voltage_deviations.append(np.random.normal(13, 1))
        frequency_variations.append(np.random.normal(0, 0.1))
        load_imbalances.append(np.random.normal(10, 3))
        anomaly_present.append(True)
    
    # Generate frequency anomaly
    frequency_anomaly_samples = 5
    for i in range(frequency_anomaly_samples):
        timestamps.append(normal_samples + voltage_drop_samples + voltage_spike_samples + i)
        voltage_deviations.append(np.random.normal(0, 2))
        frequency_variations.append(np.random.normal(0.8, 0.1))
        load_imbalances.append(np.random.normal(15, 5))
        anomaly_present.append(True)
    
    # Generate load imbalance anomaly
    load_anomaly_samples = 5
    for i in range(load_anomaly_samples):
        timestamps.append(normal_samples + voltage_drop_samples + voltage_spike_samples + 
                         frequency_anomaly_samples + i)
        voltage_deviations.append(np.random.normal(2, 2))
        frequency_variations.append(np.random.normal(0.1, 0.1))
        load_imbalances.append(np.random.normal(70, 10))
        anomaly_present.append(True)
    
    # Generate combined anomaly (multiple parameters are abnormal)
    combined_anomaly_samples = 5
    for i in range(combined_anomaly_samples):
        timestamps.append(normal_samples + voltage_drop_samples + voltage_spike_samples + 
                         frequency_anomaly_samples + load_anomaly_samples + i)
        voltage_deviations.append(np.random.normal(10, 2))
        frequency_variations.append(np.random.normal(0.6, 0.1))
        load_imbalances.append(np.random.normal(60, 10))
        anomaly_present.append(True)
    
    # More normal conditions to end the dataset
    for i in range(normal_samples):
        timestamps.append(normal_samples + voltage_drop_samples + voltage_spike_samples + 
                         frequency_anomaly_samples + load_anomaly_samples + combined_anomaly_samples + i)
        voltage_deviations.append(np.random.normal(0, 1))
        frequency_variations.append(np.random.normal(0, 0.05))
        load_imbalances.append(np.random.normal(5, 2))
        anomaly_present.append(False)
    
    # Create DataFrame from the lists
    df = pd.DataFrame({
        'timestamp': timestamps,
        'voltage_deviation': voltage_deviations,
        'frequency_variation': frequency_variations,
        'load_imbalance': load_imbalances,
        'anomaly_present': anomaly_present
    })
    
    return df

def evaluate_detector_performance(detector, test_data, anomaly_threshold=40):
    """
    Evaluate the detector's performance on test data
    
    Args:
        detector: FuzzyGridAnomalyDetector instance
        test_data: DataFrame with test data
        anomaly_threshold: Severity threshold to classify as anomaly (0-100)
        
    Returns:
        Dict with performance metrics
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    results = []
    
    for _, row in test_data.iterrows():
        # Get the actual values
        voltage_deviation = row['voltage_deviation']
        frequency_variation = row['frequency_variation']
        load_imbalance = row['load_imbalance']
        actual_anomaly = row['anomaly_present']
        
        # Detect anomalies
        result = detector.detect_and_correct(voltage_deviation, frequency_variation, load_imbalance)
        results.append(result)
        
        # Determine if the detector found an anomaly
        detected_anomaly = result['anomaly_severity'] > anomaly_threshold
        
        # Update confusion matrix
        if actual_anomaly and detected_anomaly:
            true_positives += 1
        elif actual_anomaly and not detected_anomaly:
            false_negatives += 1
        elif not actual_anomaly and detected_anomaly:
            false_positives += 1
        else:  # not actual_anomaly and not detected_anomaly
            true_negatives += 1
    
    # Calculate metrics
    total_samples = len(test_data)
    accuracy = (true_positives + true_negatives) / total_samples
    
    # Handle division by zero
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
        
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0
        
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    # Calculate average response time (measured by action intensity)
    action_intensities = [result['action_intensity'] for result in results]
    avg_action_intensity = sum(action_intensities) / len(action_intensities)
    
    # Return metrics
    metrics = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_action_intensity': avg_action_intensity
    }
    
    return metrics, results

def visualize_test_results(test_data, results):
    """
    Visualize the test results
    
    Args:
        test_data: DataFrame with test data
        results: List of detector results
    """
    # Extract severity and action values from results
    severity_values = [result['anomaly_severity'] for result in results]
    action_values = [result['action_intensity'] for result in results]
    
    # Create figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    
    # Plot 1: Input Parameters
    axs[0].plot(test_data['timestamp'], test_data['voltage_deviation'], 'r-', label='Voltage Deviation (%)')
    axs[0].plot(test_data['timestamp'], test_data['frequency_variation'] * 10, 'g-', label='Frequency Variation (Hz) x10')
    axs[0].plot(test_data['timestamp'], test_data['load_imbalance'], 'b-', label='Load Imbalance (%)')
    axs[0].set_title('Grid Parameters')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Anomaly Presence (Ground Truth)
    axs[1].plot(test_data['timestamp'], test_data['anomaly_present'].astype(int) * 100, 'r-', linewidth=2)
    axs[1].set_title('Actual Anomaly Presence')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Anomaly Present')
    axs[1].set_yticks([0, 100])
    axs[1].set_yticklabels(['False', 'True'])
    axs[1].grid(True)
    
    # Plot 3: Detected Anomaly Severity
    axs[2].plot(test_data['timestamp'], severity_values, 'g-', linewidth=2)
    axs[2].axhline(y=40, color='r', linestyle='--', label='Anomaly Threshold')
    axs[2].set_title('Detected Anomaly Severity')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Severity (%)')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot 4: Corrective Action Intensity
    axs[3].plot(test_data['timestamp'], action_values, 'b-', linewidth=2)
    axs[3].set_title('Corrective Action Intensity')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Action Intensity (%)')
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.show()

def run_optimization_test(anomaly_thresholds=None):
    """
    Run optimization tests with different anomaly thresholds
    to find the best configuration
    
    Args:
        anomaly_thresholds: List of thresholds to test
    
    Returns:
        Best threshold and metrics
    """
    if anomaly_thresholds is None:
        anomaly_thresholds = [20, 30, 40, 50, 60]
    
    detector = FuzzyGridAnomalyDetector()
    test_data = generate_test_dataset()
    
    best_threshold = None
    best_f1 = -1
    best_metrics = None
    
    results = []
    
    print("\n=== Optimization Test Results ===")
    print("Threshold | Accuracy | Precision | Recall | F1 Score")
    print("-" * 60)
    
    for threshold in anomaly_thresholds:
        metrics, _ = evaluate_detector_performance(detector, test_data, threshold)
        
        print(f"{threshold:9} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f}")
        
        results.append({
            'threshold': threshold,
            'metrics': metrics
        })
        
        # Track best configuration
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = threshold
            best_metrics = metrics
    
    print("-" * 60)
    print(f"Best threshold: {best_threshold} (F1 Score: {best_f1:.4f})")
    
    # Plot optimization results
    thresholds = [result['threshold'] for result in results]
    accuracies = [result['metrics']['accuracy'] for result in results]
    precisions = [result['metrics']['precision'] for result in results]
    recalls = [result['metrics']['recall'] for result in results]
    f1_scores = [result['metrics']['f1_score'] for result in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, 'o-', label='Accuracy')
    plt.plot(thresholds, precisions, 'o-', label='Precision')
    plt.plot(thresholds, recalls, 'o-', label='Recall')
    plt.plot(thresholds, f1_scores, 'o-', label='F1 Score')
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best Threshold = {best_threshold}')
    
    plt.title('Optimization Results: Finding the Best Anomaly Threshold')
    plt.xlabel('Anomaly Threshold')
    plt.ylabel('Performance Metric')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.show()
    
    return best_threshold, best_metrics

def run_specific_test_cases():
    """
    Run specific test cases to analyze detector behavior
    in different anomaly scenarios
    """
    detector = FuzzyGridAnomalyDetector()
    
    test_cases = [
        {
            'name': 'Normal Operation',
            'voltage_deviation': 0,
            'frequency_variation': 0,
            'load_imbalance': 5,
            'expected_anomaly': False
        },
        {
            'name': 'Voltage Drop',
            'voltage_deviation': -12,
            'frequency_variation': 0,
            'load_imbalance': 8,
            'expected_anomaly': True
        },
        {
            'name': 'Voltage Spike',
            'voltage_deviation': 13,
            'frequency_variation': 0,
            'load_imbalance': 10,
            'expected_anomaly': True
        },
        {
            'name': 'Frequency Instability',
            'voltage_deviation': 0,
            'frequency_variation': 0.8,
            'load_imbalance': 15,
            'expected_anomaly': True
        },
        {
            'name': 'Load Imbalance',
            'voltage_deviation': 2,
            'frequency_variation': 0.1,
            'load_imbalance': 70,
            'expected_anomaly': True
        },
        {
            'name': 'Combined Anomaly',
            'voltage_deviation': 10,
            'frequency_variation': 0.6,
            'load_imbalance': 60,
            'expected_anomaly': True
        },
        {
            'name': 'Borderline Case',
            'voltage_deviation': 5,
            'frequency_variation': 0.3,
            'load_imbalance': 30,
            'expected_anomaly': 'Borderline'
        }
    ]
    
    print("\n=== Specific Test Cases ===")
    print("Test Case | Severity | Action | Action Type | Correct Detection")
    print("-" * 75)
    
    for case in test_cases:
        result = detector.detect_and_correct(
            case['voltage_deviation'], 
            case['frequency_variation'], 
            case['load_imbalance']
        )
        
        # Determine if detection is correct
        is_anomaly = result['anomaly_severity'] > 40
        
        if case['expected_anomaly'] == 'Borderline':
            correct_detection = "N/A (Borderline)"
        else:
            correct_detection = is_anomaly == case['expected_anomaly']
        
        print(f"{case['name']:<20} | {result['anomaly_severity']:7.2f} | {result['action_intensity']:6.2f} | {result['corrective_actions']['action_type']:<12} | {correct_detection}")
        
        # Print detailed action recommendation for anomalies
        if is_anomaly:
            print(f"  - Recommended action: {result['corrective_actions']['description']}")
            for param, value in result['corrective_actions']['parameters'].items():
                if isinstance(value, (int, float, str, bool)):
                    print(f"    * {param}: {value}")
            print()
    
    print("-" * 75)

def main():
    print("=== Fuzzy Logic-Based Anomaly Detection and Correction System Testing ===")
    print(f"Date: 2025-04-27 06:49:14 UTC")
    print(f"User: SachithPathiranage")
    
    # Create detector instance
    detector = FuzzyGridAnomalyDetector()
    
    # Generate test dataset
    print("\nGenerating test dataset with simulated grid anomalies...")
    test_data = generate_test_dataset()
    
    # Evaluate detector performance
    print("\nEvaluating detector performance...")
    metrics, results = evaluate_detector_performance(detector, test_data)
    
    # Print performance metrics
    print("\n=== Performance Metrics ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Average Action Intensity: {metrics['avg_action_intensity']:.2f}")
    print("\nConfusion Matrix:")
    print(f"True Positives: {metrics['true_positives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    
    # Visualize results
    print("\nVisualizing test results...")
    visualize_test_results(test_data, results)
    
    # Run specific test cases
    run_specific_test_cases()
    
    # Run optimization tests
    print("\nRunning optimization tests...")
    best_threshold, best_metrics = run_optimization_test()
    
    print("\n=== Summary ===")
    print(f"Best anomaly threshold: {best_threshold}")
    print(f"Optimized performance metrics:")
    print(f"- Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"- Precision: {best_metrics['precision']:.4f}")
    print(f"- Recall: {best_metrics['recall']:.4f}")
    print(f"- F1 Score: {best_metrics['f1_score']:.4f}")
    
    print("\nTesting completed successfully.")

if __name__ == "__main__":
    main()