import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from collections import defaultdict

class FuzzyGridAnomalyDetector:
    """
    A fuzzy logic-based system for detecting and correcting anomalies in smart grid systems.
    Monitors voltage deviation, frequency variation, and line load imbalance to determine
    appropriate corrective actions.
    """
    
    def __init__(self):
        # Define universe of discourse ranges for each parameter
        self.voltage_range = np.linspace(-15, 15, 100)  # % deviation from nominal
        self.frequency_range = np.linspace(-1, 1, 100)  # Hz deviation from nominal (e.g., 50/60 Hz)
        self.load_imbalance_range = np.linspace(0, 100, 100)  # % imbalance
        
        # Severity output ranges
        self.severity_range = np.linspace(0, 100, 100)  # % severity
        self.action_range = np.linspace(0, 100, 100)  # % action intensity
        
        # Initialize membership functions using skfuzzy
        self.initialize_membership_functions()
        
        # Initialize fuzzy rules
        self.initialize_fuzzy_rules()
    
    def initialize_membership_functions(self):
        """Define membership functions for input and output variables using skfuzzy"""
        # Voltage deviation membership functions
        voltage_low = fuzz.trapmf(self.voltage_range, [-15, -15, -5, 0])
        voltage_normal = fuzz.trimf(self.voltage_range, [-5, 0, 5])
        voltage_high = fuzz.trapmf(self.voltage_range, [0, 5, 15, 15])
        
        self.voltage_mf = {
            'low': voltage_low,
            'normal': voltage_normal,
            'high': voltage_high
        }
        
        # Frequency variation membership functions
        freq_stable = fuzz.trapmf(self.frequency_range, [-0.2, -0.1, 0.1, 0.2])
        freq_unstable_pos = fuzz.trapmf(self.frequency_range, [0.1, 0.3, 1, 1])
        freq_unstable_neg = fuzz.trapmf(self.frequency_range, [-1, -1, -0.3, -0.1])
        
        self.frequency_mf = {
            'stable': freq_stable,
            'unstable_pos': freq_unstable_pos,
            'unstable_neg': freq_unstable_neg
        }
        
        # Load imbalance membership functions - {Balanced, Unbalanced}
        load_balanced = fuzz.trapmf(self.load_imbalance_range, [0, 0, 20, 35])
        load_unbalanced = fuzz.trapmf(self.load_imbalance_range, [25, 40, 100, 100])
        
        self.load_imbalance_mf = {
            'balanced': load_balanced,
            'unbalanced': load_unbalanced
        }
        
        # Anomaly severity membership functions (output)
        severity_low = fuzz.trimf(self.severity_range, [0, 0, 40])
        severity_medium = fuzz.trimf(self.severity_range, [20, 50, 80])
        severity_high = fuzz.trimf(self.severity_range, [60, 100, 100])
        
        self.severity_mf = {
            'low': severity_low,
            'medium': severity_medium,
            'high': severity_high
        }
        
        # Corrective action membership functions (output)
        action_monitor = fuzz.trimf(self.action_range, [0, 0, 30])
        action_adjust = fuzz.trimf(self.action_range, [20, 50, 80])
        action_isolate = fuzz.trimf(self.action_range, [70, 100, 100])
        
        self.action_mf = {
            'monitor': action_monitor,
            'adjust': action_adjust,
            'isolate': action_isolate
        }
    
    def initialize_fuzzy_rules(self):
        """Define fuzzy rules for anomaly detection and correction"""
        self.rules = [
            # Format: (voltage, frequency, load_imbalance, severity, action)
            # Each rule contains the linguistic terms for each input and the resulting outputs
            
            # Normal operating conditions
            ('medium', 'stable', 'balanced', 'low', 'monitor'),
            
            # Minor anomalies
            ('low', 'stable', 'balanced', 'low', 'monitor'),
            ('high', 'stable', 'balanced', 'low', 'monitor'),
            ('medium', 'stable', 'unbalanced', 'medium', 'adjust'),
            
            # Moderate anomalies
            ('low', 'unstable_pos', 'balanced', 'medium', 'adjust'),
            ('low', 'unstable_neg', 'balanced', 'medium', 'adjust'),
            ('high', 'unstable_pos', 'balanced', 'medium', 'adjust'),
            ('high', 'unstable_neg', 'balanced', 'medium', 'adjust'),
            ('medium', 'unstable_pos', 'balanced', 'medium', 'adjust'),
            ('medium', 'unstable_neg', 'balanced', 'medium', 'adjust'),
            ('low', 'stable', 'unbalanced', 'medium', 'adjust'),
            ('high', 'stable', 'unbalanced', 'medium', 'adjust'),
            
            # Severe anomalies
            ('low', 'unstable_pos', 'unbalanced', 'high', 'isolate'),
            ('low', 'unstable_neg', 'unbalanced', 'high', 'isolate'),
            ('high', 'unstable_pos', 'unbalanced', 'high', 'isolate'),
            ('high', 'unstable_neg', 'unbalanced', 'high', 'isolate'),
            ('medium', 'unstable_pos', 'unbalanced', 'high', 'isolate'),
            ('medium', 'unstable_neg', 'unbalanced', 'high', 'isolate')
        ]
    
    def fuzzify(self, voltage_deviation, frequency_variation, load_imbalance):
        """
        Convert crisp input values to fuzzy membership degrees
        
        Args:
            voltage_deviation: Percentage deviation from nominal voltage
            frequency_variation: Frequency deviation in Hz from nominal
            load_imbalance: Percentage of load imbalance
            
        Returns:
            Dictionary of fuzzy membership degrees for each input variable
        """
        fuzzy_values = {
            'voltage': {},
            'frequency': {},
            'load_imbalance': {}
        }
        
        # Find the index in each range that corresponds to the input values
        v_idx = np.abs(self.voltage_range - voltage_deviation).argmin()
        f_idx = np.abs(self.frequency_range - frequency_variation).argmin()
        l_idx = np.abs(self.load_imbalance_range - load_imbalance).argmin()
        
        # Calculate membership degrees for voltage
        for label, mf in self.voltage_mf.items():
            fuzzy_values['voltage'][label] = mf[v_idx]
        
        # Calculate membership degrees for frequency
        for label, mf in self.frequency_mf.items():
            fuzzy_values['frequency'][label] = mf[f_idx]
        
        # Calculate membership degrees for load imbalance
        for label, mf in self.load_imbalance_mf.items():
            fuzzy_values['load_imbalance'][label] = mf[l_idx]
        
        return fuzzy_values
    
    def apply_fuzzy_rules(self, fuzzy_values):
        """
        Apply fuzzy rules to determine anomaly severity and corrective action
        
        Args:
            fuzzy_values: Dictionary of fuzzy membership degrees for each input variable
            
        Returns:
            Dictionary of activated rule consequences and their activation levels
        """
        rule_activations = {
            'severity': defaultdict(float),
            'action': defaultdict(float)
        }
        
        # Process each rule
        for rule in self.rules:
            voltage_term, freq_term, load_term, severity_term, action_term = rule
            
            # Calculate rule activation using minimum (AND operation)
            voltage_activation = fuzzy_values['voltage'].get(voltage_term, 0)
            freq_activation = fuzzy_values['frequency'].get(freq_term, 0)
            load_activation = fuzzy_values['load_imbalance'].get(load_term, 0)
            
            # Combine using AND (minimum)
            rule_activation = min(voltage_activation, freq_activation, load_activation)
            
            # Apply activation to consequents (using maximum for multiple rule activations)
            rule_activations['severity'][severity_term] = max(
                rule_activations['severity'][severity_term], rule_activation
            )
            rule_activations['action'][action_term] = max(
                rule_activations['action'][action_term], rule_activation
            )
        
        return rule_activations
    
    def defuzzify_centroid(self, activated_consequences, output_type):
        """
        Convert fuzzy output values to crisp values using centroid defuzzification
        
        Args:
            activated_consequences: Dictionary of activated rule consequences
            output_type: 'severity' or 'action'
            
        Returns:
            Crisp output value
        """
        if output_type == 'severity':
            output_range = self.severity_range
            output_mfs = self.severity_mf
        else:  # action
            output_range = self.action_range
            output_mfs = self.action_mf
        
        # Calculate the aggregated membership function
        aggregated_mf = np.zeros_like(output_range, dtype=float)
        
        for term, activation in activated_consequences.items():
            if activation > 0:
                # Clip the membership function at the activation level
                clipped_mf = np.minimum(output_mfs[term], activation)
                # Combine using maximum (OR operation)
                aggregated_mf = np.maximum(aggregated_mf, clipped_mf)
        
        # Check if the aggregated membership function is all zeros
        if not np.any(aggregated_mf):
            return 0
        
        # Calculate centroid
        numerator = np.sum(output_range * aggregated_mf)
        denominator = np.sum(aggregated_mf)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def determine_corrective_action(self, severity_value, action_value):
        """
        Determine specific corrective actions based on defuzzified severity and action values
        
        Args:
            severity_value: Defuzzified anomaly severity (0-100)
            action_value: Defuzzified action intensity (0-100)
            
        Returns:
            Dictionary of recommended actions with specific parameters
        """
        actions = {}
        
        # Monitor action (low severity/action)
        if action_value < 30:
            actions['action_type'] = 'monitor'
            actions['description'] = 'Continue monitoring with increased frequency'
            actions['parameters'] = {
                'monitoring_interval': max(1, int(30 - severity_value/3)),  # seconds
                'alert_threshold': severity_value + 10
            }
        
        # Adjust action (medium severity/action)
        elif 20 <= action_value < 80:
            actions['action_type'] = 'adjust'
            
            # Determine which parameters need adjustment based on severity
            if severity_value < 50:
                # Less severe adjustments
                if self.last_values.get('voltage_deviation', 0) > 3:
                    actions['description'] = 'Minor voltage reduction needed'
                    actions['parameters'] = {
                        'voltage_adjustment': -min(2, self.last_values.get('voltage_deviation', 0)/2),
                        'power_factor_correction': True,
                        'capacitor_banks': 'partial'
                    }
                elif self.last_values.get('voltage_deviation', 0) < -3:
                    actions['description'] = 'Minor voltage increase needed'
                    actions['parameters'] = {
                        'voltage_adjustment': min(2, abs(self.last_values.get('voltage_deviation', 0))/2),
                        'power_factor_correction': True,
                        'capacitor_banks': 'partial'
                    }
                elif abs(self.last_values.get('frequency_variation', 0)) > 0.2:
                    actions['description'] = 'Frequency stabilization needed'
                    actions['parameters'] = {
                        'frequency_adjustment': -self.last_values.get('frequency_variation', 0)/2,
                        'energy_storage': 'engage_partial',
                        'adjustment_rate': 'gradual'
                    }
                else:
                    actions['description'] = 'Load balancing needed'
                    actions['parameters'] = {
                        'load_redistribution': True,
                        'redistribution_percentage': min(30, self.last_values.get('load_imbalance', 0)/2)
                    }
            else:
                # More severe adjustments
                actions['description'] = 'Major system adjustment needed'
                actions['parameters'] = {
                    'voltage_adjustment': -self.last_values.get('voltage_deviation', 0),
                    'frequency_adjustment': -self.last_values.get('frequency_variation', 0),
                    'load_redistribution': True,
                    'redistribution_percentage': min(50, self.last_values.get('load_imbalance', 0)),
                    'power_factor_correction': True,
                    'capacitor_banks': 'full',
                    'energy_storage': 'engage_full'
                }
        
        # Isolate action (high severity/action)
        else:  # action_value >= 70
            actions['action_type'] = 'isolate'
            actions['description'] = 'Critical anomaly detected - isolation required'
            
            if self.last_values.get('voltage_deviation', 0) > 10 or self.last_values.get('voltage_deviation', 0) < -10:
                actions['parameters'] = {
                    'isolate_sections': ['high_voltage_deviation_sections'],
                    'notify_operators': True,
                    'emergency_protocol': 'voltage_emergency',
                    'backup_systems': 'activate'
                }
            elif abs(self.last_values.get('frequency_variation', 0)) > 0.5:
                actions['parameters'] = {
                    'isolate_sections': ['frequency_unstable_sections'],
                    'notify_operators': True,
                    'emergency_protocol': 'frequency_emergency',
                    'backup_systems': 'activate'
                }
            else:
                actions['parameters'] = {
                    'isolate_sections': ['overloaded_sections'],
                    'notify_operators': True,
                    'emergency_protocol': 'load_emergency',
                    'backup_systems': 'activate'
                }
        
        return actions
    
    def detect_and_correct(self, voltage_deviation, frequency_variation, load_imbalance):
        """
        Main method to detect anomalies and suggest corrective actions
        
        Args:
            voltage_deviation: Percentage deviation from nominal voltage
            frequency_variation: Frequency deviation in Hz from nominal
            load_imbalance: Percentage of load imbalance
            
        Returns:
            Dictionary with anomaly detection results and recommended actions
        """
        # Store values for use in determining corrective actions
        self.last_values = {
            'voltage_deviation': voltage_deviation,
            'frequency_variation': frequency_variation,
            'load_imbalance': load_imbalance
        }
        
        # Fuzzification
        fuzzy_values = self.fuzzify(voltage_deviation, frequency_variation, load_imbalance)
        
        # Rule evaluation
        rule_activations = self.apply_fuzzy_rules(fuzzy_values)
        
        # Defuzzification
        severity_value = self.defuzzify_centroid(rule_activations['severity'], 'severity')
        action_value = self.defuzzify_centroid(rule_activations['action'], 'action')
        
        # Determine specific corrective actions
        corrective_actions = self.determine_corrective_action(severity_value, action_value)
        
        # Prepare result
        result = {
            'input_values': {
                'voltage_deviation': voltage_deviation,
                'frequency_variation': frequency_variation,
                'load_imbalance': load_imbalance
            },
            'fuzzy_values': fuzzy_values,
            'rule_activations': dict(rule_activations),
            'anomaly_severity': severity_value,
            'action_intensity': action_value,
            'corrective_actions': corrective_actions
        }
        
        return result
    
    def plot_membership_functions_internal(voltage_range, frequency_range, load_imbalance_range, 
                                voltage_mf, frequency_mf, load_imbalance_mf):
        """
        Plot membership functions for visualization
        
        Args:
            voltage_range: Array of voltage deviation values
            frequency_range: Array of frequency variation values
            load_imbalance_range: Array of load imbalance values
            voltage_mf: Dictionary of voltage membership functions
            frequency_mf: Dictionary of frequency membership functions
            load_imbalance_mf: Dictionary of load imbalance membership functions
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot voltage membership functions
        axs[0].set_title('Voltage Deviation Membership Functions')
        for label, mf in voltage_mf.items():
            axs[0].plot(voltage_range, mf, linewidth=1.5, label=label)
        axs[0].set_xlabel('Voltage Deviation (%)')
        axs[0].set_ylabel('Membership Degree')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_ylim(-0.1, 1.1)
        
        # Plot frequency membership functions
        axs[1].set_title('Frequency Variation Membership Functions')
        for label, mf in frequency_mf.items():
            axs[1].plot(frequency_range, mf, linewidth=1.5, label=label)
        axs[1].set_xlabel('Frequency Variation (Hz)')
        axs[1].set_ylabel('Membership Degree')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].set_ylim(-0.1, 1.1)
        
        # Plot load imbalance membership functions
        axs[2].set_title('Load Imbalance Membership Functions')
        for label, mf in load_imbalance_mf.items():
            axs[2].plot(load_imbalance_range, mf, linewidth=1.5, label=label)
        axs[2].set_xlabel('Load Imbalance (%)')
        axs[2].set_ylabel('Membership Degree')
        axs[2].legend()
        axs[2].grid(True)
        axs[2].set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        plt.show()

    def plot_membership_functions(self):
        """Plot the membership functions for all input and output variables"""
        FuzzyGridAnomalyDetector.plot_membership_functions_internal(
            self.voltage_range,
            self.frequency_range,
            self.load_imbalance_range,
            self.voltage_mf,
            self.frequency_mf,
            self.load_imbalance_mf
        )