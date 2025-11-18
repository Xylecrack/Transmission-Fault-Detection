import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import signal
import argparse
import sys
import os
import json
import joblib
from pathlib import Path
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBOOST_AVAILABLE = False


# Global flags
MANUAL_TEST_LINE = None  # Will be randomly selected
MANUAL_FAULT_TYPE = 'LG'
running = True
TRAINING_MODE = True
MIN_TRAINING_SAMPLES = 2000
FAULT_TYPES = ['No Fault', 'LG', 'LL', 'LLG', '3P']
FEATURE_WINDOW_SIZE = 10
MODEL_SAVE_PATH = 'saved_models'
MODEL_DIR = Path(MODEL_SAVE_PATH)
SCALER_PATH = MODEL_DIR / 'scaler.joblib'
DETECTOR_PATH = MODEL_DIR / 'detector_rf.joblib'
CLASSIFIER_PATH = MODEL_DIR / 'classifier.joblib'
MODEL_MANIFEST_PATH = MODEL_DIR / 'model_manifest.json'
MIN_SAMPLES_PER_CLASS = 200
MAX_LOOPS = int(os.getenv("MAX_LOOPS", "10"))  # Default safety cap; can be overridden via CLI
SLEEP_INTERVAL = 1.0  # Seconds between iterations; overridden by CLI for faster data capture
DATA_DUMP_DIR = Path('training_cache')
TRAINING_FEATURES_PATH = DATA_DUMP_DIR / 'training_data.csv'
TRAINING_LABELS_PATH = DATA_DUMP_DIR / 'labels.csv'
BATCH_WRITE_SIZE = 200

# Create model save directory if it doesn't exist
if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True)

if not DATA_DUMP_DIR.exists():
    DATA_DUMP_DIR.mkdir(parents=True)

# Debug counters
class DebugStats:
    def __init__(self):
        self.train_counts = {ft: 0 for ft in FAULT_TYPES}
        self.fault_prob_history = []

debug_stats = DebugStats()

# Initialize power system
net = nw.case30()
net.load["p_mw_original"] = net.load.p_mw.copy()
net.load["q_mvar_original"] = net.load.q_mvar.copy()
pp.runpp(net)

FORTESCUE_A = np.exp(2j * np.pi / 3)
FORTESCUE_MATRIX = (1 / 3) * np.array([
    [1, 1, 1],
    [1, FORTESCUE_A ** 2, FORTESCUE_A],
    [1, FORTESCUE_A, FORTESCUE_A ** 2]
])


def complex_phasor(magnitude, angle_deg):
    """Convert polar magnitude/angle to a complex phasor."""
    return magnitude * np.exp(1j * np.deg2rad(angle_deg))


def sequence_components(phase_values):
    """Return Fortescue zero, positive and negative sequence components."""
    seq = FORTESCUE_MATRIX @ phase_values
    return {
        'zero': seq[0],
        'positive': seq[1],
        'negative': seq[2]
    }


def persist_training_batch(samples, labels):
    """Append buffered samples to disk for offline training."""
    if not samples:
        return

    features_exists = TRAINING_FEATURES_PATH.exists()
    labels_exists = TRAINING_LABELS_PATH.exists()
    pd.DataFrame(samples).to_csv(
        TRAINING_FEATURES_PATH,
        mode='a',
        header=not features_exists,
        index=False
    )
    pd.DataFrame(labels, columns=['label']).to_csv(
        TRAINING_LABELS_PATH,
        mode='a',
        header=not labels_exists,
        index=False
    )


class RealTimeMonitor:
    def __init__(self):
        self.history = deque(maxlen=3600)
        self.feature_window = deque(maxlen=FEATURE_WINDOW_SIZE)
        self.previous = {'current': {}, 'voltage': {}}
        self.start_time = datetime.now()

        for line in net.line.index:
            self.previous['current'][line] = 0
            self.previous['voltage'][line] = (0, 0)

    def calculate_current_angle(self, line, direction='from'):
        """Return the current phasor angle (degrees) for the requested end of a line."""
        if direction == 'from':
            bus = net.line.from_bus.at[line]
            p_mw = net.res_line.p_from_mw.at[line]
            q_mvar = net.res_line.q_from_mvar.at[line]
        else:
            bus = net.line.to_bus.at[line]
            p_mw = net.res_line.p_to_mw.at[line]
            q_mvar = net.res_line.q_to_mvar.at[line]

        v_pu = net.res_bus.vm_pu.at[bus]
        va_deg = net.res_bus.va_degree.at[bus]
        S = (p_mw + 1j * q_mvar) * 1e6
        V = v_pu * np.exp(1j * np.deg2rad(va_deg))
        return np.angle(np.conj(S / V), deg=True)

    def update_metrics(self):
        line_metrics = {}
        current_time = datetime.now()

        for line in net.line.index:
            from_bus = net.line.from_bus.at[line]
            to_bus = net.line.to_bus.at[line]

            i_from_mag = net.res_line.i_from_ka.at[line]
            i_to_mag = net.res_line.i_to_ka.at[line]
            i_from_angle = self.calculate_current_angle(line, 'from')
            i_to_angle = self.calculate_current_angle(line, 'to')

            v_from_mag = net.res_bus.vm_pu.at[from_bus]
            v_to_mag = net.res_bus.vm_pu.at[to_bus]
            va_from = net.res_bus.va_degree.at[from_bus]
            va_to = net.res_bus.va_degree.at[to_bus]

            # Build pseudo three-phase representations using both ends and their delta
            i_from_complex = complex_phasor(i_from_mag, i_from_angle)
            i_to_complex = complex_phasor(i_to_mag, i_to_angle)
            i_delta_complex = i_from_complex - i_to_complex
            phase_currents = np.array([i_from_complex, i_to_complex, i_delta_complex])
            seq_currents = sequence_components(phase_currents)

            v_from_complex = complex_phasor(v_from_mag, va_from)
            v_to_complex = complex_phasor(v_to_mag, va_to)
            v_delta_complex = v_from_complex - v_to_complex
            phase_voltages = np.array([v_from_complex, v_to_complex, v_delta_complex])
            seq_voltages = sequence_components(phase_voltages)

            time_diff = 1.0
            di_dt = (i_from_mag - self.previous['current'][line]) / time_diff
            dv_dt_from = (v_from_mag - self.previous['voltage'][line][0]) / time_diff
            dv_dt_to = (v_to_mag - self.previous['voltage'][line][1]) / time_diff

            seq_curr_pos = abs(seq_currents['positive'])
            seq_curr_neg = abs(seq_currents['negative'])
            seq_curr_zero = abs(seq_currents['zero'])
            seq_volt_pos = abs(seq_voltages['positive'])
            seq_volt_neg = abs(seq_voltages['negative'])
            seq_volt_zero = abs(seq_voltages['zero'])

            symmetry_current_ratio = seq_curr_neg / (seq_curr_pos + 1e-6)
            symmetry_voltage_ratio = seq_volt_neg / (seq_volt_pos + 1e-6)

            line_metrics[line] = {
                'current_mag': i_from_mag,
                'current_angle': i_from_angle,
                'voltage_from_mag': v_from_mag,
                'voltage_to_mag': v_to_mag,
                'voltage_angle_diff': abs(va_from - va_to),
                'di_dt': di_dt,
                'dv_dt_from': dv_dt_from,
                'dv_dt_to': dv_dt_to,
                'power_flow_active': net.res_line.p_from_mw.at[line],
                'power_flow_reactive': net.res_line.q_from_mvar.at[line],
                'current_imbalance': abs(i_from_mag - np.mean([net.res_line.i_from_ka.at[l] for l in net.line.index])),
                'ia_mag': abs(phase_currents[0]),
                'ib_mag': abs(phase_currents[1]),
                'ic_mag': abs(phase_currents[2]),
                'va_mag': abs(phase_voltages[0]),
                'vb_mag': abs(phase_voltages[1]),
                'vc_mag': abs(phase_voltages[2]),
                'seq_current_zero': seq_curr_zero,
                'seq_current_positive': seq_curr_pos,
                'seq_current_negative': seq_curr_neg,
                'seq_voltage_zero': seq_volt_zero,
                'seq_voltage_positive': seq_volt_pos,
                'seq_voltage_negative': seq_volt_neg,
                'current_symmetry_ratio': symmetry_current_ratio,
                'voltage_symmetry_ratio': symmetry_voltage_ratio
            }

            self.previous['current'][line] = i_from_mag
            self.previous['voltage'][line] = (v_from_mag, v_to_mag)

        self.feature_window.append(line_metrics)
        self.history.append({
            "timestamp": current_time,
            "lines": line_metrics,
            "system_load": net.load.p_mw.sum()
        })

class FaultSimulator:
    def __init__(self):
        self.last_fault = None
        self.affected_line = None
        self.recovery_data = deque(maxlen=2)
        self.fault_history = []
        self.fault_impedance_range = (0.001, 5.0)  # Expanded impedance range for richer training data
        self.fault_location_range = (0.0, 1.0)     # Range of fault locations along the line (0-100%)
        self.fault_duration_range = (0.05, 4.0)    # Range of fault durations in seconds
        self.fault_start_time = None
        self.fault_end_time = None

    def create_line_fault(self, line_idx, fault_type):
        try:
            if 'fault' in net and not net.fault.empty:
                net.fault = net.fault.iloc[0:0]

            if fault_type == 'No Fault':
                pp.runpp(net)
                self.last_fault = None
                self.affected_line = None
                self.fault_start_time = None
                self.fault_end_time = None
                return

            from_bus = net.line.from_bus.at[line_idx]
            to_bus = net.line.to_bus.at[line_idx]
            
            # Calculate line length and impedance
            line_length = net.line.length_km.at[line_idx]
            r_ohm = net.line.r_ohm_per_km.at[line_idx] * line_length
            x_ohm = net.line.x_ohm_per_km.at[line_idx] * line_length
            
            # Generate random fault parameters
            fault_impedance = np.random.uniform(*self.fault_impedance_range)
            fault_location = np.random.uniform(*self.fault_location_range)
            fault_duration = np.random.uniform(*self.fault_duration_range)
            
            # Calculate fault location impedance
            r_fault = r_ohm * fault_location
            x_fault = x_ohm * fault_location
            
            # Create fault with calculated parameters
            new_fault = pd.DataFrame([{
                'bus': from_bus,
                'fault_impedance': fault_impedance,
                'fault_type': {
                    'LG': 'lg', 'LL': 'll',
                    'LLG': 'llg', '3P': '3ph'
                }[fault_type],
                'r_fault_ohm': r_fault,
                'x_fault_ohm': x_fault
            }])

            net.fault = new_fault
            pp.runpp(net)
            
            # Record fault details
            current_time = datetime.now()
            fault_details = {
                'type': fault_type,
                'line': line_idx,
                'impedance': fault_impedance,
                'location': fault_location,
                'duration': fault_duration,
                'start_time': current_time,
                'end_time': current_time + timedelta(seconds=fault_duration)
            }
            self.fault_history.append(fault_details)
            self.last_fault = (fault_type, line_idx, current_time)
            self.affected_line = line_idx
            self.fault_start_time = current_time
            self.fault_end_time = current_time + timedelta(seconds=fault_duration)
            
            print(f"\nFault Details:")
            print(f"Type: {fault_type}")
            print(f"Line: {line_idx}")
            print(f"Impedance: {fault_impedance:.3f} ohms")
            print(f"Location: {fault_location*100:.1f}% of line length")
            print(f"Duration: {fault_duration:.1f} seconds")

        except Exception as e:
            print(f"Fault error: {str(e)}")
            if 'fault' in net:
                net.fault = net.fault.iloc[0:0]
            self.last_fault = None
            self.affected_line = None
            self.fault_start_time = None
            self.fault_end_time = None

    def should_clear_fault(self):
        if self.fault_end_time is None:
            return False
        return datetime.now() >= self.fault_end_time

class FaultLocalizer:
    def __init__(self):
        self.line_parameters = {}
        self.initialize_line_parameters()
        
    def initialize_line_parameters(self):
        """Initialize line parameters for all lines"""
        for line_idx in net.line.index:
            self.line_parameters[line_idx] = {
                'length': net.line.length_km.at[line_idx],
                'r_ohm_per_km': net.line.r_ohm_per_km.at[line_idx],
                'x_ohm_per_km': net.line.x_ohm_per_km.at[line_idx],
                'c_nf_per_km': net.line.c_nf_per_km.at[line_idx],
                'g_us_per_km': net.line.g_us_per_km.at[line_idx]
            }
    
    def impedance_based_localization(self, line_idx, measurements):
        """Impedance-based fault location method"""
        params = self.line_parameters[line_idx]
        
        # Extract measurements
        v_from = measurements['voltage_from_mag']
        v_to = measurements['voltage_to_mag']
        i_from = measurements['current_mag']
        i_angle = measurements['current_angle']
        
        # Calculate apparent impedance
        Z_apparent = (v_from - v_to) / i_from
        
        # Calculate line impedance per km
        Z_line = np.sqrt(params['r_ohm_per_km']**2 + params['x_ohm_per_km']**2)
        
        # Calculate fault distance
        fault_distance = abs(Z_apparent) / Z_line
        fault_distance = min(fault_distance, params['length'])
        
        return fault_distance
    
    def traveling_wave_localization(self, line_idx, measurements):
        """Traveling wave-based fault location method"""
        params = self.line_parameters[line_idx]
        
        # Extract rate of change measurements
        di_dt = measurements['di_dt']
        dv_dt = measurements['dv_dt_from']
        
        # Calculate wave velocity (approximately 3e8 m/s for overhead lines)
        wave_velocity = 3e8
        
        # Calculate time difference using rate of change
        time_diff = abs(di_dt / dv_dt) if dv_dt != 0 else 0
        
        # Calculate fault distance
        fault_distance = (wave_velocity * time_diff) / 2
        fault_distance = min(fault_distance, params['length'])
        
        return fault_distance
    
    def hybrid_localization(self, line_idx, measurements):
        """Combine both methods for more accurate localization"""
        # Get results from both methods
        imp_distance = self.impedance_based_localization(line_idx, measurements)
        tw_distance = self.traveling_wave_localization(line_idx, measurements)
        
        # Weight the results (can be tuned based on performance)
        imp_weight = 0.7
        tw_weight = 0.3
        
        # Calculate weighted average
        fault_distance = (imp_weight * imp_distance + tw_weight * tw_distance)
        
        # Calculate confidence based on agreement between methods
        confidence = 1 - abs(imp_distance - tw_distance) / self.line_parameters[line_idx]['length']
        
        return {
            'distance': fault_distance,
            'confidence': confidence,
            'impedance_method': imp_distance,
            'traveling_wave_method': tw_distance
        }

class KNNFaultDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.detector_model = self._create_detector_model()
        self.classifier_key = 'random_forest'
        self.classifier_model = self._create_classifier(self.classifier_key)
        self.training_data = []
        self.labels = []
        self.is_trained = False
        self.detection_buffer = deque(maxlen=5)
        self.last_detection = None
        self.min_samples_per_class = MIN_SAMPLES_PER_CLASS
        self.localizer = FaultLocalizer()
        self.load_models()  # Load persisted models, if any

    def _create_detector_model(self):
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            class_weight='balanced_subsample',
            random_state=42
        )

    def _available_classifier_keys(self):
        keys = ['random_forest', 'svm_rbf', 'mlp']
        if XGBOOST_AVAILABLE:
            keys.append('xgboost')
        return keys

    def _create_classifier(self, key):
        if key == 'random_forest':
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                n_jobs=-1,
                class_weight='balanced_subsample',
                random_state=21
            )
        if key == 'svm_rbf':
            return SVC(
                kernel='rbf',
                C=25,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=21
            )
        if key == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=800,
                random_state=21
            )
        if key == 'xgboost' and XGBOOST_AVAILABLE:
            return XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=400,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                tree_method='hist',
                eval_metric='mlogloss'
            )
        raise ValueError(f"Unsupported classifier key: {key}")

    def load_models(self):
        try:
            if SCALER_PATH.exists():
                self.scaler = joblib.load(SCALER_PATH)
            if DETECTOR_PATH.exists():
                self.detector_model = joblib.load(DETECTOR_PATH)
            if CLASSIFIER_PATH.exists():
                self.classifier_model = joblib.load(CLASSIFIER_PATH)
            if MODEL_MANIFEST_PATH.exists():
                with open(MODEL_MANIFEST_PATH, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                    key = manifest.get('classifier_key', self.classifier_key)
                    if key in self._available_classifier_keys():
                        self.classifier_key = key
            if all(path.exists() for path in [SCALER_PATH, DETECTOR_PATH, CLASSIFIER_PATH]):
                self.is_trained = True
                print(f"Loaded existing models successfully (classifier={self.classifier_key})")
        except Exception as e:
            print(f"Could not load existing models: {str(e)}")

    def save_models(self):
        try:
            joblib.dump(self.scaler, SCALER_PATH)
            joblib.dump(self.detector_model, DETECTOR_PATH)
            joblib.dump(self.classifier_model, CLASSIFIER_PATH)
            with open(MODEL_MANIFEST_PATH, 'w', encoding='utf-8') as f:
                json.dump({'classifier_key': self.classifier_key}, f, indent=2)
            print("Models saved successfully")
        except Exception as e:
            print(f"Could not save models: {str(e)}")

    def extract_features(self, window):
        features = []
        for snapshot in window:
            for line in snapshot:
                values = snapshot[line]
                # Enhanced feature set
                features.extend([
                    float(values.get('current_mag', 0.0)),
                    float(values.get('voltage_from_mag', 0.0)),
                    float(values.get('voltage_to_mag', 0.0)),
                    float(values.get('voltage_angle_diff', 0.0)),
                    float(values.get('di_dt', 0.0)),
                    float(values.get('current_imbalance', 0.0)),
                    float(values.get('power_flow_active', 0.0)),
                    float(values.get('power_flow_reactive', 0.0)),
                    float(values.get('dv_dt_from', 0.0)),
                    float(values.get('dv_dt_to', 0.0)),
                    # Multi-phase magnitudes
                    float(values.get('ia_mag', 0.0)),
                    float(values.get('ib_mag', 0.0)),
                    float(values.get('ic_mag', 0.0)),
                    float(values.get('va_mag', 0.0)),
                    float(values.get('vb_mag', 0.0)),
                    float(values.get('vc_mag', 0.0)),
                    # Sequence components
                    float(values.get('seq_current_zero', 0.0)),
                    float(values.get('seq_current_positive', 0.0)),
                    float(values.get('seq_current_negative', 0.0)),
                    float(values.get('seq_voltage_zero', 0.0)),
                    float(values.get('seq_voltage_positive', 0.0)),
                    float(values.get('seq_voltage_negative', 0.0)),
                    # Ratios and derived indicators
                    float(values.get('current_symmetry_ratio', 0.0)),
                    float(values.get('voltage_symmetry_ratio', 0.0)),
                    float(values.get('current_mag', 0.0) * values.get('voltage_from_mag', 0.0)),
                    float(abs(values.get('di_dt', 0.0)) * values.get('voltage_angle_diff', 0.0)),
                    float(values.get('current_imbalance', 0.0) * values.get('power_flow_active', 0.0))
                ])
        return np.array(features, dtype=np.float64)

    def train(self):
        if len(self.training_data) < MIN_TRAINING_SAMPLES:
            print(f"Not enough training samples ({len(self.training_data)}/{MIN_TRAINING_SAMPLES})")
            return

        X = np.array(self.training_data)
        y = np.array(self.labels)

        # Print class distribution
        print("\nTraining Data Distribution:")
        for ft in FAULT_TYPES:
            count = np.sum(y == ft)
            print(f"{ft}: {count} samples")

        # Ensure minimum samples per class
        class_counts = {ft: np.sum(y == ft) for ft in FAULT_TYPES}
        if any(count < self.min_samples_per_class for count in class_counts.values()):
            print("Insufficient samples for some fault types")
            return

        X_scaled = self.scaler.fit_transform(X)

        # Train RandomForest for fault detection
        y_binary = (y != 'No Fault').astype(int)
        self.detector_model = self._create_detector_model()
        self.detector_model.fit(X_scaled, y_binary)

        # Train and select best classifier for fault typing
        fault_mask = y_binary == 1
        if sum(fault_mask) > self.min_samples_per_class:
            X_fault = X_scaled[fault_mask]
            y_fault = y[fault_mask]
            self.classifier_key, self.classifier_model, cls_score = self._select_best_classifier(X_fault, y_fault)
            self.is_trained = True

            det_score = self.detector_model.score(X_scaled, y_binary)
            print("\nTraining Results:")
            print(f"Detection Accuracy (RF): {det_score:.2%}")
            print(f"Classification Accuracy ({self.classifier_key}): {cls_score:.2%}")

            self.save_models()
        else:
            print("Insufficient fault samples for classifier training")

    def _select_best_classifier(self, X_fault, y_fault):
        best_key = None
        best_model = None
        best_score = -np.inf
        for key in self._available_classifier_keys():
            try:
                model = self._create_classifier(key)
            except ValueError:
                continue
            model.fit(X_fault, y_fault)
            score = model.score(X_fault, y_fault)
            print(f"Model {key} training accuracy: {score:.2%}")
            if score > best_score:
                best_key = key
                best_model = model
                best_score = score
        if best_model is None:
            raise RuntimeError("No classifier could be trained; check dataset")
        return best_key, best_model, best_score

    def predict(self, features, latest_measurements=None, line_idx=None):
        if not self.is_trained:
            return {'fault': False}

        X_scaled = self.scaler.transform([features])

        det_probs = self.detector_model.predict_proba(X_scaled)[0]
        fault_probability = det_probs[1] if len(det_probs) > 1 else det_probs[0]
        is_fault_prediction = int(fault_probability > 0.5)

        if fault_probability > 0.6:
            self.detection_buffer.append(is_fault_prediction)
        else:
            self.detection_buffer.append(0)

        # Clear detection buffer if stale
        if self.last_detection and (datetime.now() - self.last_detection).total_seconds() > 5:
            self.detection_buffer.clear()
            self.last_detection = None

        if sum(self.detection_buffer) >= 3 and not self.last_detection:
            class_probs = self.classifier_model.predict_proba(X_scaled)[0]
            fault_type = self.classifier_model.predict(X_scaled)[0]

            localization_result = None
            if latest_measurements is not None and line_idx is not None:
                localization_result = self.localizer.hybrid_localization(line_idx, latest_measurements)

            combined_confidence = (0.6 * fault_probability) + (0.4 * np.max(class_probs))

            self.last_detection = datetime.now()
            result = {
                'fault': True,
                'type': fault_type,
                'confidence': combined_confidence,
                'detector_confidence': fault_probability,
                'classifier_confidence': np.max(class_probs)
            }

            if localization_result is not None:
                result['fault_location'] = {
                    'distance_km': localization_result['distance'],
                    'percentage': (localization_result['distance'] /
                                   self.localizer.line_parameters[line_idx]['length']) * 100,
                    'confidence': localization_result['confidence'],
                    'impedance_method': localization_result['impedance_method'],
                    'traveling_wave_method': localization_result['traveling_wave_method']
                }

            return result

        return {'fault': False}

def live_analysis_loop(max_loops=MAX_LOOPS, sleep_interval=SLEEP_INTERVAL):
    global running, TRAINING_MODE, MANUAL_TEST_LINE
    monitor = RealTimeMonitor()
    fault_simulator = FaultSimulator()
    detector = KNNFaultDetector()

    manual_fault_triggered = False
    manual_fault_active = False
    training_complete_time = None
    detection_stats = {
        'total_detections': 0,
        'correct_detections': 0,
        'false_positives': 0,
        'missed_faults': 0
    }
    buffered_samples = []
    buffered_labels = []

    loop_counter = 0
    while running and loop_counter < max_loops:
        start_time = time.time()

        # --- Training mode fault injection / clearing logic ---
        if TRAINING_MODE:
            # Auto-clear fault when scheduled end time reached to allow no-fault windows
            if fault_simulator.should_clear_fault():
                fault_simulator.create_line_fault(
                    fault_simulator.affected_line if fault_simulator.affected_line is not None else np.random.choice(net.line.index),
                    'No Fault'
                )
            # Inject a new random fault with given probability
            elif np.random.rand() < 0.22:  # slightly >0.2 for variety
                fault_type = np.random.choice(FAULT_TYPES[1:])
                line_idx = np.random.choice(net.line.index)
                fault_simulator.create_line_fault(line_idx, fault_type)
            # Intentionally force a clean no-fault snapshot segment sometimes to boost negative class
            elif np.random.rand() < 0.10 and (fault_simulator.last_fault and fault_simulator.last_fault[0] != 'No Fault'):
                fault_simulator.create_line_fault(
                    fault_simulator.affected_line if fault_simulator.affected_line is not None else np.random.choice(net.line.index),
                    'No Fault'
                )
        else:
            if not TRAINING_MODE and not manual_fault_triggered:
                if training_complete_time is None:
                    training_complete_time = datetime.now()
                elif (datetime.now() - training_complete_time).total_seconds() > 5:
                    # Randomly select a line for testing
                    MANUAL_TEST_LINE = np.random.choice(net.line.index)
                    print(f"\n=== INJECTING MANUAL {MANUAL_FAULT_TYPE} FAULT ON LINE {MANUAL_TEST_LINE} ===")
                    fault_simulator.create_line_fault(MANUAL_TEST_LINE, MANUAL_FAULT_TYPE)
                    manual_fault_triggered = True
                    manual_fault_active = True

            # Check if current fault should be cleared
            if manual_fault_active and fault_simulator.should_clear_fault():
                print("\n=== CLEARING MANUAL FAULT ===")
                fault_simulator.create_line_fault(MANUAL_TEST_LINE, 'No Fault')
                manual_fault_active = False
                detector.last_detection = None
                detector.detection_buffer.clear()  # Clear detection buffer
                manual_fault_triggered = False

        elapsed = (datetime.now() - monitor.start_time).total_seconds()
        load_factor = 0.9 + 0.1 * np.sin(elapsed / 3600 * np.pi)
        net.load.p_mw = net.load.p_mw_original * load_factor
        net.load.q_mvar = net.load.q_mvar_original * load_factor

        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            print("Warning: Load flow did not converge")
            continue

        monitor.update_metrics()

        if TRAINING_MODE and len(monitor.feature_window) == FEATURE_WINDOW_SIZE:
            features = detector.extract_features(monitor.feature_window)
            label = fault_simulator.last_fault[0] if fault_simulator.last_fault else 'No Fault'

            detector.training_data.append(features)
            detector.labels.append(label)
            debug_stats.train_counts[label] += 1
            buffered_samples.append(features)
            buffered_labels.append(label)
            print(f"Training sample added: {label}")

            if len(buffered_samples) >= BATCH_WRITE_SIZE:
                persist_training_batch(buffered_samples, buffered_labels)
                buffered_samples.clear()
                buffered_labels.clear()

            total_samples = sum(debug_stats.train_counts.values())
            min_class_samples = min(debug_stats.train_counts.values())
            if total_samples >= MIN_TRAINING_SAMPLES and min_class_samples >= MIN_SAMPLES_PER_CLASS:
                persist_training_batch(buffered_samples, buffered_labels)
                buffered_samples.clear()
                buffered_labels.clear()
                TRAINING_MODE = False
                print("\n=== Training Capture Complete ===")
                print(f"Samples logged to {TRAINING_FEATURES_PATH} and {TRAINING_LABELS_PATH}")
                print("Run train_offline.py (offline) to benchmark RF/XGBoost/SVM/MLP and save the best model.")

        if not TRAINING_MODE and len(monitor.feature_window) == FEATURE_WINDOW_SIZE:
            features = detector.extract_features(monitor.feature_window)
            latest_measurements = None
            if MANUAL_TEST_LINE is not None and len(monitor.feature_window) > 0:
                latest_measurements = monitor.feature_window[-1].get(MANUAL_TEST_LINE, None)
            prediction = detector.predict(features, latest_measurements, MANUAL_TEST_LINE)

            current_time = datetime.now().strftime('%H:%M:%S')
            system_load = monitor.history[-1]['system_load']
            
            print(f"\n=== Status Update [{current_time}] ===")
            print(f"System Load: {system_load:.1f} MW")
            
            if prediction['fault']:
                detection_stats['total_detections'] += 1
                actual_fault = fault_simulator.last_fault[0] if fault_simulator.last_fault else 'No Fault'
                
                if actual_fault != 'No Fault':
                    if prediction['type'] == actual_fault:
                        detection_stats['correct_detections'] += 1
                        print(f"✓ Correctly detected {actual_fault} fault")
                    else:
                        detection_stats['missed_faults'] += 1
                        print(f"✗ Missed {actual_fault} fault (detected as {prediction['type']})")
                else:
                    detection_stats['false_positives'] += 1
                    print(f"⚠ False positive: detected {prediction['type']} fault")
                
                print(f"Detector Confidence: {prediction['detector_confidence']:.1%}")
                print(f"Classifier Confidence: {prediction['classifier_confidence']:.1%}")
                print(f"Combined Confidence: {prediction['confidence']:.1%}")
                
                loc = prediction.get('fault_location')
                if loc:
                    print(f"\nFault Location:")
                    print(f"Distance: {loc['distance_km']:.2f} km ({loc['percentage']:.1f}% of line length)")
                    print(f"Localization Confidence: {loc['confidence']:.1%}")
                    print(f"Impedance Method: {loc['impedance_method']:.2f} km")
                    print(f"Traveling Wave Method: {loc['traveling_wave_method']:.2f} km")
            else:
                if fault_simulator.last_fault and fault_simulator.last_fault[0] != 'No Fault':
                    detection_stats['missed_faults'] += 1
                    print(f"✗ Missed {fault_simulator.last_fault[0]} fault")
                else:
                    print("Status: Normal")

            # Print detection statistics periodically
            if detection_stats['total_detections'] > 0 and detection_stats['total_detections'] % 10 == 0:
                print("\n=== Detection Statistics ===")
                print(f"Total Detections: {detection_stats['total_detections']}")
                print(f"Correct Detections: {detection_stats['correct_detections']}")
                print(f"False Positives: {detection_stats['false_positives']}")
                print(f"Missed Faults: {detection_stats['missed_faults']}")
                accuracy = detection_stats['correct_detections'] / detection_stats['total_detections']
                print(f"Detection Accuracy: {accuracy:.1%}")

        elapsed = time.time() - start_time
        # Adjustable pacing for faster dataset generation
        if elapsed < sleep_interval:
            time.sleep(sleep_interval - elapsed)
        loop_counter += 1

    # Flush any remaining buffered samples before exiting
    if buffered_samples:
        persist_training_batch(buffered_samples, buffered_labels)

# Main
if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
    parser = argparse.ArgumentParser(description="Real-time power system simulation & fault data collector.")
    parser.add_argument('--max-loops', type=int, default=MAX_LOOPS, help='Total iterations to run (default env MAX_LOOPS or 10).')
    parser.add_argument('--sleep', type=float, default=SLEEP_INTERVAL, help='Sleep interval per loop in seconds (default 1.0). Use smaller values to accelerate.')
    parser.add_argument('--training-mode', action='store_true', help='Force training mode (data capture).')
    parser.add_argument('--inference-mode', action='store_true', help='Force inference mode (detection/classification).')
    args = parser.parse_args()

    if args.training_mode and args.inference_mode:
        print('Specify only one of --training-mode or --inference-mode.')
        sys.exit(1)
    if args.training_mode:
        TRAINING_MODE = True
    if args.inference_mode:
        TRAINING_MODE = False

    print(f"Starting simulation (training_mode={TRAINING_MODE}, max_loops={args.max_loops}, sleep={args.sleep}s)")
    live_analysis_loop(max_loops=args.max_loops, sleep_interval=args.sleep)
