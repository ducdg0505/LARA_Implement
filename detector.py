## detector.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import genpareto
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, Union, Tuple

# Import LARA for type hinting
from lara_wrapper import LARA

class AnomalyDetector:
    """
    Anomaly Detector for LARA.
    
    Responsible for:
    1. Computing anomaly scores (reconstruction errors) using the LARA model.
    2. Determining thresholds using the Peak-Over-Threshold (POT) algorithm.
    3. Applying point-adjustment heuristics for time-series evaluation.
    4. Calculating precision, recall, and F1-score.
    """

    def __init__(self, lara: LARA, config: Dict):
        """
        Initializes the AnomalyDetector with the LARA model and configuration.

        Args:
            lara: An instance of the LARA wrapped model.
            config: Configuration dictionary containing evaluation parameters.
        """
        self.lara: LARA = lara
        self.config: Dict = config
        
        # Extract parameters from configuration
        eval_cfg = config.get('evaluation', {})
        self.pot_q: float = eval_cfg.get('pot_q', 0.001)
        self.pot_level: float = eval_cfg.get('pot_level', 0.98)
        
        model_cfg = config.get('model', {})
        self.window_size: int = model_cfg.get('window_size', 100)
        
        # Move model to appropriate device
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lara.to(self.device)

    @torch.no_grad()
    def compute_scores(self, loader: DataLoader) -> np.ndarray:
        """
        Computes point-wise anomaly scores for the test data.
        Anomaly score is defined as the reconstruction error of the LARA model.

        Args:
            loader: DataLoader containing the target test distribution windows.

        Returns:
            scores: Array of point-wise anomaly scores.
        """
        self.lara.eval()
        all_window_errors = []

        for x_batch, _ in loader:
            x_batch = x_batch.to(self.device)
            # Use LARA wrapper's internal method to get reconstruction error [Batch]
            # Error is sum of squared differences: ||X - Mx(Decoder(Mz(Encoder(X))))||^2
            batch_errors = self.lara.get_reconstruction_error(x_batch)
            all_window_errors.append(batch_errors.cpu().numpy())

        # Concatenate errors from all batches
        window_scores = np.concatenate(all_window_errors)
        
        # Map window-based scores back to point-based scores.
        # Following common practice (e.g., OmniAnomaly), the score of a window is 
        # assigned to its last timestamp.
        # To match the original sequence length, we pad the beginning with the first window's score.
        point_scores = np.zeros(len(window_scores) + self.window_size - 1)
        point_scores[self.window_size - 1:] = window_scores
        point_scores[:self.window_size - 1] = window_scores[0]
        
        return point_scores

    def run_pot_thresholding(self, scores: np.ndarray) -> np.ndarray:
        """
        Applies the Peak-Over-Threshold (POT) algorithm to determine a dynamic threshold.
        Based on Siffer et al. (2017), "Anomaly Detection in Streams with Extreme Value Theory".

        Args:
            scores: Array of point-wise reconstruction errors.

        Returns:
            predict: Binary predictions (0: normal, 1: anomaly).
        """
        # Step 1: Set initial threshold based on (1 - q) quantile
        # This initial threshold helps filter out the "normal" part of the distribution
        t = np.quantile(scores, 1 - self.pot_q)
        
        # Step 2: Extract peaks (exceedances)
        peaks = scores[scores > t] - t
        
        if len(peaks) < 10:
            # If not enough peaks to fit GPD, fallback to static threshold
            print("Warning: Too few peaks for POT. Using static quantile threshold.")
            threshold = t
        else:
            # Step 3: Fit the Generalized Pareto Distribution (GPD) to the peaks
            # Scipy's genpareto uses: c=shape (gamma), loc=0, scale=sigma
            try:
                # We fit with fixed location 0
                gamma, _, sigma = genpareto.fit(peaks, floc=0)
                
                # Step 4: Calculate the final threshold using the risk level
                # formula: z_p = t + (sigma / gamma) * ( ( (n * q) / k )^-gamma - 1 )
                # n: total samples, k: number of peaks, q: risk probability (1 - pot_level)
                n = len(scores)
                k = len(peaks)
                q_risk = 1 - self.pot_level
                
                # Avoid division by zero in gamma
                if abs(gamma) < 1e-7:
                    threshold = t - sigma * np.log(q_risk * n / k)
                else:
                    threshold = t + (sigma / gamma) * (np.power((n * q_risk) / k, -gamma) - 1)
            except Exception as e:
                print(f"Error fitting GPD: {e}. Using static quantile threshold.")
                threshold = t

        # Apply threshold to generate binary predictions
        predict = (scores > threshold).astype(np.float32)
        return predict

    def apply_point_adjustment(self, predict: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Applies the point-adjustment heuristic used in time-series anomaly detection.
        If any point in a ground-truth anomaly segment is detected, the entire segment
        is marked as correctly detected.

        Args:
            predict: Initial binary predictions.
            labels: Ground-truth binary labels.

        Returns:
            adjusted_predict: Binary predictions after segment adjustment.
        """
        adjusted_predict = predict.copy()
        
        # Identify start and end indices of ground-truth anomaly segments
        anomaly_state = False
        segment_start = 0
        
        for i in range(len(labels)):
            if labels[i] > 0 and not anomaly_state:
                anomaly_state = True
                segment_start = i
            elif labels[i] == 0 and anomaly_state:
                anomaly_state = False
                segment_end = i # exclusivity: [start, end)
                
                # Check if any point in this segment was detected
                if np.any(predict[segment_start:segment_end] > 0):
                    adjusted_predict[segment_start:segment_end] = 1.0
            
            # Handle case where anomaly segment reaches the end of the array
            if i == len(labels) - 1 and anomaly_state:
                if np.any(predict[segment_start:] > 0):
                    adjusted_predict[segment_start:] = 1.0
                    
        return adjusted_predict

    def get_metrics(self, scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculates evaluation metrics (Precision, Recall, F1) after POT and adjustment.

        Args:
            scores: Point-wise reconstruction errors.
            labels: Ground-truth point-wise labels.

        Returns:
            metrics: Dictionary containing 'f1', 'precision', and 'recall'.
        """
        # 1. Run POT to get initial predictions
        predict = self.run_pot_thresholding(scores)
        
        # 2. Apply segment adjustment (standard for TSAD evaluation)
        adjusted_predict = self.apply_point_adjustment(predict, labels)
        
        # 3. Calculate final metrics
        precision = precision_score(labels, adjusted_predict, zero_division=0)
        recall = recall_score(labels, adjusted_predict, zero_division=0)
        f1 = f1_score(labels, adjusted_predict, zero_division=0)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

