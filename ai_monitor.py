import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, List, Deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AIMonitor:
    """AI module for monitoring node behavior"""
    
    def __init__(self):
        self.node_behavior_history = {}
        self.anomaly_detector = IsolationForest(
            contamination=0.15, 
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.trust_scores = {}
        self.metrics_window = 100
        self.anomaly_threshold = 0.7
        
    def initialize_node_history(self, node_id: str):
        """Initialize behavior history for a new node"""
        if node_id not in self.node_behavior_history:
            self.node_behavior_history[node_id] = deque(maxlen=self.metrics_window)
            self.trust_scores[node_id] = 80.0
    
    def record_behavior_metric(self, node_id: str, metric_type: str, value: float, timestamp: datetime = None):
        """Record a behavior metric for a node"""
        if node_id not in self.node_behavior_history:
            self.initialize_node_history(node_id)
        
        timestamp = timestamp or datetime.now()
        metric = {
            'timestamp': timestamp,
            'type': metric_type,
            'value': value,
            'node_id': node_id
        }
        
        self.node_behavior_history[node_id].append(metric)
        self._update_trust_score(node_id, metric_type, value)
        
        if len(self.node_behavior_history[node_id]) % 20 == 0:
            self._update_anomaly_detector()
    
    def _update_trust_score(self, node_id: str, metric_type: str, value: float):
        """Update node trust score"""
        if node_id not in self.trust_scores:
            self.trust_scores[node_id] = 80.0
        
        base_score = self.trust_scores[node_id]
        adjustment = 0.0
        
        if metric_type == 'response_time':
            adjustment = -np.exp(value / 1000) * 0.5
        elif metric_type == 'availability':
            adjustment = value * 0.2
        elif metric_type == 'fragment_delivery':
            adjustment = 2.0 if value == 1 else -3.0
        elif metric_type == 'consensus_participation':
            adjustment = 1.0 if value == 1 else -2.0
        elif metric_type == 'integrity_check':
            adjustment = 3.0 if value == 1 else -5.0
        
        new_score = base_score + adjustment
        new_score = max(0.0, min(100.0, new_score))
        decay = 0.995
        self.trust_scores[node_id] = new_score * decay
    
    def _update_anomaly_detector(self):
        """Update anomaly detection model"""
        try:
            all_metrics = []
            node_ids = []
            
            for node_id, metrics in self.node_behavior_history.items():
                if len(metrics) > 5:
                    recent_metrics = list(metrics)[-5:]
                    for metric in recent_metrics:
                        feature_vector = [
                            metric['value'],
                            self.trust_scores.get(node_id, 80),
                            len(recent_metrics)
                        ]
                        all_metrics.append(feature_vector)
                        node_ids.append(node_id)
            
            if len(all_metrics) > 10:
                X = np.array(all_metrics)
                X_scaled = self.scaler.fit_transform(X)
                self.anomaly_detector.fit(X_scaled)
                self.is_trained = True
        except Exception as e:
            print(f"Anomaly detector training failed: {e}")
    
    def detect_anomalies(self) -> Dict[str, Dict]:
        """Detect anomalies across all nodes"""
        anomalies = {}
        
        if not self.is_trained:
            return anomalies
        
        try:
            for node_id, metrics in self.node_behavior_history.items():
                if len(metrics) > 3:
                    recent_metrics = list(metrics)[-3:]
                    feature_vectors = []
                    
                    for metric in recent_metrics:
                        feature_vector = [
                            metric['value'],
                            self.trust_scores.get(node_id, 80),
                            1.0
                        ]
                        feature_vectors.append(feature_vector)
                    
                    if feature_vectors:
                        X = np.array(feature_vectors)
                        X_scaled = self.scaler.transform(X)
                        scores = self.anomaly_detector.decision_function(X_scaled)
                        anomaly_score = np.mean(scores)
                        
                        if anomaly_score < self.anomaly_threshold:
                            anomalies[node_id] = {
                                'score': anomaly_score,
                                'trust_score': self.trust_scores.get(node_id, 80),
                                'metrics': len(recent_metrics),
                                'severity': 'high' if anomaly_score < 0 else 'medium'
                            }
        except Exception as e:
            print(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def get_node_trust_level(self, node_id: str) -> str:
        """Get trust level classification"""
        trust_score = self.trust_scores.get(node_id, 80)
        
        if trust_score >= 90:
            return "High Trust"
        elif trust_score >= 70:
            return "Medium Trust"
        elif trust_score >= 50:
            return "Low Trust"
        else:
            return "Untrusted"
    
    def get_recommended_nodes(self, required_nodes: int, exclude_nodes: List[str] = None) -> List[str]:
        """Get recommended nodes based on trust scores"""
        exclude_nodes = exclude_nodes or []
        
        eligible_nodes = [
            (node_id, score) for node_id, score in self.trust_scores.items()
            if node_id not in exclude_nodes
        ]
        
        if not eligible_nodes:
            return []
        
        eligible_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in eligible_nodes[:min(required_nodes, len(eligible_nodes))]]
    
    def get_behavior_analytics(self, node_id: str) -> Dict:
        """Get behavior analytics for a node"""
        if node_id not in self.node_behavior_history:
            return {}
        
        metrics = list(self.node_behavior_history[node_id])
        
        if not metrics:
            return {}
        
        response_times = [m['value'] for m in metrics if m['type'] == 'response_time']
        availability = [m['value'] for m in metrics if m['type'] == 'availability']
        
        analytics = {
            'total_metrics': len(metrics),
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'avg_availability': np.mean(availability) if availability else 100,
            'trust_score': self.trust_scores.get(node_id, 80),
            'trust_level': self.get_node_trust_level(node_id),
            'recent_activity': [
                {
                    'type': m['type'],
                    'value': m['value'],
                    'time': m['timestamp'].strftime('%H:%M:%S')
                }
                for m in metrics[-5:]
            ]
        }
        
        return analytics