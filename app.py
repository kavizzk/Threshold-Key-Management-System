
import streamlit as st
import pandas as pd
import numpy as np
import json
import uuid
import time
import os
import io
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Deque
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import hashlib
import base64
import secrets
import warnings
warnings.filterwarnings('ignore')

# Cryptography imports
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hmac

# Machine Learning imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Network imports
import networkx as nx

# Set page configuration
st.set_page_config(
    page_title="AI-Enabled Threshold KMS",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2E86AB;
        padding-bottom: 0.5rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #2E86AB;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .ai-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .node-status-online {
        color: #28a745;
        font-weight: bold;
    }
    .node-status-offline {
        color: #dc3545;
        font-weight: bold;
    }
    .node-status-suspicious {
        color: #ffc107;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.05);
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #dc3545;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
    .threshold-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: bold;
        margin: 2px;
    }
    .threshold-met {
        background-color: #28a745;
        color: white;
    }
    .threshold-not-met {
        background-color: #dc3545;
        color: white;
    }
    .file-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #dee2e6;
    }
    .message-bubble {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        max-width: 80%;
    }
    .message-sent {
        background-color: #d4edda;
        margin-left: auto;
    }
    .message-received {
        background-color: #f8f9fa;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA MODELS
# ============================================

class KeyFragment:
    def __init__(self, fragment_id: str, key_id: str, data: bytes, 
                 node_id: str, timestamp: datetime, signature: bytes = None,
                 share_index: int = 0):
        self.fragment_id = fragment_id
        self.key_id = key_id
        self.data = data
        self.node_id = node_id
        self.timestamp = timestamp
        self.signature = signature
        self.share_index = share_index
        self.is_verified = False
        
    def to_dict(self):
        return {
            "fragment_id": self.fragment_id,
            "key_id": self.key_id,
            "data": base64.b64encode(self.data).decode() if self.data else None,
            "node_id": self.node_id,
            "timestamp": self.timestamp.isoformat(),
            "signature": base64.b64encode(self.signature).decode() if self.signature else None,
            "share_index": self.share_index,
            "is_verified": self.is_verified
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            fragment_id=data["fragment_id"],
            key_id=data["key_id"],
            data=base64.b64decode(data["data"]) if data["data"] else None,
            node_id=data["node_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            signature=base64.b64decode(data["signature"]) if data.get("signature") else None,
            share_index=data.get("share_index", 0)
        )

class KeyMetadata:
    def __init__(self, key_id: str, key_type: str, key_size: int, 
                 owner_id: str, fragment_count: int, threshold: int,
                 creation_date: datetime, expiry_date: datetime = None,
                 usage_policy: dict = None, algorithm: str = "AES"):
        self.key_id = key_id
        self.key_type = key_type
        self.key_size = key_size
        self.owner_id = owner_id
        self.fragment_count = fragment_count
        self.threshold = threshold
        self.creation_date = creation_date
        self.expiry_date = expiry_date or (creation_date + timedelta(days=365))
        self.usage_policy = usage_policy or {"allowed_operations": ["encrypt", "decrypt"]}
        self.algorithm = algorithm
        self.is_active = True
        self.last_accessed = creation_date
        self.access_count = 0
        
    def to_dict(self):
        return {
            "key_id": self.key_id,
            "key_type": self.key_type,
            "key_size": self.key_size,
            "owner_id": self.owner_id,
            "fragment_count": self.fragment_count,
            "threshold": self.threshold,
            "creation_date": self.creation_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat(),
            "usage_policy": self.usage_policy,
            "algorithm": self.algorithm,
            "is_active": self.is_active,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        metadata = cls(
            key_id=data["key_id"],
            key_type=data["key_type"],
            key_size=data["key_size"],
            owner_id=data["owner_id"],
            fragment_count=data["fragment_count"],
            threshold=data["threshold"],
            creation_date=datetime.fromisoformat(data["creation_date"]),
            expiry_date=datetime.fromisoformat(data["expiry_date"]),
            usage_policy=data["usage_policy"],
            algorithm=data.get("algorithm", "AES")
        )
        metadata.is_active = data["is_active"]
        metadata.last_accessed = datetime.fromisoformat(data["last_accessed"])
        metadata.access_count = data["access_count"]
        return metadata

class Node:
    def __init__(self, node_id: str, name: str, ip_address: str, 
                 public_key: str, location: str = "Unknown", node_type: str = "KMN"):
        self.node_id = node_id
        self.name = name
        self.ip_address = ip_address
        self.public_key = public_key
        self.location = location
        self.node_type = node_type
        self.reputation_score = 100.0
        self.trust_score = 80.0
        self.last_seen = datetime.now()
        self.storage_capacity = 1000
        self.storage_used = 0
        self.is_online = True
        self.is_trusted = True
        self.fragments_stored = []
        self.response_time_avg = 0.0
        self.uptime_percentage = 100.0
        self.last_10_responses = deque(maxlen=10)
        self.ai_anomaly_score = 0.0
        
    def to_dict(self):
        return {
            "node_id": self.node_id,
            "name": self.name,
            "ip_address": self.ip_address,
            "public_key": self.public_key,
            "location": self.location,
            "node_type": self.node_type,
            "reputation_score": self.reputation_score,
            "trust_score": self.trust_score,
            "last_seen": self.last_seen.isoformat(),
            "storage_capacity": self.storage_capacity,
            "storage_used": self.storage_used,
            "is_online": self.is_online,
            "is_trusted": self.is_trusted,
            "fragments_stored": self.fragments_stored,
            "response_time_avg": self.response_time_avg,
            "uptime_percentage": self.uptime_percentage,
            "ai_anomaly_score": self.ai_anomaly_score
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        node = cls(
            node_id=data["node_id"],
            name=data["name"],
            ip_address=data["ip_address"],
            public_key=data["public_key"],
            location=data["location"],
            node_type=data.get("node_type", "KMN")
        )
        node.reputation_score = data["reputation_score"]
        node.trust_score = data.get("trust_score", 80.0)
        node.last_seen = datetime.fromisoformat(data["last_seen"])
        node.storage_capacity = data["storage_capacity"]
        node.storage_used = data["storage_used"]
        node.is_online = data["is_online"]
        node.is_trusted = data.get("is_trusted", True)
        node.fragments_stored = data["fragments_stored"]
        node.response_time_avg = data.get("response_time_avg", 0.0)
        node.uptime_percentage = data.get("uptime_percentage", 100.0)
        node.ai_anomaly_score = data.get("ai_anomaly_score", 0.0)
        return node

class User:
    def __init__(self, user_id: str, username: str, email: str, password_hash: str, 
                 role: str = "user", is_active: bool = True):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role
        self.is_active = is_active
        self.created_at = datetime.now()
        self.last_login = datetime.now()
        self.key_access = []
        self.encryption_keys = {}
        self.received_files = []  # List of received file metadata
        self.sent_files = []      # List of sent file metadata
        
    def to_dict(self):
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat(),
            "key_access": self.key_access,
            "encryption_keys": {k: base64.b64encode(v).decode() for k, v in self.encryption_keys.items()},
            "received_files": self.received_files,
            "sent_files": self.sent_files
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        user = cls(
            user_id=data["user_id"],
            username=data["username"],
            email=data["email"],
            password_hash=data["password_hash"],
            role=data["role"],
            is_active=data["is_active"]
        )
        user.created_at = datetime.fromisoformat(data["created_at"])
        user.last_login = datetime.fromisoformat(data["last_login"])
        user.key_access = data["key_access"]
        user.encryption_keys = {k: base64.b64decode(v) for k, v in data.get("encryption_keys", {}).items()}
        user.received_files = data.get("received_files", [])
        user.sent_files = data.get("sent_files", [])
        return user

class EncryptedFile:
    def __init__(self, file_id: str, filename: str, encrypted_data: bytes, 
                 key_id: str, sender_id: str, receiver_id: str, 
                 timestamp: datetime, metadata: dict = None):
        self.file_id = file_id
        self.filename = filename
        self.encrypted_data = encrypted_data
        self.key_id = key_id
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.timestamp = timestamp
        self.metadata = metadata or {}
        self.is_downloaded = False
        self.downloaded_at = None
        
    def to_dict(self):
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "encrypted_data": base64.b64encode(self.encrypted_data).decode(),
            "key_id": self.key_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "is_downloaded": self.is_downloaded,
            "downloaded_at": self.downloaded_at.isoformat() if self.downloaded_at else None
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        file = cls(
            file_id=data["file_id"],
            filename=data["filename"],
            encrypted_data=base64.b64decode(data["encrypted_data"]),
            key_id=data["key_id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )
        file.is_downloaded = data.get("is_downloaded", False)
        if data.get("downloaded_at"):
            file.downloaded_at = datetime.fromisoformat(data["downloaded_at"])
        return file

# ============================================
# CRYPTOGRAPHY UTILITIES
# ============================================

class CryptoUtils:
    """Cryptography utilities for threshold secret sharing"""
    
    @staticmethod
    def generate_key_pair():
        """Generate RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        
        pem_public = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_key, pem_public.decode()
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = secrets.token_bytes(16)
        password_bytes = password.encode()
        hashed = hashlib.pbkdf2_hmac('sha256', password_bytes, salt, 100000)
        return f"{salt.hex()}:{hashed.hex()}"
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password against hashed password"""
        try:
            salt_hex, hashed_hex = hashed_password.split(':')
            salt = bytes.fromhex(salt_hex)
            stored_hash = bytes.fromhex(hashed_hex)
            
            password_bytes = password.encode()
            new_hash = hashlib.pbkdf2_hmac('sha256', password_bytes, salt, 100000)
            
            return hmac.compare_digest(new_hash, stored_hash)
        except:
            return False
    
    @staticmethod
    def shamir_secret_sharing(secret: bytes, n: int, k: int) -> List[Tuple[int, bytes]]:
        """
        Implement (k, n) threshold secret sharing
        """
        if k > n:
            raise ValueError("k must be <= n")
        if k < 2:
            raise ValueError("k must be at least 2")
        if n < k:
            raise ValueError("n must be >= k")
        
        # Convert secret to integer
        secret_int = int.from_bytes(secret, 'big')
        
        # Generate random coefficients
        coefficients = [secret_int]
        for _ in range(k-1):
            coeff = secrets.randbelow(2**256)
            coefficients.append(coeff)
        
        # Generate n shares
        shares = []
        for x in range(1, n+1):
            y = 0
            for coeff_idx, coeff in enumerate(coefficients):
                y += coeff * (x ** coeff_idx)
            
            y_bytes = y.to_bytes((y.bit_length() + 7) // 8, 'big')
            share_data = json.dumps({
                'x': x,
                'y': y_bytes.hex(),
                'k': k,
                'n': n
            }).encode()
            
            shares.append((x, share_data))
        
        return shares
    
    @staticmethod
    def reconstruct_secret(shares: List[Tuple[int, bytes]], k: int) -> bytes:
        """
        Reconstruct secret from shares
        """
        if len(shares) < k:
            raise ValueError(f"Need at least {k} shares, got {len(shares)}")
        
        points = []
        for x, share_data in shares[:k]:
            try:
                data = json.loads(share_data.decode())
                y_bytes = bytes.fromhex(data['y'])
                y_int = int.from_bytes(y_bytes, 'big')
                points.append((x, y_int))
            except:
                continue
        
        if len(points) < k:
            raise ValueError("Invalid shares format")
        
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        
        secret_int = 0
        for i in range(k):
            xi, yi = x_points[i], y_points[i]
            
            li = 1
            for j in range(k):
                if i != j:
                    xj = x_points[j]
                    li *= (0 - xj) / (xi - xj)
            
            secret_int += yi * int(round(li))
        
        return secret_int.to_bytes((secret_int.bit_length() + 7) // 8, 'big')
    
    @staticmethod
    def encrypt_with_aes(data: bytes, key: bytes) -> bytes:
        """Encrypt data using AES-GCM"""
        iv = secrets.token_bytes(12)
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return iv + encryptor.tag + ciphertext
    
    @staticmethod
    def encrypt_file(file_bytes: bytes, key: bytes) -> bytes:
        """Encrypt file using AES-GCM"""
        return CryptoUtils.encrypt_with_aes(file_bytes, key)
    
    @staticmethod
    def decrypt_with_aes(encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data using AES-GCM"""
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    @staticmethod
    def decrypt_file(encrypted_file_bytes: bytes, key: bytes) -> bytes:
        """Decrypt file using AES-GCM"""
        return CryptoUtils.decrypt_with_aes(encrypted_file_bytes, key)
    
    @staticmethod
    def generate_symmetric_key(key_size: int = 32) -> bytes:
        """Generate symmetric key"""
        return secrets.token_bytes(key_size)
    
    @staticmethod
    def calculate_hash(data: bytes) -> str:
        """Calculate SHA-256 hash"""
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def create_hmac(data: bytes, key: bytes) -> bytes:
        """Create HMAC for data integrity"""
        return hmac.new(key, data, hashlib.sha256).digest()
    
    @staticmethod
    def verify_hmac(data: bytes, key: bytes, hmac_value: bytes) -> bool:
        """Verify HMAC"""
        expected_hmac = hmac.new(key, data, hashlib.sha256).digest()
        return hmac.compare_digest(expected_hmac, hmac_value)

# ============================================
# AI MONITORING SYSTEM
# ============================================

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

# ============================================
# AI-ENABLED THRESHOLD KEY MANAGEMENT SYSTEM
# ============================================

class AIThresholdKMS:
    """Main AI-enabled Threshold Key Management System"""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.key_metadata: Dict[str, KeyMetadata] = {}
        self.fragments: Dict[str, List[KeyFragment]] = {}
        self.users: Dict[str, User] = {}
        self.encrypted_files: Dict[str, EncryptedFile] = {}
        self.audit_log: List[dict] = []
        self.ai_monitor = AIMonitor()
        self.consensus_threshold = 0.6
        self.network_graph = nx.Graph()
        self.failure_simulation_mode = False
        self.failed_nodes = set()
        self._initialize_default_data()
    
    def _initialize_default_data(self):
        """Initialize system with default data"""
        # Create admin user with default password
        admin_password_hash = CryptoUtils.hash_password("admin123")
        admin = User("admin-001", "admin", "admin@tkms.com", admin_password_hash, "admin", True)
        self.users[admin.user_id] = admin
        
        # Create sample users
        sample_users = [
            ("alice", "alice@email.com", "alice123", "user"),
            ("bob", "bob@email.com", "bob123", "user"),
            ("charlie", "charlie@email.com", "charlie123", "user"),
        ]
        
        for i, (username, email, password, role) in enumerate(sample_users, 1):
            user_id = f"user-{i:03d}"
            password_hash = CryptoUtils.hash_password(password)
            user = User(user_id, username, email, password_hash, role, True)
            self.users[user_id] = user
        
        # Create initial Key Management Nodes (KMNs)
        locations = ["US-East", "US-West", "EU-Central", "APAC-North", "APAC-South"]
        for i in range(5):
            node_id = f"kmn-{i+1:03d}"
            _, public_key = CryptoUtils.generate_key_pair()
            
            node = Node(
                node_id=node_id,
                name=f"KMN-{i+1}",
                ip_address=f"10.0.{i+1}.100",
                public_key=public_key,
                location=locations[i],
                node_type="KMN"
            )
            
            self.ai_monitor.initialize_node_history(node_id)
            self.ai_monitor.record_behavior_metric(node_id, 'availability', 100)
            self.ai_monitor.record_behavior_metric(node_id, 'response_time', np.random.uniform(10, 100))
            
            self.nodes[node_id] = node
            self.network_graph.add_node(node_id, **node.to_dict())
        
        # Connect nodes
        node_ids = list(self.nodes.keys())
        for i, node1 in enumerate(node_ids):
            for node2 in node_ids[i+1:]:
                if np.random.random() > 0.3:
                    self.network_graph.add_edge(node1, node2, weight=np.random.uniform(0.5, 1.0))
        
        # Create sample keys
        self._create_sample_keys()
        
        # Create sample encrypted files
        self._create_sample_files()
    
    def _create_sample_keys(self):
        """Create sample keys with AI-based distribution"""
        key_configs = [
            {"type": "AES-256", "size": 256, "fragments": 5, "threshold": 3},
            {"type": "AES-192", "size": 192, "fragments": 4, "threshold": 2},
            {"type": "AES-128", "size": 128, "fragments": 3, "threshold": 2},
        ]
        
        for i, config in enumerate(key_configs):
            key_id = f"key-{i+1:03d}"
            
            metadata = KeyMetadata(
                key_id=key_id,
                key_type=config["type"],
                key_size=config["size"],
                owner_id="admin-001",
                fragment_count=config["fragments"],
                threshold=config["threshold"],
                creation_date=datetime.now() - timedelta(days=np.random.randint(1, 30)),
                algorithm="AES"
            )
            
            self.key_metadata[key_id] = metadata
            self._ai_distribute_fragments(key_id, config["fragments"], config["threshold"])
    
    def _create_sample_files(self):
        """Create sample encrypted files"""
        sample_files = [
            ("document.pdf", "admin-001", "user-001", "Important document"),
            ("image.png", "user-001", "user-002", "Project image"),
            ("report.docx", "user-002", "admin-001", "Monthly report"),
        ]
        
        for filename, sender_id, receiver_id, description in sample_files:
            if sender_id in self.users and receiver_id in self.users:
                # Create a sample key for the file
                file_key = CryptoUtils.generate_symmetric_key(32)
                key_id = f"file-key-{str(uuid.uuid4())[:8]}"
                
                # Encrypt sample content
                sample_content = f"This is a sample {filename} - {description}".encode()
                encrypted_data = CryptoUtils.encrypt_file(sample_content, file_key)
                
                file_id = f"file-{str(uuid.uuid4())[:8]}"
                file = EncryptedFile(
                    file_id=file_id,
                    filename=filename,
                    encrypted_data=encrypted_data,
                    key_id=key_id,
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    timestamp=datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                    metadata={
                        "description": description,
                        "size": len(sample_content),
                        "encrypted_size": len(encrypted_data)
                    }
                )
                
                self.encrypted_files[file_id] = file
                
                # Store the encryption key with the sender
                if sender_id in self.users:
                    self.users[sender_id].encryption_keys[key_id] = file_key
    
    def _ai_distribute_fragments(self, key_id: str, n: int, k: int):
        """AI-based fragment distribution across nodes"""
        # Generate the key
        key = CryptoUtils.generate_symmetric_key(32)
        
        # Create secret shares
        shares = CryptoUtils.shamir_secret_sharing(key, n, k)
        
        # Get AI-recommended nodes
        recommended_nodes = self.ai_monitor.get_recommended_nodes(n)
        
        if len(recommended_nodes) < n:
            # Fallback to random selection if not enough recommended nodes
            all_nodes = list(self.nodes.keys())
            recommended_nodes = list(all_nodes)[:n]
        
        fragments = []
        for i, (share_index, share_data) in enumerate(shares):
            if i < len(recommended_nodes):
                node_id = recommended_nodes[i]
                fragment_id = f"frag-{key_id}-{i:03d}"
                
                fragment = KeyFragment(
                    fragment_id=fragment_id,
                    key_id=key_id,
                    data=share_data,
                    node_id=node_id,
                    timestamp=datetime.now(),
                    share_index=share_index
                )
                
                fragments.append(fragment)
                
                # Update node storage
                if node_id in self.nodes:
                    self.nodes[node_id].fragments_stored.append(fragment_id)
                    self.nodes[node_id].storage_used += len(share_data)
        
        self.fragments[key_id] = fragments
        
        # Log the distribution
        self.log_audit_event(
            event_type="KEY_FRAGMENT_DISTRIBUTION",
            user_id="system",
            details={
                "key_id": key_id,
                "fragment_count": n,
                "threshold": k,
                "nodes_used": recommended_nodes[:n]
            }
        )
    
    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[User], str]:
        """Authenticate user with username/password"""
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            return False, None, "User not found"
        
        if not user.is_active:
            return False, None, "User account is inactive"
        
        if CryptoUtils.verify_password(password, user.password_hash):
            user.last_login = datetime.now()
            return True, user, "Authentication successful"
        
        return False, None, "Invalid password"
    
    def register_user(self, username: str, email: str, password: str, role: str = "user") -> Tuple[bool, str]:
        """Register a new user"""
        # Check if username exists
        for user in self.users.values():
            if user.username == username:
                return False, "Username already exists"
            if user.email == email:
                return False, "Email already registered"
        
        # Create new user
        user_id = f"user-{len(self.users):03d}"
        password_hash = CryptoUtils.hash_password(password)
        
        new_user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            is_active=True
        )
        
        self.users[user_id] = new_user
        
        self.log_audit_event(
            event_type="USER_REGISTRATION",
            user_id="system",
            details={
                "user_id": user_id,
                "username": username,
                "role": role
            }
        )
        
        return True, f"User {username} registered successfully"
    
    def create_key(self, key_type: str, key_size: int, fragments: int, 
                   threshold: int, owner_id: str, algorithm: str = "AES") -> Tuple[bool, str]:
        """Create a new key with threshold distribution"""
        if fragments < 2:
            return False, "At least 2 fragments required"
        if threshold < 2:
            return False, "At least threshold of 2 required"
        if threshold > fragments:
            return False, "Threshold cannot exceed number of fragments"
        
        key_id = f"key-{str(uuid.uuid4())[:8]}"
        
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            key_size=key_size,
            owner_id=owner_id,
            fragment_count=fragments,
            threshold=threshold,
            creation_date=datetime.now(),
            algorithm=algorithm
        )
        
        self.key_metadata[key_id] = metadata
        
        # Distribute fragments using AI
        self._ai_distribute_fragments(key_id, fragments, threshold)
        
        # Grant owner access
        if owner_id in self.users:
            self.users[owner_id].key_access.append(key_id)
        
        self.log_audit_event(
            event_type="KEY_CREATION",
            user_id=owner_id,
            details={
                "key_id": key_id,
                "key_type": key_type,
                "fragments": fragments,
                "threshold": threshold
            }
        )
        
        return True, f"Key {key_id} created successfully"
    
    def reconstruct_key(self, key_id: str, requester_id: str) -> Tuple[bool, Optional[bytes], str]:
        """Reconstruct key from fragments"""
        if key_id not in self.key_metadata:
            return False, None, "Key not found"
        
        metadata = self.key_metadata[key_id]
        
        # Check access permission
        if requester_id not in ["admin-001", metadata.owner_id] and key_id not in self.users.get(requester_id, User("", "", "", "", "")).key_access:
            return False, None, "Access denied"
        
        # Check key status
        if not metadata.is_active:
            return False, None, "Key is inactive"
        
        if datetime.now() > metadata.expiry_date:
            return False, None, "Key has expired"
        
        # Get available fragments
        fragments = self.fragments.get(key_id, [])
        
        if len(fragments) < metadata.threshold:
            return False, None, f"Insufficient fragments. Need {metadata.threshold}, have {len(fragments)}"
        
        # AI check for suspicious nodes
        suspicious_fragments = []
        for fragment in fragments:
            node = self.nodes.get(fragment.node_id)
            if node and node.ai_anomaly_score > 0.7:
                suspicious_fragments.append(fragment.fragment_id)
        
        if suspicious_fragments and len(fragments) - len(suspicious_fragments) < metadata.threshold:
            return False, None, "Insufficient trusted fragments"
        
        # Collect shares for reconstruction
        shares = []
        used_fragments = []
        
        for fragment in fragments:
            if len(shares) >= metadata.threshold:
                break
            
            if fragment.is_verified or fragment.node_id not in self.failed_nodes:
                try:
                    shares.append((fragment.share_index, fragment.data))
                    used_fragments.append(fragment.fragment_id)
                except:
                    continue
        
        if len(shares) < metadata.threshold:
            return False, None, "Could not collect enough valid shares"
        
        try:
            # Reconstruct the key
            key = CryptoUtils.reconstruct_secret(shares, metadata.threshold)
            
            # Update key metadata
            metadata.last_accessed = datetime.now()
            metadata.access_count += 1
            
            self.log_audit_event(
                event_type="KEY_RECONSTRUCTION",
                user_id=requester_id,
                details={
                    "key_id": key_id,
                    "fragments_used": len(shares),
                    "threshold": metadata.threshold
                }
            )
            
            return True, key, f"Key reconstructed successfully using {len(shares)} fragments"
        except Exception as e:
            return False, None, f"Reconstruction failed: {str(e)}"
    
    def encrypt_file_for_user(self, file_bytes: bytes, filename: str, 
                             sender_id: str, receiver_id: str, 
                             description: str = "") -> Tuple[bool, str, Optional[str]]:
        """Encrypt a file and share with another user"""
        if sender_id not in self.users:
            return False, "Sender not found", None
        if receiver_id not in self.users:
            return False, "Receiver not found", None
        
        # Create a new encryption key
        key = CryptoUtils.generate_symmetric_key(32)
        key_id = f"file-key-{str(uuid.uuid4())[:8]}"
        
        # Encrypt the file
        encrypted_data = CryptoUtils.encrypt_file(file_bytes, key)
        
        # Create encrypted file record
        file_id = f"file-{str(uuid.uuid4())[:8]}"
        file = EncryptedFile(
            file_id=file_id,
            filename=filename,
            encrypted_data=encrypted_data,
            key_id=key_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            timestamp=datetime.now(),
            metadata={
                "description": description,
                "size": len(file_bytes),
                "encrypted_size": len(encrypted_data),
                "hash": CryptoUtils.calculate_hash(file_bytes)
            }
        )
        
        self.encrypted_files[file_id] = file
        
        # Store the encryption key with sender
        self.users[sender_id].encryption_keys[key_id] = key
        
        # Add to user's file lists
        self.users[sender_id].sent_files.append({
            "file_id": file_id,
            "filename": filename,
            "receiver_id": receiver_id,
            "timestamp": datetime.now().isoformat(),
            "description": description
        })
        
        self.users[receiver_id].received_files.append({
            "file_id": file_id,
            "filename": filename,
            "sender_id": sender_id,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "is_downloaded": False
        })
        
        self.log_audit_event(
            event_type="FILE_ENCRYPTION",
            user_id=sender_id,
            details={
                "file_id": file_id,
                "filename": filename,
                "receiver_id": receiver_id,
                "size": len(file_bytes)
            }
        )
        
        return True, f"File '{filename}' encrypted and sent to user {receiver_id}", file_id
    
    def decrypt_file(self, file_id: str, requester_id: str) -> Tuple[bool, Optional[bytes], str, Optional[str]]:
        """Decrypt a file"""
        if file_id not in self.encrypted_files:
            return False, None, "File not found", None
        
        file = self.encrypted_files[file_id]
        
        # Check permissions
        if requester_id not in [file.sender_id, file.receiver_id, "admin-001"]:
            return False, None, "Access denied", None
        
        # Get the encryption key
        key = None
        if file.sender_id in self.users:
            key = self.users[file.sender_id].encryption_keys.get(file.key_id)
        
        if not key:
            return False, None, "Encryption key not found", None
        
        try:
            # Decrypt the file
            decrypted_data = CryptoUtils.decrypt_file(file.encrypted_data, key)
            
            # Update file status if receiver is downloading
            if requester_id == file.receiver_id and not file.is_downloaded:
                file.is_downloaded = True
                file.downloaded_at = datetime.now()
            
            self.log_audit_event(
                event_type="FILE_DECRYPTION",
                user_id=requester_id,
                details={
                    "file_id": file_id,
                    "filename": file.filename
                }
            )
            
            return True, decrypted_data, "File decrypted successfully", file.filename
        except Exception as e:
            return False, None, f"Decryption failed: {str(e)}", None
    
    def rotate_key(self, key_id: str, user_id: str) -> Tuple[bool, str]:
        """Rotate (rekey) an existing key"""
        if key_id not in self.key_metadata:
            return False, "Key not found"
        
        metadata = self.key_metadata[key_id]
        
        # Check permissions
        if user_id not in ["admin-001", metadata.owner_id]:
            return False, "Access denied"
        
        # Reconstruct old key first
        success, old_key, message = self.reconstruct_key(key_id, user_id)
        if not success:
            return False, f"Cannot rotate: {message}"
        
        # Create new fragments
        fragments = self.fragments.get(key_id, [])
        old_fragments = len(fragments)
        threshold = metadata.threshold
        
        # Clear old fragments from nodes
        for fragment in fragments:
            if fragment.node_id in self.nodes:
                node = self.nodes[fragment.node_id]
                if fragment.fragment_id in node.fragments_stored:
                    node.fragments_stored.remove(fragment.fragment_id)
                    node.storage_used = max(0, node.storage_used - len(fragment.data))
        
        # Create new distribution
        self._ai_distribute_fragments(key_id, old_fragments, threshold)
        
        # Update metadata
        metadata.creation_date = datetime.now()
        metadata.expiry_date = datetime.now() + timedelta(days=365)
        metadata.last_accessed = datetime.now()
        
        self.log_audit_event(
            event_type="KEY_ROTATION",
            user_id=user_id,
            details={
                "key_id": key_id,
                "old_fragments": old_fragments,
                "new_fragments": old_fragments
            }
        )
        
        return True, f"Key {key_id} rotated successfully"
    
    def get_user_files(self, user_id: str) -> Dict[str, List]:
        """Get files sent and received by a user"""
        user = self.users.get(user_id)
        if not user:
            return {"sent": [], "received": []}
        
        sent_files = []
        for file_info in user.sent_files:
            file_id = file_info["file_id"]
            if file_id in self.encrypted_files:
                file = self.encrypted_files[file_id]
                sent_files.append({
                    "file_id": file_id,
                    "filename": file.filename,
                    "receiver": self.users.get(file.receiver_id, User("", "", "", "", "")).username,
                    "timestamp": file.timestamp,
                    "size": file.metadata.get("size", 0),
                    "description": file.metadata.get("description", "")
                })
        
        received_files = []
        for file_info in user.received_files:
            file_id = file_info["file_id"]
            if file_id in self.encrypted_files:
                file = self.encrypted_files[file_id]
                received_files.append({
                    "file_id": file_id,
                    "filename": file.filename,
                    "sender": self.users.get(file.sender_id, User("", "", "", "", "")).username,
                    "timestamp": file.timestamp,
                    "size": file.metadata.get("size", 0),
                    "description": file.metadata.get("description", ""),
                    "is_downloaded": file.is_downloaded
                })
        
        return {"sent": sent_files, "received": received_files}
    
    def simulate_node_failure(self, node_id: str):
        """Simulate a node failure"""
        if node_id in self.nodes:
            self.nodes[node_id].is_online = False
            self.failed_nodes.add(node_id)
            
            self.log_audit_event(
                event_type="NODE_FAILURE",
                user_id="system",
                details={
                    "node_id": node_id,
                    "name": self.nodes[node_id].name,
                    "time": datetime.now().isoformat()
                }
            )
    
    def restore_node(self, node_id: str):
        """Restore a failed node"""
        if node_id in self.nodes:
            self.nodes[node_id].is_online = True
            if node_id in self.failed_nodes:
                self.failed_nodes.remove(node_id)
            
            self.log_audit_event(
                event_type="NODE_RESTORATION",
                user_id="system",
                details={"node_id": node_id}
            )
    
    def log_audit_event(self, event_type: str, user_id: str, details: dict):
        """Log an audit event"""
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details
        }
        self.audit_log.append(event)
    
    def get_system_stats(self) -> dict:
        """Get system statistics"""
        total_keys = len(self.key_metadata)
        active_keys = sum(1 for k in self.key_metadata.values() if k.is_active)
        total_fragments = sum(len(frags) for frags in self.fragments.values())
        
        online_nodes = sum(1 for n in self.nodes.values() if n.is_online)
        suspicious_nodes = len(self.ai_monitor.detect_anomalies())
        
        total_users = len(self.users)
        total_files = len(self.encrypted_files)
        
        return {
            "total_keys": total_keys,
            "active_keys": active_keys,
            "total_fragments": total_fragments,
            "online_nodes": online_nodes,
            "total_nodes": len(self.nodes),
            "suspicious_nodes": suspicious_nodes,
            "total_users": total_users,
            "total_files": total_files,
            "audit_events": len(self.audit_log)
        }

# ============================================
# STREAMLIT UI APPLICATION
# ============================================

class KMSApp:
    """Streamlit application for the KMS"""
    
    def __init__(self):
        self.kms = AIThresholdKMS()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'show_registration' not in st.session_state:
            st.session_state.show_registration = False
        if 'show_file_upload' not in st.session_state:
            st.session_state.show_file_upload = False
        if 'selected_file_id' not in st.session_state:
            st.session_state.selected_file_id = None
        if 'admin_mode' not in st.session_state:
            st.session_state.admin_mode = False
        if 'admin_password' not in st.session_state:
            st.session_state.admin_password = ""
    
    def login_page(self):
        """Render login page"""
        st.markdown("<h1 class='main-header'>🔐 AI-Enabled Threshold KMS</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.container():
                st.markdown("### Login")
                
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Login", use_container_width=True):
                        success, user, message = self.kms.authenticate_user(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.current_user = user
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                
                with col_btn2:
                    if st.button("Register New User", use_container_width=True):
                        st.session_state.show_registration = True
                        st.rerun()
                
                st.divider()
                
                # Admin access
                st.markdown("#### Admin Access")
                admin_pass = st.text_input("Admin Password", type="password", key="admin_pass")
                if st.button("Enter Admin Mode", use_container_width=True):
                    if admin_pass == "admin123":
                        st.session_state.admin_mode = True
                        st.session_state.current_user = self.kms.users["admin-001"]
                        st.session_state.authenticated = True
                        st.success("Admin mode activated")
                        st.rerun()
                    else:
                        st.error("Invalid admin password")
    
    def registration_page(self):
        """Render registration page"""
        st.markdown("<h1 class='main-header'>👤 User Registration</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.container():
                st.markdown("### Create New Account")
                
                username = st.text_input("Choose Username")
                email = st.text_input("Email Address")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Register", use_container_width=True):
                        if not username or not email or not password:
                            st.error("All fields are required")
                        elif password != confirm_password:
                            st.error("Passwords do not match")
                        elif len(password) < 6:
                            st.error("Password must be at least 6 characters")
                        else:
                            success, message = self.kms.register_user(username, email, password)
                            if success:
                                st.success(message)
                                st.session_state.show_registration = False
                                st.rerun()
                            else:
                                st.error(message)
                
                with col_btn2:
                    if st.button("Back to Login", use_container_width=True):
                        st.session_state.show_registration = False
                        st.rerun()
    
    def main_dashboard(self):
        """Render main dashboard"""
        # Sidebar
        with st.sidebar:
            st.markdown(f"### 👤 Welcome, {st.session_state.current_user.username}")
            st.markdown(f"**Role:** {st.session_state.current_user.role}")
            st.markdown(f"**User ID:** {st.session_state.current_user.user_id}")
            
            st.divider()
            
            # Navigation
            st.markdown("### Navigation")
            page = st.radio(
                "Select Page",
                ["Dashboard", "Key Management", "File Encryption", "File Sharing", "Node Monitoring", "AI Analytics", "User Management", "Audit Log", "System Settings"],
                key="nav"
            )
            
            st.divider()
            
            # System Stats
            stats = self.kms.get_system_stats()
            st.markdown("### System Status")
            st.metric("Online Nodes", f"{stats['online_nodes']}/{stats['total_nodes']}")
            st.metric("Active Keys", stats['active_keys'])
            st.metric("Total Users", stats['total_users'])
            
            st.divider()
            
            if st.button("Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.session_state.admin_mode = False
                st.rerun()
        
        # Main content based on selected page
        if page == "Dashboard":
            self.render_dashboard()
        elif page == "Key Management":
            self.render_key_management()
        elif page == "File Encryption":
            self.render_file_encryption()
        elif page == "File Sharing":
            self.render_file_sharing()
        elif page == "Node Monitoring":
            self.render_node_monitoring()
        elif page == "AI Analytics":
            self.render_ai_analytics()
        elif page == "User Management":
            self.render_user_management()
        elif page == "Audit Log":
            self.render_audit_log()
        elif page == "System Settings":
            self.render_system_settings()
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.markdown("<h1 class='main-header'>📊 System Dashboard</h1>", unsafe_allow_html=True)
        
        # System Overview
        col1, col2, col3, col4 = st.columns(4)
        stats = self.kms.get_system_stats()
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Keys", stats['total_keys'])
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Online Nodes", f"{stats['online_nodes']}/{stats['total_nodes']}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Users", stats['total_users'])
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='ai-card'>", unsafe_allow_html=True)
            st.metric("AI Trust Score", f"{np.mean([n.trust_score for n in self.kms.nodes.values()]):.1f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.divider()
        
        # Recent Activity
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Recent Key Activity")
            recent_keys = sorted(
                self.kms.key_metadata.values(),
                key=lambda x: x.last_accessed,
                reverse=True
            )[:5]
            
            for key in recent_keys:
                with st.container():
                    col_key, col_info = st.columns([1, 3])
                    with col_key:
                        st.markdown(f"**{key.key_id}**")
                    with col_info:
                        st.markdown(f"Type: {key.key_type} | Fragments: {key.fragment_count}/{key.threshold}")
                        st.progress(min(key.access_count / 10, 1.0))
                        st.caption(f"Last accessed: {key.last_accessed.strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            st.markdown("### Quick Actions")
            if st.button("Create New Key", use_container_width=True):
                st.switch_page = "Key Management"
            
            if st.button("Encrypt File", use_container_width=True):
                st.session_state.show_file_upload = True
            
            if st.button("View My Files", use_container_width=True):
                st.switch_page = "File Sharing"
        
        st.divider()
        
        # Node Status Grid
        st.markdown("### Node Status Overview")
        nodes = list(self.kms.nodes.values())
        cols = st.columns(4)
        
        for idx, node in enumerate(nodes):
            with cols[idx % 4]:
                status_class = "node-status-online" if node.is_online else "node-status-offline"
                if node.ai_anomaly_score > 0.7:
                    status_class = "node-status-suspicious"
                
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**{node.name}**")
                st.markdown(f"<span class='{status_class}'>● {'Online' if node.is_online else 'Offline'}</span>", unsafe_allow_html=True)
                st.markdown(f"Location: {node.location}")
                st.markdown(f"Fragments: {len(node.fragments_stored)}")
                st.markdown(f"Trust: {node.trust_score:.1f}")
                st.markdown("</div>", unsafe_allow_html=True)
    
    def render_key_management(self):
        """Render key management interface"""
        st.markdown("<h1 class='main-header'>🔑 Key Management</h1>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Create Key", "View Keys", "Reconstruct Key", "Key Rotation"])
        
        with tab1:
            st.markdown("### Create New Encryption Key")
            
            col1, col2 = st.columns(2)
            with col1:
                key_type = st.selectbox("Key Type", ["AES-256", "AES-192", "AES-128", "RSA-2048", "RSA-4096"])
                key_size = st.selectbox("Key Size", [256, 192, 128, 2048, 4096])
            
            with col2:
                fragments = st.slider("Number of Fragments", min_value=2, max_value=10, value=5)
                threshold = st.slider("Threshold (k)", min_value=2, max_value=fragments, value=3)
            
            algorithm = st.selectbox("Algorithm", ["AES", "RSA", "ECC"])
            
            if st.button("Create Key", type="primary", use_container_width=True):
                success, message = self.kms.create_key(
                    key_type=key_type,
                    key_size=key_size,
                    fragments=fragments,
                    threshold=threshold,
                    owner_id=st.session_state.current_user.user_id,
                    algorithm=algorithm
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        with tab2:
            st.markdown("### Available Keys")
            
            # Filter keys
            user_keys = []
            for key_id, metadata in self.kms.key_metadata.items():
                if (st.session_state.current_user.user_id == metadata.owner_id or 
                    st.session_state.current_user.role == "admin" or
                    key_id in st.session_state.current_user.key_access):
                    user_keys.append((key_id, metadata))
            
            if not user_keys:
                st.info("No keys available")
            else:
                for key_id, metadata in user_keys:
                    with st.expander(f"{key_id} - {metadata.key_type}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Owner:** {self.kms.users.get(metadata.owner_id, User('','','','','')).username}")
                            st.markdown(f"**Created:** {metadata.creation_date.strftime('%Y-%m-%d')}")
                            st.markdown(f"**Expires:** {metadata.expiry_date.strftime('%Y-%m-%d')}")
                        
                        with col2:
                            st.markdown(f"**Fragments:** {metadata.fragment_count}")
                            st.markdown(f"**Threshold:** {metadata.threshold}")
                            st.markdown(f"**Status:** {'🟢 Active' if metadata.is_active else '🔴 Inactive'}")
                        
                        # Show fragments
                        fragments = self.kms.fragments.get(key_id, [])
                        if fragments:
                            st.markdown("**Fragments Distribution:**")
                            frag_cols = st.columns(4)
                            for i, frag in enumerate(fragments[:4]):
                                with frag_cols[i % 4]:
                                    node_status = "🟢" if self.kms.nodes.get(frag.node_id, Node('','','','','')).is_online else "🔴"
                                    st.markdown(f"{node_status} Node {frag.node_id[-3:]}")
        
        with tab3:
            st.markdown("### Reconstruct Key")
            
            # Get accessible keys
            accessible_keys = []
            for key_id, metadata in self.kms.key_metadata.items():
                if (st.session_state.current_user.user_id == metadata.owner_id or 
                    st.session_state.current_user.role == "admin" or
                    key_id in st.session_state.current_user.key_access):
                    accessible_keys.append(key_id)
            
            if not accessible_keys:
                st.warning("No keys accessible for reconstruction")
            else:
                selected_key = st.selectbox("Select Key", accessible_keys)
                
                if selected_key:
                    metadata = self.kms.key_metadata[selected_key]
                    fragments = self.kms.fragments.get(selected_key, [])
                    
                    st.markdown(f"**Key Info:** {metadata.key_type} | Threshold: {metadata.threshold}/{metadata.fragment_count}")
                    
                    # Show fragment status
                    available_frags = len([f for f in fragments if f.node_id not in self.kms.failed_nodes])
                    st.markdown(f"**Available Fragments:** {available_frags}/{len(fragments)}")
                    
                    if available_frags >= metadata.threshold:
                        st.success(f"✓ Sufficient fragments available ({available_frags}/{metadata.threshold} required)")
                    else:
                        st.error(f"✗ Insufficient fragments ({available_frags}/{metadata.threshold} required)")
                    
                    if st.button("Reconstruct Key", type="primary", use_container_width=True):
                        success, key_data, message = self.kms.reconstruct_key(
                            selected_key, 
                            st.session_state.current_user.user_id
                        )
                        if success:
                            st.success(message)
                            st.markdown("**Key reconstructed successfully!**")
                            st.code(base64.b64encode(key_data).decode()[:50] + "...", language="text")
                        else:
                            st.error(message)
        
        with tab4:
            st.markdown("### Key Rotation")
            
            if st.session_state.current_user.role != "admin":
                st.warning("Only administrators can rotate keys")
            else:
                rotatable_keys = list(self.kms.key_metadata.keys())
                if not rotatable_keys:
                    st.info("No keys available for rotation")
                else:
                    key_to_rotate = st.selectbox("Select Key to Rotate", rotatable_keys)
                    
                    if key_to_rotate:
                        metadata = self.kms.key_metadata[key_to_rotate]
                        st.markdown(f"**Key:** {key_to_rotate}")
                        st.markdown(f"**Type:** {metadata.key_type}")
                        st.markdown(f"**Current fragments:** {metadata.fragment_count}")
                        st.markdown(f"**Last rotated:** {metadata.creation_date.strftime('%Y-%m-%d %H:%M')}")
                        
                        if st.button("Rotate Key", type="primary", use_container_width=True):
                            with st.spinner("Rotating key..."):
                                success, message = self.kms.rotate_key(
                                    key_to_rotate, 
                                    st.session_state.current_user.user_id
                                )
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
    
    def render_file_encryption(self):
        """Render file encryption interface"""
        st.markdown("<h1 class='main-header'>📁 File Encryption</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Encrypt and Share File")
            
            uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'png', 'jpg', 'docx', 'xlsx'])
            
            if uploaded_file:
                # Get file details
                file_bytes = uploaded_file.getvalue()
                file_name = uploaded_file.name
                
                st.markdown(f"**File:** {file_name}")
                st.markdown(f"**Size:** {len(file_bytes):,} bytes")
                
                # Select receiver
                users = list(self.kms.users.values())
                user_options = {u.username: u.user_id for u in users if u.user_id != st.session_state.current_user.user_id}
                
                receiver_username = st.selectbox("Share with user:", list(user_options.keys()))
                description = st.text_input("Description (optional)")
                
                col_enc1, col_enc2 = st.columns(2)
                with col_enc1:
                    if st.button("Encrypt & Share", type="primary", use_container_width=True):
                        if not receiver_username:
                            st.error("Please select a receiver")
                        else:
                            with st.spinner("Encrypting and sharing file..."):
                                receiver_id = user_options[receiver_username]
                                success, message, file_id = self.kms.encrypt_file_for_user(
                                    file_bytes, file_name, 
                                    st.session_state.current_user.user_id,
                                    receiver_id, description
                                )
                                
                                if success:
                                    st.success(message)
                                    st.balloons()
                                else:
                                    st.error(message)
                
                with col_enc2:
                    if st.button("Encrypt Only", use_container_width=True):
                        st.info("Coming soon: Local encryption without sharing")
        
        with col2:
            st.markdown("### My Encryption Keys")
            
            user_keys = st.session_state.current_user.encryption_keys
            if not user_keys:
                st.info("No encryption keys stored")
            else:
                for key_id, key_data in list(user_keys.items())[:3]:
                    st.markdown(f"**{key_id[:8]}...**")
                    st.caption(f"{len(key_data)} bytes")
    
    def render_file_sharing(self):
        """Render file sharing interface"""
        st.markdown("<h1 class='main-header'>📤 File Sharing</h1>", unsafe_allow_html=True)
        
        # Get user files
        user_files = self.kms.get_user_files(st.session_state.current_user.user_id)
        
        tab1, tab2 = st.tabs(["📨 Received Files", "📤 Sent Files"])
        
        with tab1:
            st.markdown("### Files Shared With You")
            
            received_files = user_files.get("received", [])
            if not received_files:
                st.info("No files received yet")
            else:
                for file in received_files:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.markdown(f"**{file['filename']}**")
                            st.caption(f"From: {file['sender']} | {file['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                            if file['description']:
                                st.caption(f"Description: {file['description']}")
                        
                        with col2:
                            size_mb = file['size'] / 1024 / 1024
                            st.markdown(f"Size: {size_mb:.2f} MB")
                            status = "✅ Downloaded" if file['is_downloaded'] else "📥 Pending"
                            st.markdown(f"Status: {status}")
                        
                        with col3:
                            if st.button("Download", key=f"dl_{file['file_id']}"):
                                success, data, message, filename = self.kms.decrypt_file(
                                    file['file_id'], 
                                    st.session_state.current_user.user_id
                                )
                                if success:
                                    st.download_button(
                                        label="Save File",
                                        data=data,
                                        file_name=filename,
                                        mime="application/octet-stream",
                                        key=f"save_{file['file_id']}"
                                    )
                                    st.success("File ready for download!")
                                else:
                                    st.error(message)
                        
                        st.divider()
        
        with tab2:
            st.markdown("### Files You've Shared")
            
            sent_files = user_files.get("sent", [])
            if not sent_files:
                st.info("No files sent yet")
            else:
                for file in sent_files:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{file['filename']}**")
                            st.caption(f"To: {file['receiver']} | {file['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                            if file['description']:
                                st.caption(f"Description: {file['description']}")
                        
                        with col2:
                            size_mb = file['size'] / 1024 / 1024
                            st.markdown(f"{size_mb:.2f} MB")
                        
                        st.divider()
    
    def render_node_monitoring(self):
        """Render node monitoring interface"""
        st.markdown("<h1 class='main-header'>🖥️ Node Monitoring</h1>", unsafe_allow_html=True)
        
        # Node status table
        st.markdown("### Node Status")
        
        nodes_data = []
        for node in self.kms.nodes.values():
            nodes_data.append({
                "Node ID": node.node_id,
                "Name": node.name,
                "Location": node.location,
                "Status": "🟢 Online" if node.is_online else "🔴 Offline",
                "Trust Score": f"{node.trust_score:.1f}",
                "Fragments": len(node.fragments_stored),
                "Storage Used": f"{node.storage_used / node.storage_capacity * 100:.1f}%",
                "AI Anomaly": f"{node.ai_anomaly_score:.2f}"
            })
        
        df_nodes = pd.DataFrame(nodes_data)
        st.dataframe(df_nodes, use_container_width=True, hide_index=True)
        
        # Node control (admin only)
        if st.session_state.current_user.role == "admin":
            st.markdown("### Node Control Panel")
            
            col1, col2 = st.columns(2)
            with col1:
                node_to_fail = st.selectbox("Select node to simulate failure", 
                                          list(self.kms.nodes.keys()))
                if st.button("Simulate Failure", use_container_width=True):
                    self.kms.simulate_node_failure(node_to_fail)
                    st.success(f"Node {node_to_fail} marked as failed")
                    st.rerun()
            
            with col2:
                node_to_restore = st.selectbox("Select node to restore", 
                                             [n for n in self.kms.nodes.keys() if not self.kms.nodes[n].is_online])
                if st.button("Restore Node", use_container_width=True):
                    self.kms.restore_node(node_to_restore)
                    st.success(f"Node {node_to_restore} restored")
                    st.rerun()
        
        # Node health visualization
        st.markdown("### Node Health Visualization")
        
        fig = go.Figure(data=[
            go.Bar(
                name='Trust Score',
                x=[n.name for n in self.kms.nodes.values()],
                y=[n.trust_score for n in self.kms.nodes.values()],
                marker_color=['green' if t > 70 else 'orange' if t > 50 else 'red' 
                             for t in [n.trust_score for n in self.kms.nodes.values()]]
            )
        ])
        
        fig.update_layout(
            title="Node Trust Scores",
            yaxis_title="Trust Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_ai_analytics(self):
        """Render AI analytics dashboard"""
        st.markdown("<h1 class='main-header'>🧠 AI Analytics</h1>", unsafe_allow_html=True)
        
        # Trust score distribution
        st.markdown("### Node Trust Analysis")
        
        trust_scores = [node.trust_score for node in self.kms.nodes.values()]
        avg_trust = np.mean(trust_scores) if trust_scores else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Trust Score", f"{avg_trust:.1f}")
        with col2:
            high_trust = sum(1 for s in trust_scores if s >= 80)
            st.metric("High Trust Nodes", high_trust)
        with col3:
            anomalies = len(self.kms.ai_monitor.detect_anomalies())
            st.metric("Anomalies Detected", anomalies)
        
        # Trust score visualization
        fig = px.box(
            x=trust_scores,
            title="Trust Score Distribution",
            labels={"x": "Trust Score"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly detection results
        st.markdown("### Anomaly Detection")
        
        anomalies = self.kms.ai_monitor.detect_anomalies()
        if anomalies:
            st.warning(f"⚠️ {len(anomalies)} nodes with anomalous behavior detected")
            
            for node_id, details in anomalies.items():
                with st.expander(f"Node {node_id} - {details['severity'].upper()} severity"):
                    st.markdown(f"**Anomaly Score:** {details['score']:.3f}")
                    st.markdown(f"**Trust Score:** {details['trust_score']:.1f}")
                    
                    # Get behavior analytics
                    analytics = self.kms.ai_monitor.get_behavior_analytics(node_id)
                    if analytics:
                        st.markdown("**Recent Activity:**")
                        for activity in analytics['recent_activity']:
                            st.caption(f"{activity['time']} - {activity['type']}: {activity['value']:.2f}")
        else:
            st.success("✅ No anomalies detected")
        
        # AI recommendations
        st.markdown("### AI Recommendations")
        
        recommended_nodes = self.kms.ai_monitor.get_recommended_nodes(3)
        if recommended_nodes:
            st.info("**Top 3 Recommended Nodes for Fragment Storage:**")
            for i, node_id in enumerate(recommended_nodes, 1):
                node = self.kms.nodes.get(node_id)
                if node:
                    st.markdown(f"{i}. **{node.name}** - Trust: {node.trust_score:.1f}, Location: {node.location}")
        else:
            st.info("No recommendations available")
    
    def render_user_management(self):
        """Render user management interface (admin only)"""
        if st.session_state.current_user.role != "admin":
            st.warning("⛔ Access denied. Admin privileges required.")
            return
        
        st.markdown("<h1 class='main-header'>👥 User Management</h1>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["All Users", "Create User", "User Activity"])
        
        with tab1:
            st.markdown("### Registered Users")
            
            users_data = []
            for user in self.kms.users.values():
                users_data.append({
                    "User ID": user.user_id,
                    "Username": user.username,
                    "Email": user.email,
                    "Role": user.role,
                    "Status": "Active" if user.is_active else "Inactive",
                    "Created": user.created_at.strftime('%Y-%m-%d'),
                    "Last Login": user.last_login.strftime('%Y-%m-%d %H:%M'),
                    "Files Sent": len(user.sent_files),
                    "Files Received": len(user.received_files)
                })
            
            df_users = pd.DataFrame(users_data)
            st.dataframe(df_users, use_container_width=True, hide_index=True)
        
        with tab2:
            st.markdown("### Create New User (Admin)")
            
            with st.form("create_user_form"):
                col1, col2 = st.columns(2)
                with col1:
                    new_username = st.text_input("Username")
                    new_email = st.text_input("Email")
                
                with col2:
                    new_password = st.text_input("Password", type="password")
                    new_role = st.selectbox("Role", ["user", "admin"])
                
                submitted = st.form_submit_button("Create User", use_container_width=True)
                if submitted:
                    if not all([new_username, new_email, new_password]):
                        st.error("All fields are required")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        success, message = self.kms.register_user(
                            new_username, new_email, new_password, new_role
                        )
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        
        with tab3:
            st.markdown("### User Activity Log")
            
            # Get recent file transfers
            recent_transfers = []
            for file in self.kms.encrypted_files.values():
                sender = self.kms.users.get(file.sender_id, User('','','','','')).username
                receiver = self.kms.users.get(file.receiver_id, User('','','','','')).username
                
                recent_transfers.append({
                    "Timestamp": file.timestamp.strftime('%Y-%m-%d %H:%M'),
                    "File": file.filename,
                    "Sender": sender,
                    "Receiver": receiver,
                    "Status": "Downloaded" if file.is_downloaded else "Pending"
                })
            
            if recent_transfers:
                df_transfers = pd.DataFrame(recent_transfers)
                st.dataframe(
                    df_transfers.sort_values("Timestamp", ascending=False).head(10),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No file transfers recorded")
    
    def render_audit_log(self):
        """Render audit log interface"""
        st.markdown("<h1 class='main-header'>📋 Audit Log</h1>", unsafe_allow_html=True)
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            event_filter = st.multiselect(
                "Filter by Event Type",
                options=list(set([e['event_type'] for e in self.kms.audit_log])),
                default=[]
            )
        
        with col2:
            user_filter = st.multiselect(
                "Filter by User",
                options=list(set([e['user_id'] for e in self.kms.audit_log])),
                default=[]
            )
        
        with col3:
            date_range = st.date_input(
                "Date Range",
                value=(
                    datetime.now().date() - timedelta(days=7),
                    datetime.now().date()
                )
            )
        
        # Apply filters
        filtered_logs = self.kms.audit_log
        
        if event_filter:
            filtered_logs = [e for e in filtered_logs if e['event_type'] in event_filter]
        
        if user_filter:
            filtered_logs = [e for e in filtered_logs if e['user_id'] in user_filter]
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_logs = [
                e for e in filtered_logs
                if start_date <= datetime.fromisoformat(e['timestamp']).date() <= end_date
            ]
        
        # Display audit log
        st.markdown(f"**Total Events:** {len(filtered_logs)}")
        
        for event in filtered_logs[-50:][::-1]:  # Show last 50 events, most recent first
            with st.expander(f"{event['timestamp']} - {event['event_type']}"):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**Event ID:** {event['event_id'][:8]}...")
                    st.markdown(f"**User:** {event['user_id']}")
                
                with col2:
                    st.markdown("**Details:**")
                    st.json(event['details'])
    
    def render_system_settings(self):
        """Render system settings (admin only)"""
        if st.session_state.current_user.role != "admin":
            st.warning("⛔ Access denied. Admin privileges required.")
            return
        
        st.markdown("<h1 class='main-header'>⚙️ System Settings</h1>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["General", "Security", "AI Configuration"])
        
        with tab1:
            st.markdown("### General Settings")
            
            # System info
            stats = self.kms.get_system_stats()
            st.markdown("**System Information:**")
            st.markdown(f"- Total Keys: {stats['total_keys']}")
            st.markdown(f"- Total Users: {stats['total_users']}")
            st.markdown(f"- Total Files: {stats['total_files']}")
            st.markdown(f"- Audit Events: {stats['audit_events']}")
            
            st.divider()
            
            # System actions
            st.markdown("**System Actions:**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Refresh System Data", use_container_width=True):
                    st.rerun()
                
                if st.button("Export Audit Log", use_container_width=True):
                    # Create downloadable audit log
                    audit_data = json.dumps(self.kms.audit_log[-100:], indent=2)
                    st.download_button(
                        label="Download Audit Log",
                        data=audit_data,
                        file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("Clear AI Data", use_container_width=True):
                    st.warning("This will reset AI training data. Continue?")
                    if st.button("Confirm Reset", type="secondary"):
                        self.kms.ai_monitor = AIMonitor()
                        st.success("AI data cleared")
                        st.rerun()
        
        with tab2:
            st.markdown("### Security Settings")
            
            # Password policy
            st.markdown("**Password Policy:**")
            min_length = st.slider("Minimum Password Length", 6, 20, 8)
            require_special = st.checkbox("Require Special Characters", True)
            require_numbers = st.checkbox("Require Numbers", True)
            
            if st.button("Update Password Policy", use_container_width=True):
                st.success("Password policy updated")
            
            st.divider()
            
            # Session settings
            st.markdown("**Session Settings:**")
            session_timeout = st.slider("Session Timeout (minutes)", 15, 240, 30)
            max_login_attempts = st.slider("Max Login Attempts", 3, 10, 5)
            
            if st.button("Update Session Settings", use_container_width=True):
                st.success("Session settings updated")
        
        with tab3:
            st.markdown("### AI Configuration")
            
            # AI parameters
            st.markdown("**Anomaly Detection Parameters:**")
            
            col1, col2 = st.columns(2)
            with col1:
                contamination = st.slider("Contamination Rate", 0.05, 0.3, 0.15, 0.05)
                n_estimators = st.slider("Number of Estimators", 50, 200, 100, 10)
            
            with col2:
                anomaly_threshold = st.slider("Anomaly Threshold", 0.1, 1.0, 0.7, 0.05)
                metrics_window = st.slider("Metrics Window Size", 50, 200, 100, 10)
            
            if st.button("Update AI Parameters", use_container_width=True):
                self.kms.ai_monitor.anomaly_detector = IsolationForest(
                    contamination=contamination,
                    n_estimators=n_estimators,
                    random_state=42
                )
                self.kms.ai_monitor.anomaly_threshold = anomaly_threshold
                self.kms.ai_monitor.metrics_window = metrics_window
                st.success("AI parameters updated")
    
    def run(self):
        """Main application runner"""
        if not st.session_state.authenticated:
            if st.session_state.show_registration:
                self.registration_page()
            else:
                self.login_page()
        else:
            self.main_dashboard()

# ============================================
# MAIN APPLICATION
# ============================================

if __name__ == "__main__":
    # Initialize and run the application
    app = KMSApp()
    app.run()