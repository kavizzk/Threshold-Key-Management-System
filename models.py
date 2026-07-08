import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque
from collections import deque


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