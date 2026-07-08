import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List
import networkx as nx

from models import Node, KeyMetadata, KeyFragment, User, EncryptedFile
from crypto_utils import CryptoUtils
from ai_monitor import AIMonitor


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
            key_bytes = CryptoUtils.generate_symmetric_key(config['size'] // 8)
            
            # Distribute to AI-selected nodes
            selected_nodes = self.ai_monitor.get_recommended_nodes(config['fragments'])
            
            # Create metadata
            metadata = KeyMetadata(
                key_id=key_id,
                key_type=config['type'],
                key_size=config['size'],
                owner_id="admin-001",
                fragment_count=config['fragments'],
                threshold=config['threshold'],
                creation_date=datetime.now()
            )
            
            self.key_metadata[key_id] = metadata
            self.fragments[key_id] = []
            
            # Distribute fragments
            for j, node_id in enumerate(selected_nodes):
                fragment_id = f"frag-{key_id}-{j+1:03d}"
                fragment = KeyFragment(
                    fragment_id=fragment_id,
                    key_id=key_id,
                    data=key_bytes,  # Simplified - in real implementation, this would be the actual share
                    node_id=node_id,
                    timestamp=datetime.now(),
                    share_index=j
                )
                
                self.fragments[key_id].append(fragment)
                if node_id in self.nodes:
                    self.nodes[node_id].fragments_stored.append(fragment_id)
    
    def _create_sample_files(self):
        """Create sample encrypted files"""
        sample_files = [
            ("report.pdf", "alice", "bob"),
            ("contract.docx", "bob", "charlie"),
            ("presentation.pptx", "charlie", "alice"),
        ]
        
        for i, (filename, sender_username, receiver_username) in enumerate(sample_files, 1):
            file_id = f"file-{i:03d}"
            
            sender = next((u for u in self.users.values() if u.username == sender_username), None)
            receiver = next((u for u in self.users.values() if u.username == receiver_username), None)
            
            if sender and receiver:
                # Create a sample key for encryption
                key = CryptoUtils.generate_symmetric_key(32)
                key_id = f"filekey-{file_id}"
                
                # Create encrypted file (simulated)
                encrypted_data = CryptoUtils.encrypt_with_aes(b"Sample encrypted content", key)
                
                file = EncryptedFile(
                    file_id=file_id,
                    filename=filename,
                    encrypted_data=encrypted_data,
                    key_id=key_id,
                    sender_id=sender.user_id,
                    receiver_id=receiver.user_id,
                    timestamp=datetime.now()
                )
                
                self.encrypted_files[file_id] = file
                
                # Store key in user's key storage
                sender.encryption_keys[key_id] = key
                sender.sent_files.append(file_id)
                receiver.received_files.append(file_id)
    
    def authenticate_user(self, username: str, password: str) -> User:
        """Authenticate user"""
        for user in self.users.values():
            if user.username == username and CryptoUtils.verify_password(password, user.password_hash):
                user.last_login = datetime.now()
                return user
        return None
    
    def add_node(self, name: str, ip_address: str, location: str = "Unknown") -> Node:
        """Add a new node to the system"""
        node_id = f"node-{len(self.nodes) + 1:03d}"
        _, public_key = CryptoUtils.generate_key_pair()
        
        node = Node(
            node_id=node_id,
            name=name,
            ip_address=ip_address,
            public_key=public_key,
            location=location,
            node_type="KMN"
        )
        
        self.nodes[node_id] = node
        self.ai_monitor.initialize_node_history(node_id)
        self.network_graph.add_node(node_id, **node.to_dict())
        
        # Connect to existing nodes
        for existing_node in self.nodes:
            if existing_node != node_id:
                self.network_graph.add_edge(node_id, existing_node, weight=0.5)
        
        self._log_audit("node_added", f"Added new node: {name} ({node_id})")
        return node
    
    def create_key(self, key_type: str, key_size: int, fragments: int, 
                   threshold: int, owner_id: str) -> KeyMetadata:
        """Create a new key with AI-optimized distribution"""
        key_id = f"key-{str(uuid.uuid4())[:8]}"
        key_bytes = CryptoUtils.generate_symmetric_key(key_size // 8)
        
        # Get AI-recommended nodes
        recommended_nodes = self.ai_monitor.get_recommended_nodes(fragments)
        
        # Use Shamir's Secret Sharing
        shares = CryptoUtils.shamir_secret_sharing(key_bytes, fragments, threshold)
        
        metadata = KeyMetadata(
            key_id=key_id,
            key_type=key_type,
            key_size=key_size,
            owner_id=owner_id,
            fragment_count=fragments,
            threshold=threshold,
            creation_date=datetime.now()
        )
        
        self.key_metadata[key_id] = metadata
        self.fragments[key_id] = []
        
        # Distribute fragments to recommended nodes
        for i, (share_index, share_data) in enumerate(shares[:fragments]):
            node_id = recommended_nodes[i % len(recommended_nodes)]
            fragment_id = f"frag-{key_id}-{i+1:03d}"
            
            fragment = KeyFragment(
                fragment_id=fragment_id,
                key_id=key_id,
                data=share_data,
                node_id=node_id,
                timestamp=datetime.now(),
                share_index=share_index
            )
            
            self.fragments[key_id].append(fragment)
            if node_id in self.nodes:
                self.nodes[node_id].fragments_stored.append(fragment_id)
        
        self._log_audit("key_created", f"Created new key: {key_id} with {fragments} fragments")
        return metadata
    
    def reconstruct_key(self, key_id: str, requesting_user_id: str) -> bytes:
        """Reconstruct key from fragments"""
        if key_id not in self.fragments:
            raise ValueError(f"Key {key_id} not found")
        
        metadata = self.key_metadata.get(key_id)
        if not metadata:
            raise ValueError(f"Metadata for key {key_id} not found")
        
        # Check if user has access
        if requesting_user_id != metadata.owner_id and "admin" not in self.users[requesting_user_id].role:
            raise PermissionError(f"User {requesting_user_id} does not have access to key {key_id}")
        
        fragments = self.fragments[key_id]
        
        # Get fragments from online and trusted nodes
        available_fragments = []
        for fragment in fragments:
            node = self.nodes.get(fragment.node_id)
            if node and node.is_online and node.is_trusted:
                available_fragments.append((fragment.share_index, fragment.data))
        
        if len(available_fragments) < metadata.threshold:
            raise ValueError(f"Insufficient fragments. Need {metadata.threshold}, have