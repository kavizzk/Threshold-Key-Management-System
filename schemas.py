from datetime import datetime, timedelta
import base64

class KeyFragment:
    def __init__(self, fragment_id, key_id, data, node_id, timestamp, share_index=0):
        self.fragment_id, self.key_id, self.data = fragment_id, key_id, data
        self.node_id, self.timestamp, self.share_index = node_id, timestamp, share_index

class Node:
    def __init__(self, node_id, name, ip_address, public_key, location="Unknown"):
        self.node_id, self.name, self.ip_address = node_id, name, ip_address
        self.public_key, self.location = public_key, location
        self.trust_score = 80.0
        self.is_online = True
        self.fragments_stored = []

class User:
    def __init__(self, user_id, username, email, password_hash, role="user"):
        self.user_id, self.username, self.email = user_id, username, email
        self.password_hash, self.role = password_hash, role
        self.encryption_keys = {}