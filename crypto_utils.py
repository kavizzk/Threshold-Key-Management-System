import json
import base64
import secrets
import hashlib
import hmac
from typing import List, Tuple
import numpy as np

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


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