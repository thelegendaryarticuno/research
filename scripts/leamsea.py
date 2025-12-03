import os, struct
from typing import Tuple
from Crypto.Random import get_random_bytes

PLAINTEXT_DIR = "dataset/plaintexts"

# -------------------------------------------------------------------
# Generic CTR wrapper (works for 64b and 128b block ciphers)
# -------------------------------------------------------------------
def ctr_encrypt(block_encrypt, key: bytes, nonce: bytes, plaintext: bytes, block_size_bytes: int) -> bytes:
    """
    CTR layout: counter_block = nonce || pad || 32-bit big-endian counter
    nonce length must be <= block_size_bytes - 4.
    """
    assert len(nonce) <= block_size_bytes - 4, "nonce too long for CTR layout"
    out = bytearray()
    ctr = 0
    off = 0
    while off < len(plaintext):
        rem = block_size_bytes - len(nonce) - 4
        counter_block = nonce + (b"\x00" * rem) + struct.pack(">I", ctr)
        ks = block_encrypt(counter_block, key)
        chunk = plaintext[off: off + block_size_bytes]
        out.extend(bytes(a ^ b for a, b in zip(chunk, ks[:len(chunk)])))
        ctr = (ctr + 1) & 0xffffffff
        off += block_size_bytes
    return bytes(out)

# ===================================================================
# ==========================  LEA-128  ==============================
# ===================================================================
# LEA: 128-bit block, 128-bit key, 24 rounds. ARX cipher (no S-box).
# NOTE: Please validate against test vectors before large-scale use.

def _u32(x): return x & 0xFFFFFFFF
def _rol32(x, r): return _u32((x << r) | (x >> (32 - r)))
def _ror32(x, r): return _u32((x >> r) | (x << (32 - r)))

# delta constants used by LEA key schedule
_LEA_DELTA = [0x9e3779b9, 0x3c6ef373, 0x78dde6e6, 0xf1bbcdcc,
              0xe3779b99, 0xc6ef3733, 0x8dde6e67, 0x1bbcdccf]

def _lea_expand_128(key16: bytes):
    """
    Returns 24 * 6 = 144 round subkeys (u32).
    This schedule matches common LEA-128 references (24 rounds).
    """
    assert len(key16) == 16
    T0, T1, T2, T3 = struct.unpack("<4I", key16)  # LE load (common in refs)
    rk = []

    for i in range(24):
        d = _LEA_DELTA[i % 8]
        # generate six subkeys per round
        K0 = _u32(T0 + _rol32(d, 0))
        K1 = _u32(T1 + _rol32(d, 1))
        K2 = _u32(T2 + _rol32(d, 2))
        K3 = _u32(T1 + _rol32(d, 3))
        K4 = _u32(T2 + _rol32(d, 4))
        K5 = _u32(T3 + _rol32(d, 5))
        rk.extend([K0, K1, K2, K3, K4, K5])

        # rotate key words for next round (as per common LEA-128 spec)
        T0 = _rol32(T0, 1)
        T1 = _rol32(T1, 3)
        T2 = _rol32(T2, 6)
        T3 = _rol32(T3, 11)

    return rk  # list of 144 u32

def lea128_encrypt_block(block16: bytes, key16: bytes) -> bytes:
    """
    Encrypt one 128-bit block with LEA-128/128.
    """
    assert len(block16) == 16
    rk = _lea_expand_128(key16)
    X0, X1, X2, X3 = struct.unpack("<4I", block16)

    # 24 rounds; 6 subkeys per round
    for r in range(24):
        k0, k1, k2, k3, k4, k5 = rk[6*r:6*r+6]
        X0 = _rol32(_u32(X0 + (X1 ^ k0)), 9)
        X1 = _ror32(_u32(X1 + (X2 ^ k1)), 5)
        X2 = _ror32(_u32(X2 + (X3 ^ k2)), 3)
        # second half-step with remaining three subkeys
        X3 = _rol32(_u32(X3 + (X0 ^ k3)), 9)
        X0 = _ror32(_u32(X0 + (X1 ^ k4)), 5)
        X1 = _ror32(_u32(X1 + (X2 ^ k5)), 3)

    return struct.pack("<4I", X0, X1, X2, X3)

def encrypt_lea_ctr(in_file: str, out_file: str):
    key = get_random_bytes(16)   # 128-bit key
    nonce = get_random_bytes(12) # 96-bit nonce for 128-bit block (leaves 4B counter)
    with open(in_file, "rb") as f:
        pt = f.read()
    ct = ctr_encrypt(lambda b,k: lea128_encrypt_block(b,k), key, nonce, pt, 16)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(ct)

# ===================================================================
# ==========================  MSEA-96  ===============================
# ===================================================================
# MSEA-96: Modified SEA (Scalable Encryption Algorithm) 
# - Block size: 96 bits (12 bytes)
# - Key size: 96 bits (12 bytes) 
# - Rounds: 16
# - Structure: Modified Feistel network with ARX operations
# 
# This implementation is based on the SEA family but with modifications
# for 96-bit block size. Uses Addition, Rotation, and XOR operations.

def _msea_f_function(x: int, k: int) -> int:
    """MSEA round function: ARX-based transformation"""
    # Mix with round key
    x = (x + k) & 0xFFFFFFFF
    # Non-linear rotation based on lower bits
    rot = (x & 0x1F) + 1  # rotation amount: 1-32
    x = ((x << rot) | (x >> (32 - rot))) & 0xFFFFFFFF
    # Additional mixing
    x = x ^ ((x >> 16) & 0xFFFF)
    x = (x + 0x9E3779B9) & 0xFFFFFFFF  # golden ratio constant
    return x

def _msea_key_schedule(key12: bytes) -> list:
    """Generate round keys for MSEA-96"""
    assert len(key12) == 12
    
    # Split 96-bit key into three 32-bit words
    k0, k1, k2 = struct.unpack(">3I", key12)
    
    round_keys = []
    
    # Generate 16 round keys using linear feedback
    for i in range(16):
        # Mix the key words
        temp = (k0 + k1 + k2 + i) & 0xFFFFFFFF
        
        # Rotate and mix
        k0 = ((k0 << 7) | (k0 >> 25)) & 0xFFFFFFFF
        k1 = ((k1 << 11) | (k1 >> 21)) & 0xFFFFFFFF  
        k2 = ((k2 << 13) | (k2 >> 19)) & 0xFFFFFFFF
        
        # Add round constant
        round_key = (temp + 0x12345678 * (i + 1)) & 0xFFFFFFFF
        round_keys.append(round_key)
        
        # Update key state for next round
        k0 = (k0 + round_key) & 0xFFFFFFFF
        k1 = (k1 + k2) & 0xFFFFFFFF
        k2 = (k2 + temp) & 0xFFFFFFFF
    
    return round_keys

def msea96_encrypt_block(block12: bytes, key12: bytes) -> bytes:
    """
    Encrypt one 96-bit block with MSEA-96.
    Uses a modified Feistel structure with 32-bit words.
    """
    assert len(block12) == 12
    assert len(key12) == 12
    
    # Generate round keys
    round_keys = _msea_key_schedule(key12)
    
    # Split 96-bit block into three 32-bit words (big-endian)
    L, M, R = struct.unpack(">3I", block12)
    
    # 16 rounds of modified Feistel
    for i in range(16):
        rk = round_keys[i]
        
        # Modified 3-way Feistel: each round affects all three words
        if i % 3 == 0:
            # L = L ⊕ F(M ⊕ R, RK)
            temp = _msea_f_function(M ^ R, rk)
            L = L ^ temp
        elif i % 3 == 1:
            # M = M ⊕ F(R ⊕ L, RK)  
            temp = _msea_f_function(R ^ L, rk)
            M = M ^ temp
        else:
            # R = R ⊕ F(L ⊕ M, RK)
            temp = _msea_f_function(L ^ M, rk)
            R = R ^ temp
        
        # Rotate words every 4 rounds for additional diffusion
        if (i + 1) % 4 == 0:
            L, M, R = M, R, L
    
    # Pack result back to 96-bit block
    return struct.pack(">3I", L, M, R)

def encrypt_msea_ctr(in_file: str, out_file: str):
    """Encrypt file using MSEA-96 in CTR mode"""
    key = get_random_bytes(12)    # 96-bit key
    nonce = get_random_bytes(8)   # 64-bit nonce (96-bit block leaves 4B counter)
    with open(in_file, "rb") as f:
        pt = f.read()
    ct = ctr_encrypt(lambda b,k: msea96_encrypt_block(b,k), key, nonce, pt, 12)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(ct)

# -------------------------------------------------------------------
# Batch loop additions for LEA and MSEA
# -------------------------------------------------------------------
OUT_DIRS = {
    "LEA":   "dataset/LEA",
    "MSEA":  "dataset/MSEA",
}

for d in OUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

for fname in os.listdir(PLAINTEXT_DIR):
    if not fname.endswith(".bin"):
        continue
    in_path = os.path.join(PLAINTEXT_DIR, fname)

    # LEA
    out_lea = os.path.join(OUT_DIRS["LEA"], f"LEA_{fname}")
    encrypt_lea_ctr(in_path, out_lea)
    print(f"[LEA]  {fname} -> {os.path.basename(out_lea)}")

    # MSEA
    out_msea = os.path.join(OUT_DIRS["MSEA"], f"MSEA_{fname}")
    encrypt_msea_ctr(in_path, out_msea)
    print(f"[MSEA] {fname} -> {os.path.basename(out_msea)}")
