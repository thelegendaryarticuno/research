#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_generator.py
--------------------
Rebuilds the dataset from plaintexts for multiple ciphers AND collects system metrics
(time, CPU, memory, throughput) per encryption, saving both ciphertexts and a metrics CSV.

Algorithms included:
- AES (CTR, PyCryptodome)
- PRESENT-80 (CTR)
- XTEA (CTR)
- SIMON64/128 (CTR)
- PRINCE (CTR)
- RECTANGLE-80 (CTR)
- ASCON-128 v2 (AEAD; outputs ct||tag)
- LEA-128 (CTR)
- MSEA-96 (CTR)

Input:
  dataset/plaintexts/*.bin    (already generated plaintexts)

Outputs:
  dataset/<ALGO>/ALGO_<name>.bin    (ciphertexts)
  metrics/perf_metrics.csv          (per-run metrics)
  metrics/dataset_manifest.csv      (ciphertext manifest with label & has_tag)

Notes:
- Requires: pycryptodome, psutil
    pip install pycryptodome psutil
"""

import os, sys, time, csv, struct, argparse
from typing import Tuple, Dict, Any, Callable
from dataclasses import dataclass, asdict

# 3rd party
try:
    import psutil
except Exception as e:
    print("psutil is required. Install: pip install psutil", file=sys.stderr)
    raise

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
except Exception as e:
    print("pycryptodome is required. Install: pip install pycryptodome", file=sys.stderr)
    raise

# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def file_size_bytes(path: str) -> int:
    return os.path.getsize(path)

def bytes_to_kb(n: int) -> float:
    return n / 1024.0

def bytes_to_mb(n: int) -> float:
    return n / (1024.0 * 1024.0)

# Generic CTR wrapper (works for 64b, 96b, 128b block ciphers)
def ctr_encrypt(block_encrypt: Callable[[bytes, bytes], bytes],
                key: bytes, nonce: bytes, plaintext: bytes, block_size_bytes: int) -> bytes:
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

# -----------------------------
# AES-128 CTR (PyCryptodome)
# -----------------------------

def encrypt_aes_ctr_bytes(plaintext: bytes) -> bytes:
    key = get_random_bytes(16)   # 128-bit key
    nonce = get_random_bytes(8)  # 64-bit nonce
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    return cipher.encrypt(plaintext)

# -----------------------------
# PRESENT-80 (64-bit block)
# -----------------------------

PRESENT_SBOX = [0xC,5,6,0xB,9,0,0xA,0xD,3,0xE,0xF,8,4,7,1,2]
PRESENT_PBOX = [
    0,16,32,48,1,17,33,49,2,18,34,50,3,19,35,51,
    4,20,36,52,5,21,37,53,6,22,38,54,7,23,39,55,
    8,24,40,56,9,25,41,57,10,26,42,58,11,27,43,59,
    12,28,44,60,13,29,45,61,14,30,46,62,15,31,47,63
]

def _present_sbox_layer(x: int) -> int:
    y = 0
    for i in range(16):
        y |= PRESENT_SBOX[(x >> (i*4)) & 0xF] << (i*4)
    return y

def _present_p_layer(x: int) -> int:
    y = 0
    for i in range(64):
        y |= ((x >> i) & 1) << PRESENT_PBOX[i]
    return y

def _present_key_schedule_80(key_hi: int, key_lo: int):
    for round_counter in range(1, 32+1):
        round_key = ((key_hi << 64) | key_lo) >> 16
        yield round_key & ((1<<64)-1)
        whole = ((key_hi << 64) | key_lo) & ((1<<80)-1)
        whole = ((whole << 61) | (whole >> (80-61))) & ((1<<80)-1)
        key_hi = (whole >> 64) & 0xFFFF
        key_lo = whole & ((1<<64)-1)
        high4 = (key_hi >> 12) & 0xF
        high4 = PRESENT_SBOX[high4]
        key_hi = ((key_hi & 0x0FFF) | (high4 << 12)) & 0xFFFF
        key_lo ^= (round_counter & 0x1F) << 15

def present80_encrypt_block(block8: bytes, key10: bytes) -> bytes:
    assert len(block8) == 8 and len(key10) == 10
    state = int.from_bytes(block8, "big")
    key_hi = int.from_bytes(key10[:2], "big")
    key_lo = int.from_bytes(key10[2:], "big")
    rks = list(_present_key_schedule_80(key_hi, key_lo))
    for i in range(31):
        state ^= rks[i]
        state = _present_sbox_layer(state)
        state = _present_p_layer(state)
    state ^= rks[31]
    return state.to_bytes(8, "big")

def encrypt_present_ctr_bytes(plaintext: bytes) -> bytes:
    key = get_random_bytes(10)   # 80-bit
    nonce = get_random_bytes(4)  # 32-bit nonce (leaves 4B counter) for 64-bit block
    return ctr_encrypt(lambda b,k: present80_encrypt_block(b,k), key, nonce, plaintext, 8)

# -----------------------------
# XTEA (64-bit block)
# -----------------------------
def xtea_encrypt_block(block8: bytes, key16: bytes, rounds: int = 32) -> bytes:
    v0, v1 = struct.unpack(">2I", block8)
    k = struct.unpack(">4I", key16)
    delta = 0x9E3779B9
    s = 0
    for _ in range(rounds):
        v0 = (v0 + (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (s + k[s & 3])) & 0xFFFFFFFF
        s = (s + delta) & 0xFFFFFFFF
        v1 = (v1 + (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (s + k[(s>>11) & 3])) & 0xFFFFFFFF
    return struct.pack(">2I", v0, v1)

def encrypt_xtea_ctr_bytes(plaintext: bytes) -> bytes:
    key = get_random_bytes(16)  # 128-bit
    nonce = get_random_bytes(4) # 32-bit nonce
    return ctr_encrypt(lambda b,k: xtea_encrypt_block(b,k), key, nonce, plaintext, 8)

# -----------------------------
# SIMON64/128 (64-bit block)
# -----------------------------
def _rol32(x, r): return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF
def _ror32(x, r): return ((x >> r) | (x << (32 - r))) & 0xFFFFFFFF
_Z3 = int("11111010001001010110000111001101111101000100101011000011100110", 2)

def _simon64_128_expand_keys(key16: bytes):
    k_words = list(struct.unpack(">4I", key16))
    m = 4
    T = 72
    c = 0xFFFFFFFC
    z = _Z3
    ks = [0]*T
    ks[:m] = k_words[::-1]
    for i in range(m, T):
        tmp = _ror32(ks[i-1], 3)
        tmp ^= ks[i-3]
        tmp ^= _ror32(tmp, 1)
        zbit = (z >> ((i - m) % 62)) & 1
        ks[i] = (c ^ zbit ^ ks[i-m] ^ tmp) & 0xFFFFFFFF
    return ks

def simon64_128_encrypt_block(block8: bytes, key16: bytes) -> bytes:
    x, y = struct.unpack(">2I", block8)
    ks = _simon64_128_expand_keys(key16)
    for k in ks:
        tmp = ((_rol32(x,1) & _rol32(x,8)) ^ _rol32(x,2)) ^ y ^ k
        y = x
        x = tmp & 0xFFFFFFFF
    return struct.pack(">2I", x, y)

def encrypt_simon_ctr_bytes(plaintext: bytes) -> bytes:
    key = get_random_bytes(16)  # 128-bit key
    nonce = get_random_bytes(4) # 32-bit nonce
    return ctr_encrypt(lambda b,k: simon64_128_encrypt_block(b,k), key, nonce, plaintext, 8)

# -----------------------------
# PRINCE (64-bit block, 128-bit key)
# -----------------------------
_PRINCE_S = [0xB,0xF,0x3,0x2,0xA,0xC,0x9,0x1,0x6,0x7,0x8,0x0,0xE,0x5,0xD,0x4]
_PRINCE_S_INV = [0xB,0x7,0x3,0x2,0xF,0xD,0x8,0x9,0xA,0x6,0x4,0x0,0x5,0xE,0xC,0x1]
def _m_prime(x: int) -> int:
    def mix4(n0,n1,n2,n3):
        r0 = n2 ^ n3
        r1 = n0 ^ n3
        r2 = n0 ^ n1
        r3 = n1 ^ n2
        return r0 & 0xF, r1 & 0xF, r2 & 0xF, r3 & 0xF
    ns = [(x >> (4*i)) & 0xF for i in range(16)]
    cols = [
        (ns[0], ns[4], ns[8],  ns[12]),
        (ns[1], ns[5], ns[9],  ns[13]),
        (ns[2], ns[6], ns[10], ns[14]),
        (ns[3], ns[7], ns[11], ns[15]),
    ]
    out_cols = [mix4(*c) for c in cols]
    res = 0
    positions = [(0,4,8,12),(1,5,9,13),(2,6,10,14),(3,7,11,15)]
    for col_idx, (p0,p1,p2,p3) in enumerate(positions):
        for nib, pos in zip(out_cols[col_idx], (p0,p1,p2,p3)):
            res |= (nib & 0xF) << (4*pos)
    return res

_PRINCE_ALPHA = 0xC0AC29B7C97C50DD
_PRINCE_RCS = [
    0x0000000000000000,0x13198A2E03707344,0xA4093822299F31D0,0x082EFA98EC4E6C89,
    0x452821E638D01377,0xBE5466CF34E90C6C,0x7EF84F78FD955CB1,0x85840851F1AC43AA,
    0xC882D32F25323C54,0x64A51195E0E3610D,0xD3B5A399CA0C2399,0xC0AC29B7C97C50DD
]

def _rotr64(x,n): return ((x>>n)|(x<<(64-n))) & 0xFFFFFFFFFFFFFFFF

def _prince_sbox_layer(x: int) -> int:
    y = 0
    for i in range(16):
        y |= _PRINCE_S[(x >> (i*4)) & 0xF] << (i*4)
    return y

def _prince_sbox_inv_layer(x: int) -> int:
    y = 0
    for i in range(16):
        y |= _PRINCE_S_INV[(x >> (i*4)) & 0xF] << (i*4)
    return y

def _prince_key_schedule(k: bytes) -> Tuple[int,int]:
    assert len(k)==16
    k0 = int.from_bytes(k[:8], "big")
    k1 = int.from_bytes(k[8:], "big")
    k0_prime = _rotr64(k0,1) ^ (k0 >> 63)
    return k0_prime, k1

def prince_encrypt_block(block8: bytes, key16: bytes) -> bytes:
    x = int.from_bytes(block8, "big")
    k0p, k1 = _prince_key_schedule(key16)
    x ^= k0p
    for r in range(5):
        x ^= _PRINCE_RCS[r]
        x = _prince_sbox_layer(x)
        x = _m_prime(x)
        x ^= k1
    x ^= _PRINCE_RCS[5]; x = _prince_sbox_layer(x); x = _m_prime(x); x ^= _PRINCE_RCS[6]
    for r in range(7,12):
        x ^= k1
        x = _m_prime(x)
        x = _prince_sbox_inv_layer(x)
        x ^= _PRINCE_RCS[r]
    x ^= k0p ^ _PRINCE_ALPHA
    return x.to_bytes(8, "big")

def encrypt_prince_ctr_bytes(plaintext: bytes) -> bytes:
    key = get_random_bytes(16)
    nonce = get_random_bytes(4)
    return ctr_encrypt(lambda b,k: prince_encrypt_block(b,k), key, nonce, plaintext, 8)

# -----------------------------
# RECTANGLE-80 (64-bit block)
# -----------------------------
_RECT_S = [0x6,0x5,0xC,0xA,0x1,0xE,0x7,0x9,0xB,0x0,0x3,0xD,0x8,0xF,0x4,0x2]
def _rect_sbox(x: int) -> int:
    y=0
    for i in range(16):
        y |= (_RECT_S[(x>>(i*4))&0xF]&0xF)<<(i*4)
    return y

_RECT_P = [0]*64
for r in range(4):
    for c in range(16):
        _RECT_P[r*16 + c] = c*4 + r

def _rect_p(x: int) -> int:
    y = 0
    for i in range(64):
        y |= ((x>>i)&1) << _RECT_P[i]
    return y

def _rect_key_schedule_80(k: bytes):
    w = list(struct.unpack(">5H", k))
    round_keys = []
    for r in range(1, 26+1):
        rk = (w[0] << 48) | (w[1] << 32) | (w[2] << 16) | w[3]
        round_keys.append(rk & ((1<<64)-1))
        whole = (w[0]<<64)|(w[1]<<48)|(w[2]<<32)|(w[3]<<16)|w[4]
        whole = ((whole >> 61) | ((whole & ((1<<61)-1)) << (80-61))) & ((1<<80)-1)
        w = [ (whole>>64)&0xFFFF, (whole>>48)&0xFFFF, (whole>>32)&0xFFFF, (whole>>16)&0xFFFF, whole&0xFFFF ]
        hi_nib = (w[0] >> 12) & 0xF
        hi_nib = _RECT_S[hi_nib]
        w[0] = ((w[0] & 0x0FFF) | (hi_nib << 12)) & 0xFFFF
        w[1] ^= ((r & 0x1F) << 11)
        w[2] ^= ((r >> 5) & 0x1) << 15
    return round_keys

def rectangle80_encrypt_block(block8: bytes, key10: bytes) -> bytes:
    assert len(block8)==8 and len(key10)==10
    x = int.from_bytes(block8, "big")
    rks = _rect_key_schedule_80(key10)
    for i in range(25):
        x ^= rks[i]
        x = _rect_sbox(x)
        x = _rect_p(x)
    x ^= rks[-1]
    return x.to_bytes(8, "big")

def encrypt_rectangle_ctr_bytes(plaintext: bytes) -> bytes:
    key = get_random_bytes(10)  # 80-bit key
    nonce = get_random_bytes(4) # 32-bit nonce
    return ctr_encrypt(lambda b,k: rectangle80_encrypt_block(b,k), key, nonce, plaintext, 8)

# -----------------------------
# ASCON-128 (v2) AEAD (ct||tag)
# -----------------------------
_ASCON_IV_128 = (0x80400c0600000000).to_bytes(8, "big")
_ASCON_ROUNDS_A = 12
_ASCON_ROUNDS_B = 6
_ASCON_RC = [
    0x0f0e0d0c0b0a0908,0x1716151413121110,
    0x1f1e1d1c1b1a1918,0x2726252423222120,
    0x2f2e2d2c2b2a2928,0x3736353433323130,
    0x3f3e3d3c3b3a3938,0x4746454443424140,
    0x4f4e4d4c4b4a4948,0x5756555453525150,
    0x5f5e5d5c5b5a5958,0x6766656463626160
]

def _rotr64(x, n): return ((x >> n) | (x << (64 - n))) & 0xFFFFFFFFFFFFFFFF

def _ascon_sbox(x0,x1,x2,x3,x4):
    x0 ^= x4; x4 ^= x3; x2 ^= x1
    t0 = (~x0) & x1; t1 = (~x1) & x2; t2 = (~x2) & x3; t3 = (~x3) & x4; t4 = (~x4) & x0
    x0 ^= t1; x1 ^= t2; x2 ^= t3; x3 ^= t4; x4 ^= t0
    x1 ^= x0; x0 ^= x4; x3 ^= x2; x2 = ~x2 & 0xFFFFFFFFFFFFFFFF
    return x0&0xFFFFFFFFFFFFFFFF, x1&0xFFFFFFFFFFFFFFFF, x2&0xFFFFFFFFFFFFFFFF, x3&0xFFFFFFFFFFFFFFFF, x4&0xFFFFFFFFFFFFFFFF

def _ascon_linear(x0,x1,x2,x3,x4):
    x0 ^= _rotr64(x0,19) ^ _rotr64(x0,28)
    x1 ^= _rotr64(x1,61) ^ _rotr64(x1,39)
    x2 ^= _rotr64(x2,1)  ^ _rotr64(x2,6)
    x3 ^= _rotr64(x3,10) ^ _rotr64(x3,17)
    x4 ^= _rotr64(x4,7)  ^ _rotr64(x4,41)
    return x0,x1,x2,x3,x4

def _ascon_permute(x0,x1,x2,x3,x4, rounds):
    for i in range(12 - rounds, 12):
        x2 ^= _ASCON_RC[i]
        x0,x1,x2,x3,x4 = _ascon_sbox(x0,x1,x2,x3,x4)
        x0,x1,x2,x3,x4 = _ascon_linear(x0,x1,x2,x3,x4)
    return x0,x1,x2,x3,x4

def ascon128_encrypt_bytes(plaintext: bytes) -> bytes:
    key16 = get_random_bytes(16)
    nonce16 = get_random_bytes(16)
    k0 = int.from_bytes(key16[:8],"big"); k1 = int.from_bytes(key16[8:],"big")
    n0 = int.from_bytes(nonce16[:8],"big"); n1 = int.from_bytes(nonce16[8:],"big")
    x0 = int.from_bytes(_ASCON_IV_128, "big")
    x1 = k0; x2 = k1; x3 = n0; x4 = n1
    x0,x1,x2,x3,x4 = _ascon_permute(x0,x1,x2,x3,x4, _ASCON_ROUNDS_A)
    x3 ^= k0; x4 ^= k1
    ct = bytearray()
    off = 0
    while off < len(plaintext):
        m = plaintext[off:off+8]
        m64 = int.from_bytes(m.ljust(8,b"\x00"), "big")
        x0 ^= m64
        c64 = x0
        ct.extend(c64.to_bytes(8,"big")[:len(m)])
        x0,x1,x2,x3,x4 = _ascon_permute(x0,x1,x2,x3,x4, _ASCON_ROUNDS_B)
        off += 8
    x1 ^= k0; x2 ^= k1
    x0,x1,x2,x3,x4 = _ascon_permute(x0,x1,x2,x3,x4, _ASCON_ROUNDS_A)
    x3 ^= k0; x4 ^= k1
    tag = (x3.to_bytes(8,"big") + x4.to_bytes(8,"big"))
    return bytes(ct) + tag  # ct||tag

# -----------------------------
# LEA-128 (128-bit block)
# -----------------------------
def _u32(x): return x & 0xFFFFFFFF
def _rol32(x, r): return _u32((x << r) | (x >> (32 - r)))
def _ror32(x, r): return _u32((x >> r) | (x << (32 - r)))

_LEA_DELTA = [0x9e3779b9, 0x3c6ef373, 0x78dde6e6, 0xf1bbcdcc,
              0xe3779b99, 0xc6ef3733, 0x8dde6e67, 0x1bbcdccf]

def _lea_expand_128(key16: bytes):
    assert len(key16) == 16
    T0, T1, T2, T3 = struct.unpack("<4I", key16)
    rk = []
    for i in range(24):
        d = _LEA_DELTA[i % 8]
        K0 = _u32(T0 + _rol32(d, 0))
        K1 = _u32(T1 + _rol32(d, 1))
        K2 = _u32(T2 + _rol32(d, 2))
        K3 = _u32(T1 + _rol32(d, 3))
        K4 = _u32(T2 + _rol32(d, 4))
        K5 = _u32(T3 + _rol32(d, 5))
        rk.extend([K0, K1, K2, K3, K4, K5])
        T0 = _rol32(T0, 1)
        T1 = _rol32(T1, 3)
        T2 = _rol32(T2, 6)
        T3 = _rol32(T3, 11)
    return rk

def lea128_encrypt_block(block16: bytes, key16: bytes) -> bytes:
    assert len(block16) == 16
    rk = _lea_expand_128(key16)
    X0, X1, X2, X3 = struct.unpack("<4I", block16)
    for r in range(24):
        k0, k1, k2, k3, k4, k5 = rk[6*r:6*r+6]
        X0 = _rol32(_u32(X0 + (X1 ^ k0)), 9)
        X1 = _ror32(_u32(X1 + (X2 ^ k1)), 5)
        X2 = _ror32(_u32(X2 + (X3 ^ k2)), 3)
        X3 = _rol32(_u32(X3 + (X0 ^ k3)), 9)
        X0 = _ror32(_u32(X0 + (X1 ^ k4)), 5)
        X1 = _ror32(_u32(X1 + (X2 ^ k5)), 3)
    return struct.pack("<4I", X0, X1, X2, X3)

def encrypt_lea_ctr_bytes(plaintext: bytes) -> bytes:
    key = get_random_bytes(16)   # 128-bit key
    nonce = get_random_bytes(12) # 96-bit nonce, leaves 4B counter
    return ctr_encrypt(lambda b,k: lea128_encrypt_block(b,k), key, nonce, plaintext, 16)

# -----------------------------
# MSEA-96 (96-bit block)
# -----------------------------
def _msea_f_function(x: int, k: int) -> int:
    x = (x + k) & 0xFFFFFFFF
    rot = (x & 0x1F) + 1
    x = ((x << rot) | (x >> (32 - rot))) & 0xFFFFFFFF
    x = x ^ ((x >> 16) & 0xFFFF)
    x = (x + 0x9E3779B9) & 0xFFFFFFFF
    return x

def _msea_key_schedule(key12: bytes) -> list:
    assert len(key12) == 12
    k0, k1, k2 = struct.unpack(">3I", key12)
    round_keys = []
    for i in range(16):
        temp = (k0 + k1 + k2 + i) & 0xFFFFFFFF
        k0 = ((k0 << 7) | (k0 >> 25)) & 0xFFFFFFFF
        k1 = ((k1 << 11) | (k1 >> 21)) & 0xFFFFFFFF
        k2 = ((k2 << 13) | (k2 >> 19)) & 0xFFFFFFFF
        round_key = (temp + 0x12345678 * (i + 1)) & 0xFFFFFFFF
        round_keys.append(round_key)
        k0 = (k0 + round_key) & 0xFFFFFFFF
        k1 = (k1 + k2) & 0xFFFFFFFF
        k2 = (k2 + temp) & 0xFFFFFFFF
    return round_keys

def msea96_encrypt_block(block12: bytes, key12: bytes) -> bytes:
    assert len(block12) == 12 and len(key12) == 12
    round_keys = _msea_key_schedule(key12)
    L, M, R = struct.unpack(">3I", block12)
    for i in range(16):
        rk = round_keys[i]
        if i % 3 == 0:
            L = L ^ _msea_f_function(M ^ R, rk)
        elif i % 3 == 1:
            M = M ^ _msea_f_function(R ^ L, rk)
        else:
            R = R ^ _msea_f_function(L ^ M, rk)
        if (i + 1) % 4 == 0:
            L, M, R = M, R, L
    return struct.pack(">3I", L, M, R)

def encrypt_msea_ctr_bytes(plaintext: bytes) -> bytes:
    key = get_random_bytes(12)    # 96-bit key
    nonce = get_random_bytes(8)   # 64-bit nonce (leaves 4B counter)
    return ctr_encrypt(lambda b,k: msea96_encrypt_block(b,k), key, nonce, plaintext, 12)

# -----------------------------
# Metric structures
# -----------------------------

@dataclass
class MetricRow:
    algorithm: str
    infile: str
    outfile: str
    file_bytes: int
    file_kb: float
    has_tag: int
    elapsed_ms: float
    throughput_mb_s: float
    cpu_user_s: float
    cpu_system_s: float
    rss_mb_before: float
    rss_mb_after: float
    rss_mb_delta: float
    repeats: int

# -----------------------------
# Core measurement runner
# -----------------------------

def measure_and_write(algoname: str,
                      encrypt_bytes_fn: Callable[[bytes], bytes],
                      in_path: str, out_path: str,
                      repeats: int = 1) -> MetricRow:
    """
    Loads plaintext once, runs encryption, writes output, captures metrics.
    If repeats > 1, runs the encryption function multiple times (only writes once)
    and averages the timing (the output saved is from the first run).
    """
    ensure_dir(os.path.dirname(out_path))
    with open(in_path, "rb") as f:
        pt = f.read()

    proc = psutil.Process(os.getpid())

    # measure with optional repeats
    times = []
    cpu_user = 0.0
    cpu_sys  = 0.0
    rss_before = proc.memory_info().rss / (1024*1024)
    ct_first = None

    for i in range(repeats):
        t0_wall = time.perf_counter()
        c0 = proc.cpu_times()
        ct = encrypt_bytes_fn(pt)
        c1 = proc.cpu_times()
        t1_wall = time.perf_counter()

        if i == 0:
            ct_first = ct

        times.append(t1_wall - t0_wall)
        cpu_user += (c1.user - c0.user)
        cpu_sys  += (c1.system - c0.system)

    # write only once
    with open(out_path, "wb") as f:
        f.write(ct_first)

    rss_after = proc.memory_info().rss / (1024*1024)

    elapsed_sec = sum(times) / len(times)
    fbytes = len(pt)
    throughput = (fbytes / (1024.0*1024.0)) / elapsed_sec if elapsed_sec > 0 else 0.0

    has_tag = 1 if algoname.upper().startswith("ASCON") else 0

    row = MetricRow(
        algorithm=algoname,
        infile=os.path.basename(in_path),
        outfile=os.path.basename(out_path),
        file_bytes=fbytes,
        file_kb=bytes_to_kb(fbytes),
        has_tag=has_tag,
        elapsed_ms=elapsed_sec * 1000.0,
        throughput_mb_s=throughput,
        cpu_user_s=cpu_user / repeats,
        cpu_system_s=cpu_sys / repeats,
        rss_mb_before=rss_before,
        rss_mb_after=rss_after,
        rss_mb_delta=(rss_after - rss_before),
        repeats=repeats
    )
    return row

# -----------------------------
# Main pipeline
# -----------------------------

ALGOS: Dict[str, Callable[[bytes], bytes]] = {
    "AES":        encrypt_aes_ctr_bytes,
    "PRESENT":    encrypt_present_ctr_bytes,
    "XTEA":       encrypt_xtea_ctr_bytes,
    "SIMON":      encrypt_simon_ctr_bytes,
    "PRINCE":     encrypt_prince_ctr_bytes,
    "RECTANGLE":  encrypt_rectangle_ctr_bytes,
    "ASCONv2":    ascon128_encrypt_bytes,
    "LEA":        encrypt_lea_ctr_bytes,
    "MSEA":       encrypt_msea_ctr_bytes,
}

def run(plaintext_dir: str = "dataset/plaintexts",
        out_root: str = "dataset",
        metrics_dir: str = "metrics",
        repeats: int = 3,
        filter_sizes_kb: str = "") -> None:
    """
    plaintext_dir: folder with *.bin plaintexts
    out_root: base output dataset folder
    metrics_dir: where to write CSVs
    repeats: number of encryption repeats to average timing (higher -> smoother)
    filter_sizes_kb: optional comma-separated list (e.g., "16,64,256") to restrict which sizes to process
    """
    ensure_dir(metrics_dir)

    # collect target sizes if filter provided
    allowed_sizes = set()
    if filter_sizes_kb.strip():
        try:
            allowed_sizes = {int(x.strip()) for x in filter_sizes_kb.split(",") if x.strip()}
        except ValueError:
            print("Invalid filter_sizes_kb; must be comma-separated integers like '16,64,256'.", file=sys.stderr)
            sys.exit(2)

    # discover plaintexts
    try:
        all_files = [f for f in os.listdir(plaintext_dir) if f.endswith(".bin")]
    except FileNotFoundError:
        print(f"Plaintext dir not found: {plaintext_dir}", file=sys.stderr)
        sys.exit(1)

    if allowed_sizes:
        files = [f for f in all_files if any(f.startswith(f"{sz}KB_") for sz in allowed_sizes)]
    else:
        files = all_files

    files.sort()
    if not files:
        print("No plaintext .bin files found to process.", file=sys.stderr)
        sys.exit(1)

    # prepare outputs
    for algo in ALGOS:
        ensure_dir(os.path.join(out_root, algo))

    metrics_path = os.path.join(metrics_dir, "perf_metrics.csv")
    manifest_path = os.path.join(metrics_dir, "dataset_manifest.csv")

    # CSV writers
    fieldnames = list(MetricRow.__annotations__.keys())
    wrote_header = False
    wrote_manifest_header = False

    with open(metrics_path, "w", newline="") as mfp, open(manifest_path, "w", newline="") as manfp:
        mw = csv.DictWriter(mfp, fieldnames=fieldnames)
        manw = csv.DictWriter(manfp, fieldnames=["path","label","size_kb","has_tag"])

        mw.writeheader(); wrote_header = True
        manw.writeheader(); wrote_manifest_header = True

        # iterate
        for fname in files:
            in_path = os.path.join(plaintext_dir, fname)
            for algo, fn in ALGOS.items():
                out_dir = os.path.join(out_root, algo)
                out_path = os.path.join(out_dir, f"{algo}_{fname}")
                try:
                    row = measure_and_write(algo, fn, in_path, out_path, repeats=repeats)
                    mw.writerow(asdict(row))
                except Exception as e:
                    print(f"[{algo}] FAILED on {fname}: {e}", file=sys.stderr)
                    continue
                # manifest row
                size_kb = int(round(row.file_kb))
                manw.writerow({
                    "path": out_path.replace("\\","/"),
                    "label": algo,
                    "size_kb": size_kb,
                    "has_tag": row.has_tag
                })
                print(f"[{algo:<10}] {fname} -> {os.path.basename(out_path)} | {row.elapsed_ms:.2f} ms, {row.throughput_mb_s:.2f} MB/s")

    print("\nDone.")
    print(f"Metrics CSV : {metrics_path}")
    print(f"Manifest CSV: {manifest_path}")
    print("You can now train ML models using these system-level features for higher separability.")

def parse_args():
    ap = argparse.ArgumentParser(description="Rebuild dataset and collect system metrics per algorithm.")
    ap.add_argument("--plaintext_dir", default="dataset/plaintexts", help="Folder with *.bin plaintexts")
    ap.add_argument("--out_root", default="dataset", help="Root output folder for ciphertexts")
    ap.add_argument("--metrics_dir", default="metrics", help="Folder to save metrics CSVs")
    ap.add_argument("--repeats", type=int, default=3, help="Repeats per encryption to average timing")
    ap.add_argument("--filter_sizes_kb", default="", help="Comma-separated sizes to process, e.g. '16,64,256'")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.plaintext_dir, args.out_root, args.metrics_dir, args.repeats, args.filter_sizes_kb)
