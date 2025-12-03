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

# -------------------------------------------------------------------
# ============ PRINCE (64-bit block, 128-bit key) ====================
# Minimal reference-style implementation (untweaked constants)
# -------------------------------------------------------------------
_PRINCE_S = [0xB,0xF,0x3,0x2,0xA,0xC,0x9,0x1,0x6,0x7,0x8,0x0,0xE,0x5,0xD,0x4]
_PRINCE_S_INV = [0xB,0x7,0x3,0x2,0xF,0xD,0x8,0x9,0xA,0x6,0x4,0x0,0x5,0xE,0xC,0x1]
# M' linear diffusion (bit permutation by 4x4 matrix per nibble column)
def _m_prime(x: int) -> int:
    # operate as 16 nibbles in a 4x4 structure; compact row/col mixing
    # Precomputed nibble-wise linear layer (fast path)
    def mix4(n0,n1,n2,n3):
        # From PRINCE spec (M'): [0 0 1 1; 1 0 0 1; 1 1 0 0; 0 1 1 0]
        r0 = n2 ^ n3
        r1 = n0 ^ n3
        r2 = n0 ^ n1
        r3 = n1 ^ n2
        return r0 & 0xF, r1 & 0xF, r2 & 0xF, r3 & 0xF
    # decompose into 4 columns of 4 nibbles
    ns = [(x >> (4*i)) & 0xF for i in range(16)]
    cols = [
        (ns[0], ns[4], ns[8],  ns[12]),
        (ns[1], ns[5], ns[9],  ns[13]),
        (ns[2], ns[6], ns[10], ns[14]),
        (ns[3], ns[7], ns[11], ns[15]),
    ]
    out_cols = []
    for c in cols:
        out_cols.append(mix4(*c))
    # write back
    res = 0
    # column 0 fills nibble positions 0,4,8,12 etc.
    positions = [
        (0,4,8,12),
        (1,5,9,13),
        (2,6,10,14),
        (3,7,11,15),
    ]
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

def _rotl64(x,n): return ((x<<n)|(x>>(64-n))) & 0xFFFFFFFFFFFFFFFF
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
    # 5 forward rounds
    for r in range(5):
        x ^= _PRINCE_RCS[r]
        x = _prince_sbox_layer(x)
        x = _m_prime(x)
        x ^= k1
    # middle
    x ^= _PRINCE_RCS[5]; x = _prince_sbox_layer(x); x = _m_prime(x); x ^= _PRINCE_RCS[6]
    # 5 backward rounds
    for r in range(7,12):
        x ^= k1
        x = _m_prime(x)
        x = _prince_sbox_inv_layer(x)
        x ^= _PRINCE_RCS[r]
    x ^= k0p ^ _PRINCE_ALPHA
    return x.to_bytes(8, "big")

def encrypt_prince_ctr(in_file: str, out_file: str):
    key = get_random_bytes(16)
    nonce = get_random_bytes(4)  # Fixed: 4 bytes for 64-bit block with 4-byte counter
    with open(in_file, "rb") as f:
        pt = f.read()
    ct = ctr_encrypt(lambda b,k: prince_encrypt_block(b,k), key, nonce, pt, 8)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as f: f.write(ct)

# -------------------------------------------------------------------
# ============ RECTANGLE-80 (64-bit block, 80-bit key) ===============
# Bit-sliced 4-bit S-box + pLayer + key schedule (spec)
# -------------------------------------------------------------------
_RECT_S = [0x6,0x5,0xC,0xA,0x1,0xE,0x7,0x9,0xB,0x0,0x3,0xD,0x8,0xF,0x4,0x2]
def _rect_sbox(x: int) -> int:
    y=0
    for i in range(16):
        y |= (_RECT_S[(x>>(i*4))&0xF]&0xF)<<(i*4)
    return y

# pLayer: permutes bit (r,c) -> (r, (c+ r*16) mod 64) with row-major mapping; use table from spec
# Precompute permutation mapping for 64 bits:
_RECT_P = [0]*64
# From spec: position (row r in [0..3], column c in [0..15]) -> new position c*4 + r
for r in range(4):
    for c in range(16):
        _RECT_P[r*16 + c] = c*4 + r

def _rect_p(x: int) -> int:
    y = 0
    for i in range(64):
        y |= ((x>>i)&1) << _RECT_P[i]
    return y

def _rot_right(val: int, r: int, bits: int) -> int:
    return ((val >> r) | ((val & ((1<<r)-1)) << (bits - r))) & ((1<<bits)-1)

def _rect_key_schedule_80(k: bytes):
    # Key as five 16-bit words k0..k4 (MSB first)
    w = list(struct.unpack(">5H", k))  # 5 * 16 = 80 bits
    round_keys = []
    for r in range(1, 26+1):  # 25 rounds
        # round key: concatenate w0..w3 -> 64 bits
        rk = (w[0] << 48) | (w[1] << 32) | (w[2] << 16) | w[3]
        round_keys.append(rk & ((1<<64)-1))
        # key schedule:
        # 1) rotate right by 61 bits over 80-bit register
        whole = (w[0]<<64)|(w[1]<<48)|(w[2]<<32)|(w[3]<<16)|w[4]
        whole = ((whole >> 61) | ((whole & ((1<<61)-1)) << (80-61))) & ((1<<80)-1)
        w = [ (whole>>64)&0xFFFF, (whole>>48)&0xFFFF, (whole>>32)&0xFFFF, (whole>>16)&0xFFFF, whole&0xFFFF ]
        # 2) S-box on w0 high nibble
        hi_nib = (w[0] >> 12) & 0xF
        hi_nib = _RECT_S[hi_nib]
        w[0] = ((w[0] & 0x0FFF) | (hi_nib << 12)) & 0xFFFF
        # 3) round counter XOR to w1 bits [7:0]
        w[1] ^= ((r & 0x1F) << 11)  # aligns with spec bit placement
        w[2] ^= ((r >> 5) & 0x1) << 15
    return round_keys

def rectangle80_encrypt_block(block8: bytes, key10: bytes) -> bytes:
    assert len(block8)==8 and len(key10)==10
    x = int.from_bytes(block8, "big")
    rks = _rect_key_schedule_80(key10)
    for i in range(25):
        # AddRoundKey (XOR)
        x ^= rks[i]
        # S-box layer (nibble-wise)
        x = _rect_sbox(x)
        # pLayer
        x = _rect_p(x)
    # final ARK
    x ^= rks[-1]
    return x.to_bytes(8, "big")

def encrypt_rectangle_ctr(in_file: str, out_file: str):
    key = get_random_bytes(10)  # 80-bit key
    nonce = get_random_bytes(4) # Fixed: 4 bytes for 64-bit block with 4-byte counter
    with open(in_file, "rb") as f: pt = f.read()
    ct = ctr_encrypt(lambda b,k: rectangle80_encrypt_block(b,k), key, nonce, pt, 8)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as f: f.write(ct)

# -------------------------------------------------------------------
# ============ ASCON-128 (v2) AEAD (encrypt only) ====================
# We output: ciphertext || 16-byte tag. For classification features,
# you can strip the last 16 bytes later.
# -------------------------------------------------------------------
# Based on Ascon v1.2 spec; 320-bit state (5x64). Permutation p^a and p^b.
_ASCON_IV_128 = (0x80400c0600000000).to_bytes(8, "big")  # iv for ascon-128: a=12, b=6, k=16, r=8
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
    # χ-like sbox on 5x64-bit lanes
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

def ascon128_encrypt(key16: bytes, nonce16: bytes, ad: bytes, plaintext: bytes) -> Tuple[bytes, bytes]:
    # rate r = 8 bytes
    k0 = int.from_bytes(key16[:8],"big"); k1 = int.from_bytes(key16[8:],"big")
    n0 = int.from_bytes(nonce16[:8],"big"); n1 = int.from_bytes(nonce16[8:],"big")
    # init
    x0 = int.from_bytes(_ASCON_IV_128, "big")
    x1 = k0; x2 = k1; x3 = n0; x4 = n1
    x0,x1,x2,x3,x4 = _ascon_permute(x0,x1,x2,x3,x4, _ASCON_ROUNDS_A)
    x3 ^= k0; x4 ^= k1
    # absorb AD (rate 8)
    if ad:
        off = 0
        while off < len(ad):
            bi = int.from_bytes(ad[off:off+8].ljust(8,b"\x00"), "big")
            x0 ^= bi
            x0,x1,x2,x3,x4 = _ascon_permute(x0,x1,x2,x3,x4, _ASCON_ROUNDS_B)
            off += 8
        x4 ^= 1  # domain separation
    # encrypt plaintext
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
    # finalize
    x1 ^= k0; x2 ^= k1
    x0,x1,x2,x3,x4 = _ascon_permute(x0,x1,x2,x3,x4, _ASCON_ROUNDS_A)
    x3 ^= k0; x4 ^= k1
    tag = (x3.to_bytes(8,"big") + x4.to_bytes(8,"big"))
    return bytes(ct), tag

def encrypt_ascon128(in_file: str, out_file: str):
    key = get_random_bytes(16)
    nonce = get_random_bytes(16)
    with open(in_file, "rb") as f: pt = f.read()
    ct, tag = ascon128_encrypt(key, nonce, b"", pt)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(ct + tag)  # store tag at end; you can strip for ML features

# -------------------------------------------------------------------
# Batch loop: build outputs for PRINCE, RECTANGLE, ASCONv2
# -------------------------------------------------------------------
OUT_DIRS = {
    "PRINCE":     "dataset/PRINCE",
    "RECTANGLE":  "dataset/RECTANGLE",
    "ASCONv2":    "dataset/ASCONv2",
    # "LEA":      "dataset/LEA",   # (pending)
    # "MSEA":     "dataset/MSEA",  # (pending MSEA variant details)
}

for d in OUT_DIRS.values():
    os.makedirs(d, exist_ok=True)

for fname in os.listdir(PLAINTEXT_DIR):
    if not fname.endswith(".bin"):
        continue
    in_path = os.path.join(PLAINTEXT_DIR, fname)

    out_p = os.path.join(OUT_DIRS["PRINCE"],    f"PRINCE_{fname}")
    encrypt_prince_ctr(in_path, out_p)
    print(f"[PRINCE]    {fname} -> {os.path.basename(out_p)}")

    out_r = os.path.join(OUT_DIRS["RECTANGLE"], f"RECTANGLE_{fname}")
    encrypt_rectangle_ctr(in_path, out_r)
    print(f"[RECTANGLE] {fname} -> {os.path.basename(out_r)}")

    out_a = os.path.join(OUT_DIRS["ASCONv2"],   f"ASCONv2_{fname}")
    encrypt_ascon128(in_path, out_a)
    print(f"[ASCONv2]   {fname} -> {os.path.basename(out_a)}")
