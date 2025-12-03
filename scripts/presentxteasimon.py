import os
import struct
from typing import Tuple
from Crypto.Random import get_random_bytes

# =========================
# Helpers (CTR mode wrapper)
# =========================

def ctr_encrypt(block_encrypt, key: bytes, nonce: bytes, plaintext: bytes, block_size_bytes: int) -> bytes:
    """
    Generic CTR: counter = nonce || ctr (big-endian), sized to block.
    nonce length must be <= block_size_bytes - 4 (we use 32-bit counter).
    """
    assert len(nonce) <= block_size_bytes - 4, "nonce too long for CTR layout"
    out = bytearray()
    ctr = 0
    off = 0
    while off < len(plaintext):
        # assemble counter block
        rem = block_size_bytes - len(nonce) - 4
        counter_block = nonce + (b"\x00" * rem) + struct.pack(">I", ctr)
        keystream = block_encrypt(counter_block, key)
        chunk = plaintext[off: off + block_size_bytes]
        out.extend(bytes(a ^ b for a, b in zip(chunk, keystream[:len(chunk)])))
        ctr = (ctr + 1) & 0xffffffff
        off += block_size_bytes
    return bytes(out)

# =========================
# PRESENT-80 (block = 64b)
# =========================

# S-box and P-layer for PRESENT
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
    """
    PRESENT-80 key schedule.
    key = 80-bit as (key_hi: 16 bits, key_lo: 64 bits)
    Yields 32 round subkeys (k79..k16 of current state).
    """
    for round_counter in range(1, 32+1):
        # round key is the MSB 64 bits: (key_hi[15:0] || key_lo[63:?]) — effectively (key >> 16)
        round_key = ((key_hi << 64) | key_lo) >> 16
        yield round_key & ((1<<64)-1)

        # rotate key left by 61
        whole = ((key_hi << 64) | key_lo) & ((1<<80)-1)
        whole = ((whole << 61) | (whole >> (80-61))) & ((1<<80)-1)
        key_hi = (whole >> 64) & 0xFFFF
        key_lo = whole & ((1<<64)-1)

        # S-box on high 4 bits
        high4 = (key_hi >> 12) & 0xF
        high4 = PRESENT_SBOX[high4]
        key_hi = ((key_hi & 0x0FFF) | (high4 << 12)) & 0xFFFF

        # XOR round counter into bits k19..k15 (i.e., into key_lo bits [19:15])
        key_lo ^= (round_counter & 0x1F) << 15

def present80_encrypt_block(block8: bytes, key10: bytes) -> bytes:
    assert len(block8) == 8 and len(key10) == 10
    state = int.from_bytes(block8, "big")
    key_hi = int.from_bytes(key10[:2], "big")      # 16 bits
    key_lo = int.from_bytes(key10[2:], "big")      # 64 bits

    rks = list(_present_key_schedule_80(key_hi, key_lo))
    for i in range(31):
        state ^= rks[i]
        state = _present_sbox_layer(state)
        state = _present_p_layer(state)
    state ^= rks[31]
    return state.to_bytes(8, "big")

# Wrapper for CTR with PRESENT-80
def encrypt_present_ctr(in_file: str, out_file: str):
    key = get_random_bytes(10)   # 80-bit key
    nonce = get_random_bytes(4)  # 32-bit nonce for 64-bit block (leaves 4 bytes for counter)
    with open(in_file, "rb") as f:
        pt = f.read()
    # pad last partial block with zeros for CTR xor (safe since CTR keystream)
    # (ctr_encrypt handles arbitrary length)
    def be(block, keybytes):
        return present80_encrypt_block(block, keybytes)
    ct = ctr_encrypt(lambda b,k: be(b,k), key, nonce, pt, 8)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(ct)

# =========================
# XTEA (block = 64b)
# =========================

def xtea_encrypt_block(block8: bytes, key16: bytes, rounds: int = 32) -> bytes:
    assert len(block8) == 8 and len(key16) == 16
    v0, v1 = struct.unpack(">2I", block8)
    k = struct.unpack(">4I", key16)
    delta = 0x9E3779B9
    s = 0
    for _ in range(rounds):
        v0 = (v0 + (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (s + k[s & 3])) & 0xFFFFFFFF
        s = (s + delta) & 0xFFFFFFFF
        v1 = (v1 + (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (s + k[(s>>11) & 3])) & 0xFFFFFFFF
    return struct.pack(">2I", v0, v1)

def encrypt_xtea_ctr(in_file: str, out_file: str):
    key = get_random_bytes(16)  # 128-bit
    nonce = get_random_bytes(4) # 32-bit nonce for 64-bit block (leaves 4 bytes for counter)
    with open(in_file, "rb") as f:
        pt = f.read()
    def be(block, keybytes):
        return xtea_encrypt_block(block, keybytes)
    ct = ctr_encrypt(lambda b,k: be(b,k), key, nonce, pt, 8)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(ct)

# =========================
# SIMON64/128 (block = 64b)
# =========================

def _rol32(x, r): return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF
def _ror32(x, r): return ((x >> r) | (x << (32 - r))) & 0xFFFFFFFF

# z3 sequence for SIMON64/128 (from the spec), as bits MSB->LSB in a repeating pattern:
_Z3 = int("11111010001001010110000111001101111101000100101011000011100110", 2)

def _simon64_128_expand_keys(key16: bytes):
    # Key as four 32-bit words k0..k3 (k0 is least significant in many refs; here use big-endian order)
    k_words = list(struct.unpack(">4I", key16))
    m = 4
    T = 72
    c = 0xFFFFFFFC
    z = _Z3
    ks = [0]*T
    ks[:m] = k_words[::-1]  # many refs load k_{m-1}..k0; this ordering matches the commonly used schedule
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

def encrypt_simon_ctr(in_file: str, out_file: str):
    key = get_random_bytes(16)  # 128-bit key
    nonce = get_random_bytes(4) # 32-bit nonce for 64-bit block (leaves 4 bytes for counter)
    with open(in_file, "rb") as f:
        pt = f.read()
    def be(block, keybytes):
        return simon64_128_encrypt_block(block, keybytes)
    ct = ctr_encrypt(lambda b,k: be(b,k), key, nonce, pt, 8)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(ct)

# =========================
# Batch loop over plaintexts
# =========================

PLAINTEXT_DIR = "dataset/plaintexts"
OUT_DIRS = {
    "PRESENT": "dataset/PRESENT",
    "XTEA":    "dataset/XTEA",
    "SIMON":   "dataset/SIMON",
}

os.makedirs(OUT_DIRS["PRESENT"], exist_ok=True)
os.makedirs(OUT_DIRS["XTEA"], exist_ok=True)
os.makedirs(OUT_DIRS["SIMON"], exist_ok=True)

for fname in os.listdir(PLAINTEXT_DIR):
    if not fname.endswith(".bin"):
        continue
    in_path = os.path.join(PLAINTEXT_DIR, fname)

    # PRESENT-80
    out_present = os.path.join(OUT_DIRS["PRESENT"], f"PRESENT_{fname}")
    encrypt_present_ctr(in_path, out_present)
    print(f"[PRESENT]  {fname} -> {os.path.basename(out_present)}")

    # XTEA
    out_xtea = os.path.join(OUT_DIRS["XTEA"], f"XTEA_{fname}")
    encrypt_xtea_ctr(in_path, out_xtea)
    print(f"[XTEA]     {fname} -> {os.path.basename(out_xtea)}")

    # SIMON64/128
    out_simon = os.path.join(OUT_DIRS["SIMON"], f"SIMON_{fname}")
    encrypt_simon_ctr(in_path, out_simon)
    print(f"[SIMON]    {fname} -> {os.path.basename(out_simon)}")
