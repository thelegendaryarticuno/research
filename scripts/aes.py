import os
import sys
from typing import Optional
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


def encrypt_aes_ctr(in_file: str, out_file: str) -> bool:
    """
    Encrypt a file using AES in CTR mode.
    
    Args:
        in_file: Path to the input plaintext file
        out_file: Path to the output ciphertext file
        
    Returns:
        True if encryption was successful, False otherwise
    """
    try:
        key = get_random_bytes(16)   # 128-bit key
        nonce = get_random_bytes(8)  # 64-bit nonce
        cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)

        # Read the plaintext file
        with open(in_file, "rb") as f:
            plaintext = f.read()
        
        # Encrypt the data
        ciphertext = cipher.encrypt(plaintext)

        # Write the ciphertext to output file
        with open(out_file, "wb") as f:
            f.write(ciphertext)
            
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{in_file}' not found.")
        return False
    except PermissionError:
        print(f"Error: Permission denied when accessing files.")
        return False
    except Exception as e:
        print(f"Error during encryption: {str(e)}")
        return False


def main():
    """Main function to process all plaintext files."""
    # Input plaintext folder
    plaintext_dir = "dataset/plaintexts"
    # Output ciphertext folder
    ciphertext_dir = "dataset/AES"
    
    # Validate input directory exists
    if not os.path.exists(plaintext_dir):
        print(f"Error: Input directory '{plaintext_dir}' does not exist.")
        sys.exit(1)
    
    # Create output directory
    try:
        os.makedirs(ciphertext_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {str(e)}")
        sys.exit(1)
    
    # Get list of .bin files
    try:
        files = [f for f in os.listdir(plaintext_dir) if f.endswith(".bin")]
    except Exception as e:
        print(f"Error reading directory: {str(e)}")
        sys.exit(1)
    
    if not files:
        print(f"No .bin files found in '{plaintext_dir}'")
        return
    
    print(f"Found {len(files)} files to encrypt...")
    
    # Process each file
    successful = 0
    failed = 0
    
    for fname in files:
        in_path = os.path.join(plaintext_dir, fname)
        out_name = f"AES_{fname}"   # prepend AES_ to file name
        out_path = os.path.join(ciphertext_dir, out_name)
        
        if encrypt_aes_ctr(in_path, out_path):
            print(f"✓ Encrypted {fname} -> {out_name}")
            successful += 1
        else:
            print(f"✗ Failed to encrypt {fname}")
            failed += 1
    
    print(f"\nEncryption complete: {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()
