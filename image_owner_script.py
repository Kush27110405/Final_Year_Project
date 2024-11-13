import cv2
import numpy as np
import pywt
from scipy.linalg import svd
from web3 import Web3
import requests
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

INFURA_PROJECT_ID = os.getenv('INFURA_PROJECT_ID')
PRIVATE_KEY = os.getenv('PRIVATE_KEY_1')
IMAGE_DEPOSITORY_ADDRESS = os.getenv('IMAGE_DEPOSITORY_ADDRESS')
PINATA_API_KEY = os.getenv('PINATA_API_KEY')
PINATA_SECRET_API_KEY = os.getenv('PINATA_SECRET_API_KEY')

# Load ABIs
with open('abi.json') as f:
    abi_data = json.load(f)

image_depository_abi = abi_data["ImageDepository"]

# Connect to Ethereum network (Sepolia testnet)
w3 = Web3(Web3.HTTPProvider(f'https://sepolia.infura.io/v3/{INFURA_PROJECT_ID}'))

# Instantiate the contract
image_depository_contract = w3.eth.contract(address=IMAGE_DEPOSITORY_ADDRESS, abi=image_depository_abi)

# Wallet setup
account = w3.eth.account.from_key(PRIVATE_KEY)

# Image processing and watermarking functions
def normalize_image(image, M):
    return cv2.resize(image, (M, M))

def extract_effective_region(image, N):
    h, w = image.shape
    cx, cy = h // 2, w // 2
    x_start, y_start = cx - N // 2, cy - N // 2
    return image[x_start:x_start + N, y_start:y_start + N]

def wavelet_transform(image, level):
    coeffs = pywt.wavedec2(image, 'haar', level=level)
    return coeffs[0]

def calculate_svd_features(subband, n):
    k = subband.shape[0] // n
    feature_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            subblock = subband[i * n:(i + 1) * n, j * n:(j + 1) * n]
            _, s, _ = svd(subblock)
            feature_matrix[i, j] = s[0]
    return feature_matrix

def generate_feature_image(feature_matrix):
    F = np.zeros_like(feature_matrix, dtype=int)
    for i in range(feature_matrix.shape[0]):
        for j in range(feature_matrix.shape[1]):
            F[i, j] = 1 if int(feature_matrix[i, j]) % 2 != 0 else 0
    return F

def resize_logo(logo_image, target_size):
    return cv2.resize(logo_image, target_size, interpolation=cv2.INTER_NEAREST)

def generate_zero_watermark_image(original_image, logo_image, M, N, n, level, key, save_path=None):
    normalized_image = normalize_image(original_image, M)
    effective_region = extract_effective_region(normalized_image, N)
    low_freq_subband = wavelet_transform(effective_region, level)
    feature_matrix = calculate_svd_features(low_freq_subband, n)
    feature_image = generate_feature_image(feature_matrix)
    logo_image_resized = resize_logo(logo_image, feature_image.shape)
    scrambled_F = np.bitwise_xor(feature_image, logo_image_resized)
    if save_path:
        cv2.imwrite(save_path, (scrambled_F * 255).astype(np.uint8))
    return scrambled_F, key

# Pinata API interaction
def upload_to_pinata(file_path):
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    headers = {
        "pinata_api_key": PINATA_API_KEY,
        "pinata_secret_api_key": PINATA_SECRET_API_KEY
    }
    with open(file_path, 'rb') as file:
        response = requests.post(url, headers=headers, files={"file": file})
    if response.status_code == 200:
        return response.json()["IpfsHash"]
    else:
        raise Exception(f"Failed to upload to Pinata: {response.text}")

def add_key_to_depository(user_info, ipfs_address, scrambling_params):
    # Verify DATA_OWNER matches the current account
    data_owner = image_depository_contract.functions.DATA_OWNER().call()
    print(f"Contract DATA_OWNER: {data_owner}")
    print(f"Script account address: {account.address}")

    if data_owner != account.address:
        raise Exception("Account address does not match DATA_OWNER in the contract.")

    # Build the transaction
    try:
        tx = image_depository_contract.functions.addKey(user_info, ipfs_address, scrambling_params).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 400000,  # Explicitly set gas limit
        })
        print(f"Transaction built: {tx}")

        # Sign and send the transaction
        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"Transaction hash for addKey: {tx_hash.hex()}")
        return tx_hash
    except Exception as e:
        print(f"Error in addKey transaction: {e}")
        raise


# Main process for IO
if __name__ == "__main__":
    original_image = cv2.imread("original_image.png", cv2.IMREAD_GRAYSCALE)
    logo_image = cv2.imread("logo_image.png", cv2.IMREAD_GRAYSCALE)
    M, N, n = 512, 128, 8
    level = 2
    key = 12345
    save_path = "zero_watermark_image.png"

    zero_watermark_image, encryption_key = generate_zero_watermark_image(
        original_image, logo_image, M, N, n, level, key, save_path=save_path
    )
    
    ipfs_hash = upload_to_pinata(save_path)
    print(f"IPFS hash for zero-watermark: {ipfs_hash}")

    add_key_to_depository("user_info_example", ipfs_hash, str(encryption_key))
