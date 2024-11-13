import cv2
import numpy as np
from web3 import Web3
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

INFURA_PROJECT_ID = os.getenv('INFURA_PROJECT_ID')
PRIVATE_KEY = os.getenv('PRIVATE_KEY_2')
IMAGE_VALIDATION_ADDRESS = os.getenv('IMAGE_VALIDATION_ADDRESS')

# Load ABIs
with open('abi.json') as f:
    abi_data = json.load(f)

image_validation_abi = abi_data["ImageValidation"]

# Connect to Ethereum network (Sepolia testnet)
w3 = Web3(Web3.HTTPProvider(f'https://sepolia.infura.io/v3/{INFURA_PROJECT_ID}'))

# Instantiate the contract
image_validation_contract = w3.eth.contract(address=IMAGE_VALIDATION_ADDRESS, abi=image_validation_abi)

# Wallet setup
account = w3.eth.account.from_key(PRIVATE_KEY)

# Image processing and validation functions
def inverse_cat_map(img, key):
    x_len, y_len = img.shape
    descrambled_img = np.zeros_like(img)
    for x in range(x_len):
        for y in range(y_len):
            nx = (x + y) % x_len
            ny = (x + 2 * y) % y_len
            descrambled_img[x, y] = img[nx, ny]
    return descrambled_img

def resize_logo(logo_image, target_size):
    return cv2.resize(logo_image, target_size, interpolation=cv2.INTER_NEAREST)

# Pinata IPFS gateway interaction
def retrieve_ipfs_data(ipfs_hash):
    url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to retrieve from Pinata: {response.text}")

def validate_zero_watermark(original_image, zero_watermark_image, logo_image, key):
    logo_image_resized = resize_logo(logo_image, zero_watermark_image.shape)
    recovered_scrambled_logo = np.bitwise_xor(zero_watermark_image, logo_image_resized)
    recovered_logo = inverse_cat_map(recovered_scrambled_logo, key)
    similarity = np.sum(recovered_logo == logo_image_resized) / logo_image_resized.size
    return similarity

def search_via_validation_contract(user_info):
    tx = image_validation_contract.functions.search(user_info).build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
    })
    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"Transaction hash for search: {tx_hash.hex()}")
    return tx_hash

# Main process for IU
if __name__ == "__main__":
    # Define the original image and logo
    original_image = cv2.imread("superset final photo.png", cv2.IMREAD_GRAYSCALE)
    logo_image = cv2.imread("logo_image.png", cv2.IMREAD_GRAYSCALE)

    # Search contract for IPFS address and scrambling parameters
    user_info = "user_info_example"
    tx_hash = search_via_validation_contract(user_info)  # Call search
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    # Retrieve IPFS address and scrambling params from the contract
    ipfs_address, scrambling_params = image_validation_contract.functions.getResult(user_info).call()
    watermark_data = retrieve_ipfs_data(ipfs_address)
    zero_watermark_image = np.array(watermark_data["zero_watermark"])
    scrambling_key = int(watermark_data["scrambling_key"])

    # Validate the watermark
    similarity_score = validate_zero_watermark(
        original_image, zero_watermark_image, logo_image, scrambling_key
    )
    print(f"Similarity score: {similarity_score}")
    if similarity_score > 0.8:
        print("Copyright authentication successful.")
    else:
        print("Copyright authentication failed.")
