import cv2
import numpy as np
from web3 import Web3
import requests
import json
import os
from dotenv import load_dotenv
import io
from PIL import Image
import base64
from skimage.metrics import structural_similarity as ssim

# Load environment variables
load_dotenv()

INFURA_PROJECT_ID = os.getenv('INFURA_PROJECT_ID')
PRIVATE_KEY = os.getenv('PRIVATE_KEY_2')
IMAGE_VALIDATION_ADDRESS = os.getenv('IMAGE_VALIDATION_ADDRESS')

# Load ABIs
with open('abi.json') as f:
    abi_data = json.load(f)

image_validation_abi = abi_data["ImageValidation"]
image_depository_abi = abi_data["ImageDepository"]

# Connect to Ethereum network (Sepolia testnet)
w3 = Web3(Web3.HTTPProvider(f'https://sepolia.infura.io/v3/{INFURA_PROJECT_ID}'))

# Instantiate the ImageValidation contract
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
def retrieve_ipfs_data(ipfs_hash, gateway="https://gateway.pinata.cloud"):
    url = f"{gateway}/ipfs/{ipfs_hash}"
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        print(f"HTTP Status Code: {response.status_code}")
        print(f"Response Content Type: {response.headers.get('Content-Type')}")
        
        if "application/json" in response.headers.get("Content-Type", ""):
            return response.json()  # Return JSON data
        else:
            print("Non-JSON response received. Returning raw content.")
            return response.content  # Return raw bytes for non-JSON files
    except Exception as e:
        raise Exception(f"Failed to retrieve from IPFS: {e}")

from skimage.metrics import structural_similarity as ssim
import numpy as np

def validate_zero_watermark(original_image, zero_watermark_image, logo_image, key):
    # Resize the logo to match the size of the zero watermark image
    logo_image_resized = resize_logo(logo_image, zero_watermark_image.shape)
    
    # Recover the scrambled logo
    recovered_scrambled_logo = np.bitwise_xor(zero_watermark_image, logo_image_resized)
    
    # Decode the scrambled logo using the inverse cat map and key
    recovered_logo = inverse_cat_map(recovered_scrambled_logo, key)
    """
    # Compute SSIM for similarity
    win_size = min(logo_image_resized.shape[0], logo_image_resized.shape[1]) - 1
    if win_size % 2 == 0:
        win_size -= 1  # Ensure win_size is odd
    
    if win_size < 3:  # Handle very small images
        raise ValueError("Images are too small for SSIM computation with a meaningful window size.")
    
    # Ensure SSIM considers the valid range of pixel intensities
    similarity = ssim(
        logo_image_resized, 
        recovered_logo, 
        data_range=recovered_logo.max() - recovered_logo.min(),
        win_size=win_size
    )
    """
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
    # Debug: Verify Authorization
    print("Verifying authorization...")

    # Check repository address
    repository_address = image_validation_contract.functions.repository().call()
    print(f"Repository address: {repository_address}")

    # Instantiate ImageDepository contract to check `auth_user`
    image_depository_contract = w3.eth.contract(address=repository_address, abi=image_depository_abi)
    is_authorized = image_depository_contract.functions.auth_user(account.address).call()
    print(f"Is caller authorized in repository? {is_authorized}")

    if not is_authorized:
        raise Exception("Caller is not authorized to access repository functions.")

    # Define the original image and logo
    original_image = cv2.imread("original_image_2.png", cv2.IMREAD_GRAYSCALE)
    logo_image = cv2.imread("logo_image.png", cv2.IMREAD_GRAYSCALE)

    # Call search and wait for transaction receipt
    user_info = "count4"
    print("Calling search in ImageValidation contract...")
    tx_hash = search_via_validation_contract(user_info)  # Call search
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Search transaction receipt: {tx_receipt}")

    # Debug: Fetch results from ImageValidation contract
    print("Fetching results from ImageValidation contract...")
    try:
        ipfs_address, scrambling_params = image_validation_contract.functions.getResult(user_info).call({
            'from': account.address
        })
        print(f"IPFS Address: {ipfs_address}, Scrambling Params: {scrambling_params}")
    except Exception as e:
        print(f"Error in getResult: {e}")
        raise

    # Retrieve watermark data from IPFS
    # Retrieve watermark data from IPFS
    print(f"Retrieving watermark data from IPFS (hash: {ipfs_address})...")
    watermark_data = retrieve_ipfs_data(ipfs_address)

    # Process watermark data
    if isinstance(watermark_data, dict):
        # JSON data with base64-encoded image
        zero_watermark_base64 = watermark_data["zero_watermark"]
        scrambling_key = int(watermark_data["scrambling_key"])
        zero_watermark_image = cv2.imdecode(
            np.frombuffer(base64.b64decode(zero_watermark_base64), np.uint8),
            cv2.IMREAD_UNCHANGED
        )
        print(f"Decoded watermark image. Shape: {zero_watermark_image.shape}")
    elif isinstance(watermark_data, bytes):
        # Binary data
        try:
            # Attempt to decode as an image
            nparr = np.frombuffer(watermark_data, np.uint8)
            zero_watermark_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            print(f"Image decoded. Shape: {zero_watermark_image.shape}")
            
            # Handle scrambling key (hardcoded or fetched separately)
            scrambling_key = int(input("Enter the scrambling key (if available): "))
        except Exception as e:
            print(f"Failed to decode as image: {e}")
            # Save as raw binary for manual inspection
            with open("downloaded_watermark.raw", "wb") as f:
                f.write(watermark_data)
            raise Exception("Saved raw data for inspection. Could not process watermark image.")
    else:
        raise Exception("Unexpected data type received from IPFS.")
    
    # Save the retrieved watermark image as a PNG file
    output_filename = "retrieved_watermark.png"
    cv2.imwrite(output_filename, zero_watermark_image)
    print(f"Watermark image saved to {output_filename}")


    # Validate the watermark
    similarity_score = validate_zero_watermark(
        original_image=original_image, zero_watermark_image=zero_watermark_image, logo_image=logo_image, key=scrambling_key
    )
    print(f"Similarity score: {similarity_score}")
    if similarity_score > 0.8:
        print("Copyright authentication successful.")
    else:
        print("Copyright authentication failed.")
