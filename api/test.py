import requests
import os
from PIL import Image
import matplotlib.pyplot as plt

def send_image_to_api(image_path, url):
    with open(image_path, 'rb') as file:
        files = {'file': (os.path.basename(image_path), file, 'image/png')}
        response = requests.post(url, files=files)
        return response.json()

def display_image_with_response(image_path, response):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')  
    plt.show()
    print(f'Response for {os.path.basename(image_path)}: {response}')

def main():
    api_url = 'http://127.0.0.1:8000/predict/'
    image_folder = '../imgs/'

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if image_path.lower().endswith('.png'):
            response = send_image_to_api(image_path, api_url)
            display_image_with_response(image_path, response)

if __name__ == "__main__":
    main()
