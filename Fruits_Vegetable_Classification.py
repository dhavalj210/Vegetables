import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup
import cv2
import time
import cvlib as cv
from cvlib.object_detection import draw_bbox

model = load_model('FV.h5')


labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']
def fetch_price(vegetable_name):
    try:
        # Format the URL based on the vegetable name
        formatted_name = vegetable_name.lower().replace(" ", "-")
        url = f"https://www.oneindia.com/{formatted_name}-price-in-ahmedabad.html"

        # Headers to simulate a user request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

        # Make the request with headers and parse the HTML
        req = requests.get(url, headers=headers)
        soup = BeautifulSoup(req.text, 'lxml')

        # Extract the price from the HTML structure
        price_element = soup.find("div", class_="item-price-details")
        if price_element:
            # Extract the price text and remove any leading/trailing spaces
            price = price_element.text.strip()
            return price
        else:
            return f"Price not found for {vegetable_name} on the website."

    except Exception as e:
        return f"Error: {e}"

def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)

def fetch_weight(item_name):
    # Define a dictionary mapping each item to its average weight
    avg_weight = {
        'apple': '150 grams',
        'banana': '120 grams',
        'beetroot': '200 grams',
        'bell pepper': '150 grams',
        'cabbage': '1 kg',
        'capsicum': '100 grams',
        'carrot': '80 grams',
        'cauliflower': '500 grams',
        'chilli pepper': '50 grams',
        'corn': '100 grams',
        'cucumber': '300 grams',
        'eggplant': '250 grams',
        'garlic': '30 grams',
        'ginger': '50 grams',
        'grapes': '200 grams',
        'jalepeno': '25 grams',
        'kiwi': '100 grams',
        'lemon': '50 grams',
        'lettuce': '200 grams',
        'mango': '200 grams',
        'onion': '150 grams',
        'orange': '150 grams',
        'paprika': '100 grams',
        'pear': '200 grams',
        'peas': '100 grams',
        'pineapple': '900 grams',
        'pomegranate': '200 grams',
        'potato': '200 grams',
        'raddish': '50 grams',
        'soy beans': '150 grams',
        'spinach': '100 grams',
        'sweetcorn': '100 grams',
        'sweetpotato': '200 grams',
        'tomato': '100 grams',
        'turnip': '150 grams',
        'watermelon': '5 kg'
    }

    # Check if the item_name exists in the dictionary
    if item_name.lower() in avg_weight:
        return avg_weight[item_name.lower()]
    else:
        return f"Average weight not available for {item_name}"

def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


import streamlit as st
from PIL import Image
import cv2
import tempfile

def prepare_image(img):
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img


def run():
    st.markdown(
        """
        <style>
            body {
                background-image: url('ef411ae1-60de-4724-b2ad-ae56034ee130.jpg');
                background-size: cover;
            }
        </style>
        """
        , unsafe_allow_html=True)
    st.title("Fruits🍍-Vegetable🍅 Classification")
    option = st.radio(
        "Choose classification method:",
        ("Real-time Classification", "Image Upload")
    )
    if option == "Real-time Classification":
        stop_button = st.button('Stop')
        if stop_button:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = frame
                frame = cv2.resize(frame, (250, 250))

                st.image(frame, channels="RGB", use_column_width=False)

                pil_img = Image.fromarray(frame)

                prepared_img = prepare_image(pil_img)

                prediction = model.predict(prepared_img)
                predicted_class = labels[np.argmax(prediction)]

                if predicted_class in vegetables:
                    st.info('**Category : Vegetables**')
                else:
                    st.info('**Category : Fruit**')
                st.success("**Predicted : " + predicted_class.capitalize() + '**')

                box, label,c_score=cv.detect_common_objects(frame_rgb)
                output= draw_bbox(frame_rgb, box, label, c_score)
                st.write(f'Number of objects detected: {len(label)}')
                cal = fetch_calories(predicted_class)
                if cal:
                    st.warning('**' + cal + '(100 grams)**')
                price = fetch_price(predicted_class)
                if price:
                    st.info('*Price: ' + price + '*')
                weight = fetch_weight(predicted_class)
                if weight:
                    st.info('*Weight: ' + weight + '*')

            cap.release()
            cv2.destroyAllWindows()
    else:
        st.title("Image Upload")
        img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
        if img_file is not None:
            img = Image.open(img_file).resize((250, 250))
            st.image(img, use_column_width=False)
            save_image_path = './upload_images/' + img_file.name
            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())

            if img_file is not None:
                result = processed_img(save_image_path)
                print(result)
                if result in vegetables:
                    st.info('**Category : Vegetables**')
                else:
                    st.info('**Category : Fruit**')
                st.success("**Predicted : " + result + '**')
                cal = fetch_calories(result)
                if cal:
                    st.warning('**' + cal + '(100 grams)**')
                price = fetch_price(result)
                if price:
                    st.info('*Price: ' + price + '*')
                weight = fetch_weight(result)
                if weight:
                    st.info('*Weight: ' + weight + '*')
                im = np.array(img)
                box, label, c_score = cv.detect_common_objects(im)
                output = draw_bbox(im, box, label, c_score)
                st.write(f'Number of objects detected: {len(label)}')

if __name__ == "__main__":
    run()
