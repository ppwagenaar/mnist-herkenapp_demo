import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Pagina instellingen
st.set_page_config(page_title="Cijferherkenning", layout="centered")
st.title("✍️ Teken een cijfer (0–9)")

# Model laden
col1,col2 = st.columns(2)

# Canvas instellingen
canvas_size = 280  # 10x upscale voor tekenen
canvas = col1.st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    width=canvas_size,
    height=canvas_size,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_image(img):
    img = img.resize((28, 28))
    img = img.convert("L")  # grayscale
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

if st.button("🔍 Voorspel"):
    if canvas.image_data is not None:
        img = Image.fromarray(canvas.image_data.astype("uint8"))
        processed_img = preprocess_image(img)


        col2.st.subheader(f"👉 Voorspelling: **2**")
        col2.st.write(f"Zekerheid: **50%**")

        col2.st.image(img.resize((28, 28)), caption="Jouw tekening (28×28)")
    else:
        st.warning("Teken eerst een cijfer!")
