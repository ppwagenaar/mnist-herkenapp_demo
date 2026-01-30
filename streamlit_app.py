import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as cp
import time
import math
from numpy.lib.stride_tricks import sliding_window_view
import json
class neuron:
    def __init__(self,neuronen_vorige_laag:int,laatste_laag=False):
        self.laatste_laag=laatste_laag
        scale = math.sqrt(2.0 / neuronen_vorige_laag)
        self.wegingen = cp.random.randn(neuronen_vorige_laag, 1).astype(cp.float32) * scale
        self.bias = 0.0
        # waardes voor de optimalisatiefunctie (Adam)
        self.m_w = cp.zeros_like(self.wegingen)
        self.v_w = cp.zeros_like(self.wegingen)
        self.m_b=0
        self.v_b=0
        self.t=0
    def activatie_functie(self,x,kale_waardes=None):
        if not self.laatste_laag:
            return (x+abs(x))/2 #ReLU
        else:
            # exp_x=cp.exp(kale_waardes-cp.max(kale_waardes))
            # return exp_x / cp.sum(exp_x) #softmax
            # return math.e**x/np.sum(cp.exp(kale_waardes)) #softmax
            return x #softmax voor alle lagen in 1x
    def bereken_output_kaal(self,neuronen_vorige_laag):
        self.neuronen_vorige_laag=neuronen_vorige_laag
        product=cp.dot(self.neuronen_vorige_laag,self.wegingen)
        self.output_kaal=cp.add(product,self.bias)#/(len(neuronen_vorige_laag)+1) ###

        return self.output_kaal
    def bereken_output_bewerkt(self,kale_waardes):
        product=cp.dot(self.neuronen_vorige_laag,self.wegingen)
        self.output_bewerkt=self.activatie_functie(self.bereken_output_kaal(self.neuronen_vorige_laag),kale_waardes) #product v.d. matrix

        return self.output_bewerkt
    def bereken_output(self, vorige_input):
        self.neuronen_vorige_laag = vorige_input
        product = cp.dot(self.neuronen_vorige_laag, self.wegingen)
        out = cp.add(product, self.bias)# / (len(vorige_input)+1)
        if not self.laatste_laag:
            return (out + cp.abs(out)) / 2  # ReLU
        else:
            return out  # wordt buiten de neuronenclass berekend

    def to_dict(self):
        return {
            'wegingen': self.wegingen.tolist(),
            'bias': self.bias,}
    def from_dict(self, data):
        self.wegingen = cp.array(data['wegingen'])
        self.bias = data['bias']
lagen=2
neuronen_per_laag=[256,256,10]
neuronen={}
for x in range(len(neuronen_per_laag)):
    neuronen[f'laag:{x+1}'] = []
for i, aantal in enumerate(neuronen_per_laag):
    if i==0:
        neuronen_vorige_laag=180
    else:
        neuronen_vorige_laag=neuronen_per_laag[i-1]
    for _ in range(aantal):
        if aantal==neuronen_per_laag[-1]: #laatste laag
            laatste_laag=True
        else:
            laatste_laag=False
        neuronen[f'laag:{i+1}'].append(neuron(neuronen_vorige_laag,laatste_laag=laatste_laag))
def im2col(x, kh, kw):
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    out_h = H - kh + 1
    out_w = W - kw + 1

    cols = np.zeros((B, C, kh, kw, out_h, out_w))
    for y in range(kh):
        for x_ in range(kw):
            cols[:, :, y, x_, :, :] = x[:, :, y:y+out_h, x_:x_+out_w]

    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(B*out_h*out_w, -1)
class convolutionele_laag:
    def __init__(self, input_vorm, aantal_kernels, kernel_grootte):
        self.C, self.H, self.W = input_vorm
        self.K = aantal_kernels
        self.kh = self.kw = kernel_grootte

        self.out_h = self.H - self.kh + 1
        self.out_w = self.W - self.kw + 1
        self.output_vorm = (self.K, self.out_h, self.out_w)

        self.W_kernels = np.random.randn(self.K, self.C * self.kh * self.kw)
        self.bias = np.zeros(self.K)

    def forwardprop(self, x):
        # x: (B, C, H, W)
        self.x_shape = x.shape
        self.cols = im2col(x, self.kh, self.kw)

        out = self.cols @ self.W_kernels.T + self.bias
        out = out.reshape(-1, self.out_h, self.out_w, self.K)

        return out.transpose(0, 3, 1, 2)

    def to_dict(self):
        return {
            'W_kernels': self.W_kernels.tolist(),
            'bias': self.bias.tolist()}
    def from_dict(self, data):
        self.W_kernels = cp.array(data['W_kernels'])
        self.bias = data['bias']
class pooling_laag:
    def __init__(self, input_vorm, pool_grootte):
        self.C, self.H, self.W = input_vorm
        self.p = pool_grootte

        self.out_h = self.H // self.p
        self.out_w = self.W // self.p
        self.output_vorm = (self.C, self.out_h, self.out_w)

    def forwardprop(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        self.x = x

        x_reshaped = x.reshape(
            B, C,
            self.out_h, self.p,
            self.out_w, self.p
        )

        self.mask = (x_reshaped == x_reshaped.max(axis=(3, 5), keepdims=True))
        return x_reshaped.max(axis=(3, 5))
def softmax_over_laag(z):
    # z vorm: (n_neuronen, batch)
    z_stable = z - cp.max(z, axis=0, keepdims=True)    # -C waarde om geen overflow te krijgen er verandert wiskundig gezien niks
    exp_z = cp.exp(z_stable)
    return exp_z / cp.sum(exp_z, axis=0, keepdims=True)
convolutionele_lagen=[]
convolutionele_lagen.append(convolutionele_laag((1,28,28), 5, 3))
convolutionele_lagen.append(pooling_laag(convolutionele_lagen[0].output_vorm,2))
convolutionele_lagen.append(convolutionele_laag(convolutionele_lagen[1].output_vorm, 5, 2))
convolutionele_lagen.append(pooling_laag(convolutionele_lagen[2].output_vorm,2))
with open("data.json") as f:
    data = json.load(f)

# dense
idx = 0
for i in range(len(neuronen_per_laag)):
    for j in range(neuronen_per_laag[i]):
        neuronen[f'laag:{i+1}'][j].from_dict(data['neuronen'][idx])
        idx += 1

# convolutioneel
conv_idx = 0
for laag in convolutionele_lagen:
    if isinstance(laag, convolutionele_laag):
        laag.from_dict(data['convolutionele'][conv_idx])
        conv_idx += 1
def voorspel_enkel_plaatje(afb, convolutionele_lagen, neuronen, neuronen_per_laag):

    x = afb[np.newaxis, np.newaxis, :, :]  # (1,1,28,28)

    # CNN
    for laag in convolutionele_lagen:
        x = laag.forwardprop(x)

    x = x.reshape(1, -1)
    x /= np.std(x) + 1e-6

    # Dense layers
    input = cp.array(x)
    for i in range(len(neuronen_per_laag)):
        outputs = []
        for n in neuronen[f'laag:{i+1}']:
            outputs.append(n.bereken_output(input))
        input = cp.hstack(outputs)

        if i < len(neuronen_per_laag) - 1:
            input = (input + cp.abs(input)) / 2  # ReLU

    # Softmax
    kans = softmax_over_laag(input.T)
    voorspelling = int(cp.argmax(kans))

    return voorspelling, kans

# Pagina instellingen
col1,col2 = st.columns(2)
st.set_page_config(page_title="Cijferherkenning", layout="centered")
col1.title("✍️ Teken een cijfer (0–9)")

# Model laden

# Canvas instellingen
canvas_size = 280  # 10x upscale voor tekenen
canvas = st_canvas(
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
    return img_array

if st.button("🔍 Voorspel"):
    if canvas.image_data is not None:
        img = Image.fromarray(canvas.image_data.astype("uint8"))
        processed_img = preprocess_image(img)
        getal, kans = voorspel_enkel_plaatje(processed_img, convolutionele_lagen,neuronen,neuronen_per_laag)


        col2.subheader(f"👉 Voorspelling: **{getal}**")
        col2.write(f"Zekerheid: **{kans[getal[0]]*100}%**")

        col2.image(img.resize((28, 28)), caption="Jouw tekening (28×28)")
    else:
        st.warning("Teken eerst een cijfer!")
