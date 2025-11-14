import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time

#  Neural Style Transfer — Streamlit Frontend + Core Logic

st.set_page_config(page_title="Neural Style Transfer 🎨", page_icon="🎨")
st.title("🎨 Neural Style Transfer App")
st.write(
    """
    Upload a **content image** (like a photo) and a **style image** (like a painting), 
    and this app will blend them together — recreating your photo *painted in the style* of the artwork.
    """
)

# Utility Functions

def load_image(uploaded_file, max_dim=512):
    img = Image.open(uploaded_file).convert("RGB")
    scale = max_dim / max(img.size)
    new_size = tuple((np.array(img.size) * scale).astype(int))
    img = img.resize(new_size)
    img = np.array(img)
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = img[tf.newaxis, ...] / 255.0
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

# Core Deep Learning Components

def vgg_layers(layer_names):
    """Load VGG19 and return a model that outputs intermediate layer activations."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]
        content_dict = {name: value
                        for name, value in zip(self.content_layers, content_outputs)}
        style_dict = {name: value
                      for name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

# Style Transfer Function (⚠️ no caching of tensors)
@st.cache_resource
def load_extractor():
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layers = ['block5_conv2']
    return StyleContentModel(style_layers, content_layers), style_layers, content_layers

def run_style_transfer(content_img, style_img, epochs=400, style_weight=1e-2, content_weight=1e4):
    extractor, style_layers, content_layers = load_extractor()
    style_targets = extractor(style_img)['style']
    content_targets = extractor(content_img)['content']

    image = tf.Variable(content_img)
    opt = tf.keras.optimizers.Adam(learning_rate=0.02)

    @tf.function
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            style_loss = tf.add_n([
                tf.reduce_mean((outputs['style'][name] - style_targets[name]) ** 2)
                for name in outputs['style'].keys()
            ])
            style_loss *= style_weight / len(style_layers)
            content_loss = tf.add_n([
                tf.reduce_mean((outputs['content'][name] - content_targets[name]) ** 2)
                for name in outputs['content'].keys()
            ])
            content_loss *= content_weight / len(content_layers)
            total_loss = style_loss + content_loss + 30 * tf.image.total_variation(image)

        grad = tape.gradient(total_loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, 0.0, 1.0))
        return total_loss

    for n in range(epochs):
        loss = train_step(image)
        if n % 50 == 0:
            print(f"Step {n}, Loss: {float(np.mean(loss.numpy())):.4e}")


    return image

# Streamlit UI Logic

content_file = st.file_uploader("📸 Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("🖌️ Upload Style Image", type=["jpg", "jpeg", "png"])

if content_file and style_file:
    st.image([content_file, style_file], caption=["Content Image", "Style Image"], width=300)
    if st.button("✨ Generate Stylized Image"):
        with st.spinner("Transforming your photo... 🪄"):
            content_image = load_image(content_file)
            style_image = load_image(style_file)
            start = time.time()
            stylized_image = run_style_transfer(content_image, style_image)
            end = time.time()

        result_image = tensor_to_image(stylized_image)
        st.image(result_image, caption="Stylized Output", use_container_width=True)
        st.success(f"✅ Done in {end - start:.2f} seconds!")

        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        st.download_button(
            "💾 Download Stylized Image",
            data=buf.getvalue(),
            file_name="stylized.png",
            mime="image/png"
        )


# Educational Info Panel

with st.expander(" Learn the Science Behind It"):
    st.markdown("""
    **Neural Style Transfer (NST)** merges the *content* of one image with the *style* of another.
    
    - **Feature Extractor:** Uses pre-trained **VGG19** to extract deep features.
    - **Content Loss:** Ensures generated image preserves the subject and layout.
    - **Style Loss (Gram Matrix):** Measures texture and color relationships.
    - **Total Variation Loss:** Adds smoothness and reduces noise.
    - **Optimizer:** **Adam**, trained on the image itself via gradient updates.
    """)
    st.info("Model: VGG19 | Framework: TensorFlow 2.x | Optimizer: Adam | Loss: Custom (Style + Content + TV)")

