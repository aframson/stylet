import streamlit as st
import coremltools as ct
from PIL import Image


mlmodel = ct.models.MLModel('style1.mlmodel')


# Get the spec from the MLModel
spec = mlmodel.get_spec()

file = st.file_uploader('upload image')

st.image(file)

im = Image.open(file)
newi = im.resize((512,512))
# Call predict.
output_dict = mlmodel.predict({'image':newi})
if file:
    st.image(output_dict['stylizedImage'])