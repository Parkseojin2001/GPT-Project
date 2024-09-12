import streamlit as st
from langchain.prompts import PromptTemplate


st.write("hello")

st.write([1, 2, 3, 4, 5])

d = {"x": 1}

st.write(PromptTemplate)

p = PromptTemplate.from_template("xxxx")

# magic(st.write없이도 화면에 출력)
d

p


st.selectbox(
    "Choose your model",
    ("GPT-3", "GPT-4"),
)
