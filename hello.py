import streamlit as st
import pandas as pd

st.write("""
# Hello!
Hello World!
""")

df = pd.DataFrame({'a': [1,2,3,3,4,4,3,2,2,3,4,2,2,4,2,34,3,2]})
st.line_chart(df)
