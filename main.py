import streamlit as st 

st.title("Wel-Come to Streamlit Demo Application")
st.header("Streamlit is python library")
st.subheader("Streamlit is very easy to start")
st.markdown("Just a new line, to show a markdown text..!")

name = st.text_input("Enter Name : ")
fname = st.text_input("Enter Father Name : ")
adr = st.text_area("Enter Your Text : ")
classdata = st.selectbox("Select Your Option : ",(1,2,3,4,5,6))
button = st.button("Submit")
if button :
    st.markdown(f""" 
    Name: {name}
    Father Name : {fname}
    Your Text : {adr}
    Your Option : {classdata}""")
