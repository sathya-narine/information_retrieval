# app.py

import streamlit as st
from Search_Algorithm import search_engine

# Set the page configuration
st.set_page_config(
    page_title="Simple Information Retrieval System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Dataset")
if st.sidebar.button("CISI"):
    pass  # The button doesn't perform any action

# Main content
st.title("Simple Information Retrieval System")

st.subheader("Search")
query = st.text_input("Enter your query:", "")
expand_query_flag = st.checkbox("Query Expansion")

algorithm = st.radio("Algorithm", ("BM25", "TF-IDF"))

if st.button("Search"):
    if query == "":
        st.error("Please enter a query.")
    else:
        algorithm_choice = 1 if algorithm == "BM25" else 2
        results = search_engine(
            algorithm=algorithm_choice,
            query=query,
            expand_query_flag=expand_query_flag
        )

        st.subheader("Search Results")
        if not results:
            st.write("No results found.")
        else:
            for doc_id, content in results:
                st.write(f"**Document ID: {doc_id}**")
                st.write(content[:500] + "...")
                st.markdown(f"[Read More](#{doc_id})")

else:
    st.subheader("Search Results")
    st.write("Search results will be displayed here.")
