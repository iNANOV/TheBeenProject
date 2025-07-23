import streamlit as st


research_description = """
Welcome to the Research Hub of The Been Project.

At the core of The Been Project lies a rigorous research effort focused on developing advanced Positive Trading Strategies Classification (PTSC) models. Our work harnesses the power of machine learning and statistical optimization to identify and classify promising trade opportunities in complex financial markets.

This page serves as a gateway to explore the methodologies, experiments, and insights that drive the PTSC engine. Here, you’ll find detailed documentation on model architectures, optimization techniques, performance evaluations, and ongoing research initiatives aimed at continuously enhancing our predictive capabilities.

Our mission is to blend cutting-edge data science with practical trading applications, building a self-optimizing system that adapts and improves over time. Whether you’re a fellow researcher, data scientist, or enthusiast, we invite you to dive into the science behind the scenes and join us on this exciting journey.
"""


def show_research_page():
    st.title("Research")
    st.markdown(research_description)
    st.write("Research snapshot")
