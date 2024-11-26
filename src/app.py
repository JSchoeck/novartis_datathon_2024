import pandas as pd
import streamlit as st
from streamlit import session_state as ss

import utils


def init() -> None:
    ss.settings = utils.load_settings()


def main() -> None:
    st.title("Streamlit web app")
    st.write("Web app for Novartis datathon.")

    st.write(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))

    if "load_data" in ss:
        st.markdown("## Data")
        st.write(ss.data_input)

    with st.sidebar:
        st.write("This is the sidebar.")
        ss.load_data = st.button("Load data", on_click=utils.load_data)


if __name__ == "__main__":
    main()
