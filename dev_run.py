import importlib
import sys

import streamlit as st

from common.ui import demo_banner, init_state


st.set_page_config(page_title="SIRPS dev", layout="wide")
init_state()
demo_banner()
page = sys.argv[-1] if len(sys.argv) > 1 else "home"
importlib.import_module(f"modules.{page}").render()
