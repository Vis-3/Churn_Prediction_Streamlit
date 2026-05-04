# Entry point for Streamlit Community Cloud.
# The actual app lives in src/dashboard.py — this shim lets Streamlit Cloud
# find it when the file is configured as the main app path.
import runpy, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
runpy.run_path(os.path.join(os.path.dirname(__file__), "src", "dashboard.py"), run_name="__main__")
