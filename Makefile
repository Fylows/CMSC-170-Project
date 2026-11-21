app = app.py

all:build

dev:
	streamlit run $(app) --server.runOnSave true

build:
	streamlit run $(app)