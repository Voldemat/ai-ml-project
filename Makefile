include .env
export
download-dataset:
	kaggle datasets download jcoral02/inriaperson -p ./dataset --unzip
