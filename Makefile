include .env
export
download-dataset:
	#kaggle datasets download jcoral02/inriaperson -p ./datasets/inria-person --unzip
	kaggle datasets download tejasvdante/pedestrian-no-pedestrian -p ./data/raw/pedestrian-no-pedestrian --unzip
	rm data/raw/pedestrian-no-pedestrian/data.rar
strip-jupyter-notebooks:
	nbstripout notebooks/*.ipynb
verify-jupyter-notebooks:
	nbstripout --verify notebooks/*.ipynb
