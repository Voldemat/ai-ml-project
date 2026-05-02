include .env
export
download-dataset:
	#kaggle datasets download jcoral02/inriaperson -p ./datasets/inria-person --unzip
	kaggle datasets download tejasvdante/pedestrian-no-pedestrian -p ./datasets/pedestrian-no-pedestrian --unzip
strip-jupyter-notebooks:
	nbstripout *.ipynb
