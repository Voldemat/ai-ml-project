# Pedestrian Detection using HOG Features + SVM (Classical Object Detection)

## Installing dependencies
```bash
uv sync
source .venv/bin/activate
```

## Installing dataset
Make an account on Kaggle, obtain API token from settings and save it in .env file, like so:
```
KAGGLE_API_TOKEN=<your-token>
```
Then run command that will install dataset
```bash
make download-dataset
```

## Start jupyter lab
If you want to explore various jupyter notebooks in repo, start jupyter lab in root of repository:
```bash
jupyter lab
```
Make sure you run this command after activating virtual environment(
```
source .venv/bin/activate
```
)
