# Install enviroment
## 1. Create conda environment
```
conda create -n rfdetr python=3.12
```

## 2. Install requirement libraries
```
conda activate rfdetr
pip install -r requirements.txt
```

# Prepare training data
Prepare datasets folder including data and mask folders
## 1. Augment data
Augment data by cropping, rotating, flipping images and masks
```
python tools/aug.py --invert
```
- Parameter: --invert - use to invert mask of objects and background if mask of objects are black

## 2. Generate coco format data
Generate COCO data based on the augmented data. 
```
python tools/gen_coco.py --split train
```
- Parameter: --split - Choose **train**, **valid**, or **test** mode to split the data with percentages of 80%, 10%, and 10%, respectively.

# Train

```
python train.py
```
# Predict
```
python predict.py
```
