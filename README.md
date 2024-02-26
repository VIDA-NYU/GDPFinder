# Granularity at Scale

Paper: [Granularity at Scale: Estimating Neighborhood Socioeconomic Indicators from High-Resolution Orthographic Imagery and Hybrid Learning](https://doi.org/10.1109/JSTARS.2024.3368018)

## For the supervised approach:

```notebooks/supervised_approach.ipynb``` contains data (census and satellite imagery) preparation and analysis. Cells to test saved models and interpret results are also there.

### Example usage to train a model:

Within ```scripts``` directory:

```
$ nohup python -u supervised_training.py --metric 'density' --imagetype 'resize' --newwidth 1234 --newheight 1234 &
```

See ```supervised_training.py``` for more details on the arguments and training process.

In ```supervised_approach.ipynb``` and ```supervised_training.py```, the dataset is generated from ```scripts/create_dataset.py``` and the model from ```scripts/supervised_models.py```
