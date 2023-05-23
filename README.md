# GDPFinder

## For the supervised approach,

/notebooks/supervised_approach.ipynb contains data (census and satellite imagery) preparation and analysis.

/saved_models/saved_model_info.txt contains information about the models runs saved in the directory.


### Example usage to train a model:
nohup python -u scripts/supervised_training.py --imagetype ['patches' or 'resized']

If images are to be resized, specify additional integer parameters --newwidth and --newheight
Optional arguement --batchsize