# UrbanSat (GDPFinder)

## For the supervised approach:

/notebooks/supervised_approach.ipynb contains data (census and satellite imagery) preparation and analysis.

/saved_models/[metric]/[metric]_saved_model_info.txt contain information about the model runs saved in the directory.


### Example usage to train a model:
nohup python -u scripts/supervised_training.py --metric ['density', 'mhi', or 'ed'] --imagetype ['patches' or 'resized']

If images are to be resized, specify additional integer parameters --newwidth and --newheight  
Optional argument --fconly, defaults to True to train only fully-connected layers. False trains FC then all layers.
Optional arguement --batchsize, defaults to 8