# deep-learning

## Folder Structure

```
- dataset: dataset file location
- outputs: contains csv output & other output from execution
- visualization: contains visualization output
- src
    - finetune_data_cleaning
    - utils: various utility classes & scripts
        - config_utils.py: class to help with configuration
        - dataloader.py: class for pytorch data loader 
        - error_analysis.py: python script to run error analysis
        - file_converter.py: python script to clean up unreadable files & transform files to jpg
        - metrics.py: class for evaluation metrics
        - model_loading.py: class to simplify model loading. Support SPSL, CLIP, Xception
    - visualizers: various utility related to visualization
```

## Inference
To run inference, first update src/config.yml  
- `model_type`: model name `spsl|clip|xception`
- `weights_path`: full path to model weights
- `data_root_dir`: full path to the data directory containing images. need to contain `real` and `fake` folder

After the config is set, run the script from the root folder i.e. `python src/inference.py`

## Visualization
Update all the path to weights and viz folder
After the config is set, run the script from the root folder i.e. `python src/generate_viz.py`

