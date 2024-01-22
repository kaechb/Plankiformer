## Plankiformer_OOD

### Repo Structure

Public repository for Zoolake2.0 image classification in Out-of-Dataset (OOD) scenarios.

The repo contains the following directories:

- `utils`: contains auxiliary code for training and testing.

- `utils_anlysis`: contains auxiliary code for data and result analysis.


### Training models

In order to train a fresh model, use `main.py`. 

You can train the model using:

```python
python3 main.py -datapaths $path/of/data/ -outpath $path/to/save/ -epochs 50 -finetune 1 -finetune_epochs 50 \
-batch_size 128 -init_name Init_01 -architecture beit -add_layer no \
-last_layer_finetune yes -run_lr_scheduler no -run_early_stopping no \
-resume_from_saved no -lr 1e-4 -finetune_lr 1e-5 -weight_decay 0.03 -dropout_1 0 -dropout_2 0 \
-fc_node 0 -balance_weight yes -save_best_model_on_loss_or_f1_or_accuracy 2 \
-warmup 0 -classifier multi -aug -aug_type medium -datakind image -ttkind image -save_data yes \
-resize_images 1 -L 224 -valid_set yes -test_set yes -dataset_name zoolake -training_data False \
-run_cnn_or_on_colab yes -use_gpu yes
```
There are lots of input commands that can be given to the script. To query them, use the `-h` flag (`python main.py -h`). 

The ZooLake2.0 dataset can be downloaded [here](https://doi.org/10.25678/000C6M).