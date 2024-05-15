
timestamp=1622498401
folderpath=${SCRATCH}/${timestamp}
source /scratch/snx3000/sdennis/eco_pomati/Plankiformer_OOD/venv/bin/activate
filepath=$(printf ${folderpath}'/images')
predpath=./preds
cd /scratch/snx3000/bkch/plankiformer_minimal

python predict.py --test_path $filepath  --test_outpath $predpath --main_param_path ./params/ --model_path /scratch/snx3000/sdennis/eco_pomati/Plankiformer_OOD/trained_BEiT_models_Zoo/trained_models/01/ /scratch/snx3000/sdennis/eco_pomati/Plankiformer_OOD/trained_BEiT_models_Zoo/trained_models/02/ /scratch/snx3000/sdennis/eco_pomati/Plankiformer_OOD/trained_BEiT_models_Zoo/trained_models/03/ --ensemble 2 --finetuned 1 --threshold 0.0 --resize_images 1 --use_gpu --TTA

deactivate

# !# /bin/bash

# timestamp=1622498401
# directories=${SCRATCH}/${timestamp}/0p5xMAG/${month}.list
# folderpath=${SCRATCH}/eco_pomati/0p5xMAG/${month}
# folder=$(cat $directories)
# source ${SCRATCH}/eco_pomati/Plankiformer_OOD/venv/bin/activate

# #cycle through folders (unix timestamps)
# for i in ${folder}
# do
#     if [ ! -e ${folderpath}'/'${i}'/Ensemble_models_Plankiformer_OOD_predictions_geo_mean_tuned.txt' ] ; then

#         filepath=$(printf ${folderpath}'/'${i}'/images')
#         predpath=$(printf ${folderpath}'/'${i})
#         cd /scratch/snx3000/sdennis/eco_pomati/Plankiformer_OOD

#         python predict.py -test_path $filepath  -test_outpath $predpath -main_param_path ./trained_BEiT_models_Zoo/ -model_path ./trained_BEiT_models_Zoo/trained_models/01/ ./trained_BEiT_models_Zoo/trained_models/02/ ./trained_BEiT_models_Zoo/trained_models/03/ -ensemble 2 -finetuned 1 -threshold 0.0 -resize_images 1 -use_gpu yes -TTA yes

# fi


# done
# deactivate