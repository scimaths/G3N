cd src

# echo "Running isomorphism experiments"
# for dataset in 'exp' 'sr25' 'graph8c' 'csl'; do
#     for t in 2 3; do
#         for d in 1 2; do
#             echo "----Running isomorphism for ${dataset} with t=${t}, d=${d}----"
#             CUDA_VISIBLE_DEVICES=3 python3 iso.py --dataset $dataset --t $t --d $d > ../logs/isomorphism/${dataset}_t_${t}_d_${d} &
#         done
#     done
# done

# echo "Substructure counting experiments"
# for ntask in 3; do
#     for t in 2 3; do
#         for d in 1 2; do
#             echo "----Running substructure counting for ${ntask} with t=${t}, d=${d}----"
#             CUDA_VISIBLE_DEVICES=${ntask} python3 counting.py --ntask $ntask --t $t --d $d | tee ../logs/substructure_counting/task_${ntask}_t_${t}_d_${d}
#         done
#     done
# done

# echo "Classification experiments on TU Datasets"
# for dataset in 'PTC_MR' 'MUTAG' 'NCI1' 'PROTEINS' 'IMDB-BINARY' 'IMDB-MULTI'; do
#     echo "----Running classification experiments for ${dataset} with t,d = (1,1) & (2,2)"
#     python3 grid_tu.py --t 1 --d 1 --dataset ${dataset} | tee ../logs/tu_classification/${dataset}_t_1_d_1 &
#     python3 grid_tu.py --t 2 --d 2 --dataset ${dataset} | tee ../logs/tu_classification/${dataset}_t_2_d_2 &
# done

# echo "Experiments with ZINC"
# CUDA_VISIBLE_DEVICES=3 python3 zinc.py --t 2 --d 3 | tee ../logs/zinc/t_2_d_3

echo "Experiments with MolHIV"
CUDA_VISIBLE_DEVICES=5 python3 mol.py --t 2 --d 3 | tee ../logs/molhiv/t_2_d_3