# Arguments - directoryname to store results(arg1) l1 se l7 number of runs(arg9) dataset_name(arg10) 
#no_of_lf_classes(arg11) argument12->feature_based_model('lr' or 'nn') argument13->0/1(qg available or not)
# batch_size(arg14) lr_fnetwork (arg15) lr_gm (arg16)


#L146
# exp_name='exp/imdb/imdb_exp/l146'
# mkdir -p $exp_name
# CUDA_VISIBLE_DEVICES=1 python3 ss_generic.py $exp_name l1 0 0 l4 0 l6 0 5 dsets/imdb_original 2 lr 0 32 0.00003 0.01 > $exp_name/res.txt

# #l1436
# exp_name='exp/imdb/imdb_exp/l1346'
# mkdir -p $exp_name
# CUDA_VISIBLE_DEVICES=1 python3 ss_generic.py $exp_name l1 0 l3 l4 0 l6 0 5 dsets/imdb_original 2 lr 0 32 0.0003 0.01 > $exp_name/res.txt

# # l1246
# exp_name='exp/imdb/imdb_exp/l1246'
# mkdir -p $exp_name
# CUDA_VISIBLE_DEVICES=1 python3 ss_generic.py $exp_name l1 l2 0 l4 0 l6 0 5 dsets/imdb_original 2 lr 0 32 0.0003 0.01 > $exp_name/res.txt

# # l12346
# exp_name='exp/imdb/imdb_exp/l12346'
# mkdir -p $exp_name
# CUDA_VISIBLE_DEVICES=1 python3 ss_generic.py $exp_name l1 l2 l3 l4 0 l6 0 5 dsets/imdb_original 2 lr 0 32 0.0003 0.01 > $exp_name/res.txt

# l13456
exp_name='exp/imdb_reef/imdb_exp/l13456'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=1 python3 ss_generic.py $exp_name l1 0 l3 l4 l5 l6 0 5 reef/generated_data/imdb 2 lr 0 32 0.0003 0.01 > $exp_name/res.txt

# # l13456qg
# exp_name='exp/imdb/imdb_exp/l13456qg'
# mkdir -p $exp_name
# CUDA_VISIBLE_DEVICES=1 python3 ss_generic.py $exp_name l1 0 l3 l4 l5 l6 qg 5 dsets/imdb_original 2 lr 0 32 0.0003 0.01 > $exp_name/res.txt

# l123456
# exp_name='exp/imdb/imdb_exp/l123456'
# mkdir -p $exp_name
# CUDA_VISIBLE_DEVICES=1 python3 ss_generic.py $exp_name l1 l2 l3 l4 l5 l6 0 5 dsets/imdb_original 2 lr 0 32 0.0003 0.01 > $exp_name/res.txt

# # l123456qg
# exp_name='exp/imdb/imdb_exp/l123456qg'
# mkdir -p $exp_name
# CUDA_VISIBLE_DEVICES=1 python3 ss_generic.py $exp_name l1 l2 l3 l4 l5 l6 qg 5 dsets/imdb_original 2 lr 0 32 0.0003 0.01 > $exp_name/res.txt
