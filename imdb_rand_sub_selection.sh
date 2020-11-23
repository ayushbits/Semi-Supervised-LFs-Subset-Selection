mkdir -p imdb_rand
exp_name=mkdir -p imdb_rand/l13456qg
CUDA_VISIBLE_DEVICES=0 python audit_rand_sub_selection.py $exp_name l1 0 l3 l4 l5 l6 qg 5 datasets/imdb 2 > $exp_name/res.txt
#mkdir -p audit_rand/l1234i
#CUDA_VISIBLE_DEVICES=0 python audit_rand_sub_selection.py audit_rand/l1234 l1 l2 l3 l4 0 0 0 5 datasets/audit > audit_rand/l1234/res.txt
# mkdir -p audit_rand/l146
# CUDA_VISIBLE_DEVICES=2 python audit_rand_sub_selection.py audit_rand/l146 l1 0 0 l4 0 l6 0 5 datasets/ionosphere > audit_rand/l146/res.txt
# mkdir -p audit_rand/l1346
# CUDA_VISIBLE_DEVICES=2 python audit_rand_sub_selection.py audit_rand/l1346 l1 0 l3 l4 0 l6 0 5 datasets/ionosphere > audit_rand/l1346/res.txt
# mkdir -p audit_rand/l12346
# CUDA_VISIBLE_DEVICES=2 python audit_rand_sub_selection.py audit_rand/l123456 l1 l2 l3 l4 0 l6 0 5 datasets/ionosphere > audit_rand/l123456/res.txt
# mkdir -p audit_rand/l13456
# CUDA_VISIBLE_DEVICES=2 python audit_rand_sub_selection.py audit_rand/l13456 l1 0 l3 l4 l5 l6 0 5 datasets/ionosphere > audit_rand/l13456/res.txt
# mkdir -p audit_rand/l123456
# CUDA_VISIBLE_DEVICES=2 python audit_rand_sub_selection.py audit_rand/l123456 l1 l2 l3 l4 l5 l6 0 5 datasets/ionosphere > audit_rand/l123456/res.txt
# mkdir -p audit_rand/l123456qg
# CUDA_VISIBLE_DEVICES=2 python audit_rand_sub_selection.py audit_rand/l123456qg l1 l2 l3 l4 l5 l6 qg 5 datasets/ionosphere > audit_rand/l123456qg/res.txt
# mkdir -p audit_rand/l13456qg
# CUDA_VISIBLE_DEVICES=2 python audit_rand_sub_selection.py audit_rand/l13456qg l1 0 l3 l4 l5 l6 qg 5 datasets/ionosphere > audit_rand/l13456qg/res.txt

