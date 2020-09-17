mkdir -p audit_sup
mkdir -p audit_sup/l134
CUDA_VISIBLE_DEVICES=0 python audit_sup_sub_selection.py audit_sup/l134 l1 0 l3 l4 0 0 0 5 datasets/audit > audit_sup/l134/res.txt
mkdir -p audit_sup/l1234
CUDA_VISIBLE_DEVICES=0 python audit_sup_sub_selection.py audit_sup/l1234 l1 l2 l3 l4 0 0 0 5 datasets/audit > audit_sup/l1234/res.txt
# mkdir -p audit_sup/l146
# CUDA_VISIBLE_DEVICES=2 python audit_sup_sub_selection.py audit_sup/l146 l1 0 0 l4 0 l6 0 5 datasets/ionosphere > audit_sup/l146/res.txt
# mkdir -p audit_sup/l1346
# CUDA_VISIBLE_DEVICES=2 python audit_sup_sub_selection.py audit_sup/l1346 l1 0 l3 l4 0 l6 0 5 datasets/ionosphere > audit_sup/l1346/res.txt
# mkdir -p audit_sup/l12346
# CUDA_VISIBLE_DEVICES=2 python audit_sup_sub_selection.py audit_sup/l123456 l1 l2 l3 l4 0 l6 0 5 datasets/ionosphere > audit_sup/l123456/res.txt
# mkdir -p audit_sup/l13456
# CUDA_VISIBLE_DEVICES=2 python audit_sup_sub_selection.py audit_sup/l13456 l1 0 l3 l4 l5 l6 0 5 datasets/ionosphere > audit_sup/l13456/res.txt
# mkdir -p audit_sup/l123456
# CUDA_VISIBLE_DEVICES=2 python audit_sup_sub_selection.py audit_sup/l123456 l1 l2 l3 l4 l5 l6 0 5 datasets/ionosphere > audit_sup/l123456/res.txt
# mkdir -p audit_sup/l123456qg
# CUDA_VISIBLE_DEVICES=2 python audit_sup_sub_selection.py audit_sup/l123456qg l1 l2 l3 l4 l5 l6 qg 5 datasets/ionosphere > audit_sup/l123456qg/res.txt
# mkdir -p audit_sup/l13456qg
# CUDA_VISIBLE_DEVICES=2 python audit_sup_sub_selection.py audit_sup/l13456qg l1 0 l3 l4 l5 l6 qg 5 datasets/ionosphere > audit_unsup/l13456qg/res.txt

