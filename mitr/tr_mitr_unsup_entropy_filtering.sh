#L1
exp_name='mitr_entropy_unsup_entropy/l1'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 0 0 0 0 0 0 5 > $exp_name/res.txt
#L1+2
exp_name='mitr_entropy_unsup_entropy/l12'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 l2 0 0 0 0 0 5 > $exp_name/res.txt
#l1+l4+l3
exp_name='mitr_entropy_unsup_entropy/l143'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 0 l3 l4 0 0 0 5 > $exp_name/res.txt
#l1+l4 + l6
exp_name='mitr_entropy_unsup_entropy/l146'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 0 0 l4 0 l6 0 5 > $exp_name/res.txt
#l1 + l4 + l3 + l6
exp_name='mitr_entropy_unsup_entropy/l1436'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 0 l3 l4 0 l6 0 5 > $exp_name/res.txt
#l1 + l2 + l3 + l4 + l5 + l6
exp_name='mitr_entropy_unsup_entropy/l123456'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 l2 l3 l4 l5 l6 0 5 > $exp_name/res.txt
#l1 -6 + qg
exp_name='mitr_entropy_unsup_entropy/l123456qg'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 l2 l3 l4 l5 l6 qg 5 > $exp_name/res.txt

#l1 + l2 + l3+l4
exp_name='mitr_entropy_unsup_entropy/l1234'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 l2 l3 l4 0 0 0 5 > $exp_name/res.txt

#l1 + l2 + l4 + l6
exp_name='mitr_entropy_unsup_entropy/l1246'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 l2 0 l4 0 l6 0 5 > $exp_name/res.txt

#L1+2+3+4+6
exp_name='mitr_entropy_unsup_entropy/l12346'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 l2 l3 l4 0 l6 0 5 > $exp_name/res.txt
#l1+3+4+5+6+QG
exp_name='mitr_entropy_unsup_entropy/l13456qg'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 0 l3 l4 l5 l6 qg 5 > $exp_name/res.txt
#l1+3+4+5+6
exp_name='mitr_entropy_unsup_entropy/l13456'
mkdir -p $exp_name
CUDA_VISIBLE_DEVICES=0 python3 mitr_unsup_sub_selection_entropy_filtering.py $exp_name l1 0 l3 l4 l5 l6 0 5 > $exp_name/res.txt