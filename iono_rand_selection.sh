mkdir -p iono_rand
mkdir -p iono_rand/l134
CUDA_VISIBLE_DEVICES=2 python iono_rand_selection.py iono_rand/l134 l1 0 l3 l4 0 0 0 5 datasets/ionosphere > iono_rand/l134/res.txt
mkdir -p iono_rand/l1234
CUDA_VISIBLE_DEVICES=2 python iono_rand_selection.py iono_rand/l1234 l1 l2 l3 l4 0 0 0 5 datasets/ionosphere > iono_rand/l1234/res.txt
mkdir -p iono_rand/l146
CUDA_VISIBLE_DEVICES=2 python iono_rand_selection.py iono_rand/l146 l1 0 0 l4 0 l6 0 5 datasets/ionosphere > iono_rand/l146/res.txt
mkdir -p iono_rand/l1346
CUDA_VISIBLE_DEVICES=2 python iono_rand_selection.py iono_rand/l1346 l1 0 l3 l4 0 l6 0 5 datasets/ionosphere > iono_rand/l1346/res.txt
mkdir -p iono_rand/l12346
CUDA_VISIBLE_DEVICES=2 python iono_rand_selection.py iono_rand/l123456 l1 l2 l3 l4 0 l6 0 5 datasets/ionosphere > iono_rand/l123456/res.txt
mkdir -p iono_rand/l13456
CUDA_VISIBLE_DEVICES=2 python iono_rand_selection.py iono_rand/l13456 l1 0 l3 l4 l5 l6 0 5 datasets/ionosphere > iono_rand/l13456/res.txt
mkdir -p iono_rand/l123456
CUDA_VISIBLE_DEVICES=2 python iono_rand_selection.py iono_rand/l123456 l1 l2 l3 l4 l5 l6 0 5 datasets/ionosphere > iono_rand/l123456/res.txt
mkdir -p iono_rand/l123456qg
CUDA_VISIBLE_DEVICES=2 python iono_rand_selection.py iono_rand/l123456qg l1 l2 l3 l4 l5 l6 qg 5 datasets/ionosphere > iono_rand/l123456qg/res.txt
mkdir -p iono_rand/l13456qg
CUDA_VISIBLE_DEVICES=2 python iono_rand_selection.py iono_rand/l13456qg l1 0 l3 l4 l5 l6 qg 5 datasets/ionosphere > iono_rand/l13456qg/res.txt

