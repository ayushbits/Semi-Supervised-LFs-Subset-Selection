mkdir -p trec_rand
mkdir -p trec_rand/l134
python trec_rand_selection.py trec_rand/l134 l1 0 l3 l4 0 0 0 5 > trec_rand/l134/res.txt
mkdir -p trec_rand/l146
python trec_rand_selection.py trec_rand/l146 l1 0 0 l4 0 l6 0 5 > trec_rand/l146/res.txt
mkdir -p trec_rand/l1346
python trec_rand_selection.py trec_rand/l1346 l1 0 l3 l4 0 l6 0 5 > trec_rand/l1346/res.txt
mkdir -p trec_rand/l123456
python trec_rand_selection.py trec_rand/l123456 l1 l2 l3 l4 l5 l6 0 5 > trec_rand/l123456/res.txt
mkdir -p trec_rand/l123456qg
python trec_rand_selection.py trec_rand/l123456qg l1 l2 l3 l4 l5 l6 qg 5 > trec_rand/l123456qg/res.txt
mkdir -p trec_rand/l1234
python trec_rand_selection.py trec_rand/l1234 l1 l2 l3 l4 0 0 0 5 > trec_rand/l1234/res.txt
mkdir -p trec_rand/l12346
python trec_rand_selection.py trec_rand/l12346 l1 l2 l3 l4 0 l6 0 5 > trec_rand/l12346/res.txt
mkdir -p trec_rand/l13456qg
python trec_rand_selection.py trec_rand/l13456qg l1 0 l3 l4 l5 l6 qg 5 > trec_rand/l13456qg/res.txt
mkdir -p trec_rand/l13456
python trec_rand_selection.py trec_rand/l13456 l1 0 l3 l4 l5 l6 0 5 > trec_rand/l13456/res.txt
