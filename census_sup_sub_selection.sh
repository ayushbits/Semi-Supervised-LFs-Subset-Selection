mkdir census_sup
mkdir census_sup/l134
python census_sup_sub_selection.py census_sup/l134 l1 0 l3 l4 0 0 0 5 > census_sup/l134/res.txt
mkdir census_sup/l146
python census_sup_sub_selection.py census_sup/l146 l1 0 0 l4 0 l6 0 5 > census_sup/l146/res.txt
mkdir census_sup/l1346
python census_sup_sub_selection.py census_sup/l1346 l1 0 l3 l4 0 l6 0 5 > census_sup/l1346/res.txt
mkdir census_sup/l123456
python census_sup_sub_selection.py census_sup/l123456 l1 l2 l3 l4 l5 l6 0 5 > census_sup/l123456/res.txt
mkdir census_sup/l123456qg
python census_sup_sub_selection.py census_sup/l123456qg l1 l2 l3 l4 l5 l6 qg 5 > census_sup/l123456qg/res.txt
mkdir census_sup/l1234
python census_sup_sub_selection.py census_sup/l1234 l1 l2 l3 l4 0 0 0 5 > census_sup/l1234/res.txt
mkdir census_sup/l12346
python census_sup_sub_selection.py census_sup/l12346 l1 l2 l3 l4 0 l6 0 5 > census_sup/l12346/res.txt
mkdir census_sup/l13456qg
python census_sup_sub_selection.py census_sup/l13456qg l1 0 l3 l4 l5 l6 qg 5 > census_sup/l13456qg/res.txt
mkdir census_sup/l13456
python census_sup_sub_selection.py census_sup/l13456 l1 0 l3 l4 l5 l6 0 5 > census_sup/l13456/res.txt
