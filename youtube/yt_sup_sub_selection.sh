mkdir yt_sup
mkdir yt_sup/l134
python yt_sup_sub_selection.py yt_sup/l134 l1 0 l3 l4 0 0 0 5 > yt_sup/l134/res.txt
mkdir yt_sup/l1234
python yt_sup_sub_selection.py yt_sup/l1234 l1 l2 l3 l4 0 0 0 5 > yt_sup/l1234/res.txt
mkdir yt_sup/l146
python yt_sup_sub_selection.py yt_sup/l146 l1 0 0 l4 0 l6 0 5 > yt_sup/l146/res.txt
mkdir yt_sup/l1346
python yt_sup_sub_selection.py yt_sup/l1346 l1 0 l3 l4 0 l6 0 5 > yt_sup/l1346/res.txt
mkdir yt_sup/l12346
python yt_sup_sub_selection.py yt_sup/l123456 l1 l2 l3 l4 0 l6 0 5 > yt_sup/l123456/res.txt
mkdir yt_sup/l13456
python yt_sup_sub_selection.py yt_sup/l13456 l1 0 l3 l4 l5 l6 0 5 > yt_sup/l13456/res.txt
mkdir yt_sup/l123456
python yt_sup_sub_selection.py yt_sup/l123456 l1 l2 l3 l4 l5 l6 0 5 > yt_sup/l123456/res.txt
mkdir yt_sup/l123456qg
python yt_sup_sub_selection.py yt_sup/l123456qg l1 l2 l3 l4 l5 l6 qg 5 > yt_sup/l123456qg/res.txt
mkdir yt_sup/l13456qg
python yt_sup_sub_selection.py yt_sup/l13456qg l1 0 l3 l4 l5 l6 qg 5 > yt_sup/l13456qg/res.txt

