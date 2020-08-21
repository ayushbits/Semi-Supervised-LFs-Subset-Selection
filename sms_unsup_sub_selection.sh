mkdir sms_unsup
mkdir sms_unsup/l134
python sms_unsup_sub_selection.py sms_unsup/l134 l1 0 l3 l4 0 0 0 5 > sms_unsup/l134/res.txt
mkdir sms_unsup/l146
python sms_unsup_sub_selection.py sms_unsup/l146 l1 0 0 l4 0 l6 0 5 > sms_unsup/l146/res.txt
mkdir sms_unsup/l1346
python sms_unsup_sub_selection.py sms_unsup/l1346 l1 0 l3 l4 0 l6 0 5 > sms_unsup/l1346/res.txt
mkdir sms_unsup/l123456
python sms_unsup_sub_selection.py sms_unsup/l123456 l1 l2 l3 l4 l5 l6 0 5 > sms_unsup/l123456/res.txt
mkdir sms_unsup/l123456qg
python sms_unsup_sub_selection.py sms_unsup/l123456qg l1 l2 l3 l4 l5 l6 qg 5 > sms_unsup/l123456qg/res.txt
mkdir sms_unsup/l1234
python sms_unsup_sub_selection.py sms_unsup/l1234 l1 l2 l3 l4 0 0 0 5 > sms_unsup/l1234/res.txt
mkdir sms_unsup/l1246
python sms_unsup_sub_selection.py sms_unsup/l1246 l1 l2 0 l4 0 l6 0 5 > sms_unsup/l1246/res.txt
mkdir sms_unsup/l12346
python sms_unsup_sub_selection.py sms_unsup/l12346 l1 l2 l3 l4 0 l6 0 5 > sms_unsup/l12346/res.txt
mkdir sms_unsup/l13456qg
python sms_unsup_sub_selection.py sms_unsup/l13456qg l1 0 l3 l4 l5 l6 qg 5 > sms_unsup/l13456qg/res.txt
mkdir sms_unsup/l13456
python sms_unsup_sub_selection.py sms_unsup/l13456 l1 0 l3 l4 l5 l6 0 5 > sms_unsup/l13456/res.txt
