#https://drive.google.com/u/0/uc?export=download&confirm=80Ax&id=1dWTUC_f0Ks-Hg4TtkHrwWy0RmzLMH-X1
fileId=1dWTUC_f0Ks-Hg4TtkHrwWy0RmzLMH-X1
fileName=data.tar.gz
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/u/0/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName} 
