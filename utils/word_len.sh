#!sh
for i in texts/*.txt; do
echo $i
cat $i | python -c 'while 1:
 try:
  print(len(input().strip().split()))
 except EOFError:
  break' > wordlengths/$(basename $i .txt).wordlen
done
