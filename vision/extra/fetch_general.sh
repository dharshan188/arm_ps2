cd ~/arm2026/vision/extra/general_waste
while IFS='|' read tag q n; do THUMB=500 timeout 300 python3 -I ../../commons_train_fetch.py . "$tag" "$q" "$n" 2>&1 | tail -1; done < ../general_queries.txt
echo FETCH_DONE; ls *.jpg | wc -l
