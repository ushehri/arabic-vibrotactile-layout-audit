#!/bin/bash
# waits for features_v2.py cv (PID 392903), then analysis -> handcheck -> report -> copies
cd data/full52-v2; source ../pilot/.venv/bin/activate
while kill -0 392903 2>/dev/null; do sleep 30; done
grep -q DONE features_cv.log || { echo "$(date +%H:%M) chain: features_cv not DONE, abort" >> STATUS.md; exit 1; }
echo "- $(date +%H:%M) chain: CV features DONE; analysis_v2 started" >> STATUS.md
python analysis_v2.py > analysis_v2.log 2>&1 || { echo "- $(date +%H:%M) chain: analysis FAILED (see analysis_v2.log)" >> STATUS.md; exit 1; }
echo "- $(date +%H:%M) chain: analysis DONE" >> STATUS.md
python handcheck.py > handcheck.log 2>&1; python report_v2.py > report_v2.log 2>&1
P=.; mkdir -p $P/05-results/full-5.2-v2 $P/04-code/full-5.2-v2 $P/06-figures
cp *.md *.csv *.json $P/05-results/full-5.2-v2/ ; cp *.py chain.sh $P/04-code/full-5.2-v2/ ; for f in fig*.png; do cp $f $P/06-figures/v2_$f; cp $f $P/05-results/full-5.2-v2/; done
echo "- $(date +%H:%M) chain: handcheck/report/copies DONE (results md still to be written)" >> STATUS.md
