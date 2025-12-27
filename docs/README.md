# CS412-CV-FinalProject

## Project Hierarchy
root/  
├── HAD/  
│   ├── questions/  
│   │   ├── train.json  
│   │   ├── test.json  
│   │   └── val.json  
│   ├── outputs/  
│   │   ├── train/  
│   │   ├── test/  
│   │   └── val/  
│   ├── keyframes/  
│   │   ├── train/  
│   │   ├── test/  
│   │   └── val/  
│   └── videos/  
│       ├── train/  
│       ├── test/  
│       └── val/  
├── SUTD/  
│   ├── questions/  
│   │   ├── train.json  
│   │   ├── test.json  
│   │   └── val.json  
│   ├── outputs/  
│   │   ├── train/  
│   │   ├── test/  
│   │   └── val/  
│   ├── keyframes/  
│   │   ├── train/  
│   │   ├── test/  
│   │   └── val/  
│   └── videos/  
│       ├── train/  
│       ├── test/  
│       └── val/  
├── src/  

> Reason: Unified processing since we always process all datasets together and they share the same exact format.

## Get Google Drive API Key
1. Go to https://console.developers.google.com/
2. Create a new project.
3. Navigate to "APIs & Services" > "Credentials".
4. Click on "Create Credentials" and select "API Key".
5. Select "Restrict Key" and enable "Google Drive API".
6. Copy the generated API key.

## VideoLLava Issues
- Original VideoLLava has repetitive and irrelevant responses when handling video input.
- Reasons: Unknown
- Examples:
    - Question: "Why is this video funny?"
    - Answer: "printer printerгомömöm frequencies frequencies frequencies frequencies frequencies также также такжеідідідilianilianilianilianWRITEWRITEWRITEWRITEilianterraterraterraterraynaynaczaczaynaynaneoneoneoneoneoneoneoneoneoneoneoneoneoneoneoneoneoneoneoneoneoneoneoneoneo creature creature creature creature creature creature creature creature creature creature creature creature creature creatureterraterraalandöm frequencies frequencies frequencies frequencies frequencies frequencies frequenciesmesmesneoneoneoneo creature creature creatureagonagonagonagonbournebourneфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорфорterraterramesmesüttüttüttionyionyionyionyiony сы сыjavascriptjavascript arose arose aroseisenisenisenalandalandöm消消 influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence influence actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress actress"


## Ablation Results
- 64 frames with Katna keyframe extraction
=== Evaluation Report ===
GT file:   .\sutd_test_gt.csv
PRED file: .\sutd_tests_with_answer_llava_video_64_frames_katna.csv
Total GT ids:    6075
Total PRED ids:  6075
Compared ids:    6075
Correct:         2687
Wrong:           3388
Accuracy:        44.2305%

Wrong answers written to: .\sutd_wrong_ans_64_frames.csv

- 64 frames without Katna keyframe extraction
=== Evaluation Report ===
GT file:   .\sutd_test_gt.csv
PRED file: .\sutd_test_with_answers_llava_video_64_frames.csv
Total GT ids:    6075
Total PRED ids:  6075
Compared ids:    6075
Correct:         2465
Wrong:           3610
Accuracy:        40.5761%

Wrong answers written to: .\sutd_wrong_ans_64_frames.csv

- 8 frames with Katna keyframe extraction


- 8 frames without Katna keyframe extraction
=== Evaluation Report ===
GT file:   .\sutd_test_gt.csv
PRED file: .\sutd_test_with_answers_llava_video.csv
Total GT ids:    6075
Total PRED ids:  6075
Compared ids:    6075
Correct:         2762
Wrong:           3313
Accuracy:        45.4650%


Wrong answers written to: .\sutd_wrong_ans_64_frames.csv

- 8 frames with CNN + LSTM baseline
=== Evaluation Report ===
GT file:   .\answers\sutd_test_gt.csv
PRED file: .\sutd_cnn_lstm_predictions.csv
Total GT ids:    6075
Total PRED ids:  6075
Compared ids:    6075
Correct:         1833
Wrong:           4242
Accuracy:        30.1728%

=== Evaluation Report ===
GT file:   .\answers\sutd_test_gt.csv
PRED file: .\answers\sutd_cnn_lstm_predictions.csv
Total GT ids:    6075
Total PRED ids:  6075
Compared ids:    6075
Correct:         1823
Wrong:           4252
Accuracy:        30.0082%

Wrong answers written to: .\sutd_cnn_lstm_wrong_ans.csv

“We choose CNN+LSTM as a compute-efficient baseline for long dashcam videos and limited supervised data. Despite Transformers being popular by 2021, self-attention over long frame sequences is expensive and often requires large-scale pretraining to outperform simpler recurrent models. This baseline also makes it easier to interpret how frame sampling and augmentation affect causal reasoning accuracy.”

## Explore Video-Specific Encoders:
VideoChat vs Llava-NeXt

## Error Analysis
- Databset issues: mislabeled answers, ambiguous questions
- Model issues: difficulty understanding temporal dynamics, complex reasoning

## Video Captioning?


## 