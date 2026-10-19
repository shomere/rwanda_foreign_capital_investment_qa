# Rwanda Foreign Private Capital Q&A

Domain-specific question answering over the National Bank of Rwanda report
"Foreign Private Capital in Rwanda: Survey Report – FPC 2025".

## Links
- Demo video: https://youtube.com/live/08hG2uiJO-4?feature=share

- Model weights:https://drive.google.com/drive/folders/19P-60sSUVRtSR3_7hDcW9KN7czUPJMYu?usp=drive_link
- Some code in a Colab notebook: https://colab.research.google.com/drive/1ewjpQN_hq0ZDsakBjdP8tXY2bffnR-fX?usp=sharing

- Live system: not hosted. Hugging Face now requires a paid plan to create Gradio Spaces
  for new free accounts. The app runs locally and is shown in the video.

## Method
Retrieval (BM25, stopwords removed, top 3) -> extractive reader (RoBERTa-base fine-tuned from
deepset/roberta-base-squad2 on 58 hand-written questions) -> answer chosen by confidence margin ->
refused if margin < 11.95 or if the question asks for advice or a prediction.

## Data
68 passages (150 words, 30 overlap) from the report; 106 questions written (73 valid after
exact-match validation: 58 train, 15 validation), 20 test, 20 refusal questions. Split by chapter.

## Results
Setting	hit@1	hit@3	MRR
plain, 8 rows	0.750	1.000	0.854
+ stopwords, 8 rows	0.875	1.000	0.938
+ stopwords, 73 rows	0.507	0.836	0.675
same, matching text instead of id	0.548	0.849	


## Run it
1. pip install torch transformers rank_bm25 gradio numpy pdfplumber
2. Download the model from the Drive link into models/qa_model/
3. python src/app.py   (opens http://127.0.0.1:7860)
Evaluation scripts (src/run_eval.py) use home-folder paths: data in ~/content/{qa,test,refusal},
model in ~/capstone/models/qa_model.

## Credits
Report data: National Bank of Rwanda (CC BY 4.0 per the NISR page). Pretrained model:
deepset/roberta-base-squad2. Libraries: transformers, rank_bm25, gradio, pdfplumber.
Written by me: passages pipeline, 106 triples, validator, splits, retrieval ablation,
fine-tuning, margin-based refusal, evaluation, app.
