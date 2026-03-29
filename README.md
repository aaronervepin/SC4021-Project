# Singapore Exam Sentiment Analysis - Reddit Crawler

This project crawls Reddit data related to Singapore examinations (O-Level, A-Level, Polytechnic, University) to create a sentiment analysis dataset for assessing exam difficulty based on student feedback.

## Project Structure

```
InfoRetProj/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── config.py                 # Configuration settings
├── reddit_crawler.py         # Main Reddit scraping script
├── data_processor.py         # Data cleaning and processing
├── sentiment_labeler.py      # Sentiment labeling utilities
├── create_eval_dataset.py    # Creates the final eval.xls
├── data/                     # Raw crawled data
│   └── raw/
├── eval.xls                  # Final evaluation dataset
└── logs/                     # Crawling logs
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the crawler:
```bash
python reddit_crawler.py
```

3. Process and label the data:
```bash
python data_processor.py
```

4. Create the evaluation dataset:
```bash
python create_eval_dataset.py
```

## Dataset Format

The `eval.xls` file follows the standard sentiment benchmark format:
- **text**: The Reddit post/comment text
- **label**: Sentiment label (0 = negative, 1 = neutral, 2 = positive)

## Target Subreddits

- r/SGExams
- r/singapore
- r/NUS
- r/NTU
- r/SMU
- r/SIT
- r/SUSS
- r/poly (Singapore context)

## Keywords

O-Level, A-Level, PSLE, IB, Polytechnic, JC, University, NUS, NTU, SMU, exam difficulty, paper, module, etc.

## Ethical Considerations

- Only public Reddit posts are collected
- No personal identification information is stored
- Data is used solely for academic research purposes
- Complies with Reddit's robots.txt and terms of service
