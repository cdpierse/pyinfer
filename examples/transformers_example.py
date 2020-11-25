from pyinfer import MultiInferenceReport
from transformers import AutoModelForSequenceClassification, AutoTokenizer

distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
distilbert_tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

electra_model = AutoModelForSequenceClassification.from_pretrained(
    "monologg/electra-small-finetuned-imdb"
)
electra_tokenizer = AutoTokenizer.from_pretrained(
    "monologg/electra-small-finetuned-imdb"
)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-imdb"
)
bert_tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

TEXT = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness"

distilbert_text = distilbert_tokenizer(TEXT, return_tensors="pt")
electra_text = electra_tokenizer(TEXT, return_tensors="pt")
bert_text = bert_tokenizer(TEXT, return_tensors="pt")


multi_report = MultiInferenceReport(
    [distilbert_model, electra_model, bert_model],
    [distilbert_text["input_ids"], electra_text["input_ids"], bert_text["input_ids"]],
    n_iterations=100,
    model_names=["Distilbert", "Electra", "Bert"],
    infer_failure_point=0.170,
)
multi_report.run()
multi_report.plot()
