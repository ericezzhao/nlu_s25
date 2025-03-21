"""
Code for Problem 1 of HW 2.
"""
import pickle

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import preprocess_dataset


def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    model = BertForSequenceClassification.from_pretrained(directory)
    
    bias_params = 0
    non_bias_params = 0

    for name, param in model.named_parameters():
        if 'bias' in name:
            bias_params += param.numel()
        else:
            non_bias_params += param.numel()
            
    print(f"total parameters: {bias_params + non_bias_params}")

    training_args = TrainingArguments(
        output_dir="./test_output",
        do_train=False,
        do_predict=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=lambda p: {
            "accuracy": evaluate.load("accuracy").compute(
                predictions=p.predictions.argmax(-1),
                references=p.label_ids
            )["accuracy"]
        }
    )
    
    return trainer



if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up tester
    tester = init_tester("checkpoints/with_bitfit")  # your BitFit model path
    results = tester.predict(imdb["test"])
    with open("test_results_with_bitfit.p", "wb") as f:
        pickle.dump(results, f)
