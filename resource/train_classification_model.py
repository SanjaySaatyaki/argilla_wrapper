from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

from dotenv import load_dotenv

load_dotenv()

class TextClassificationModel:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        # Load the entire dataset dictionary (all splits)
        self.dataset_dict = self.load_classification_data(dataset_name)
        self.model, self.tokenizer = self._initialize_model()

    def _initialize_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Get num_labels from the 'train' split of the dataset
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.dataset_dict['train'].features['label'].num_classes).to(device)
        return self.model, self.tokenizer


    def load_classification_data(self, dataset_name):
        """
        Load a classification dataset using the Hugging Face datasets library.

        Args:
            dataset_name (str): The name of the dataset to load.

        Returns:
            DatasetDict: The loaded dataset dictionary containing all splits.
        """
        return load_dataset(dataset_name)

    def tokenize(self, batch):
        return self.tokenizer(batch['text'], padding=True, truncation=True)

    def train(self,epochs=3):
        """
        Train a model on the provided dataset.

        Args:
            epochs (int): The number of training epochs.

        Returns:
            The trained model.
        """
        # Tokenize all splits in the DatasetDict
        tokenized_dataset = self.dataset_dict.map(self.tokenize, batched=True)
        tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
            output_dir='./results/'+self.model_name.replace("/", "_"),
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            # Use the 'test' split for evaluation as 'imdb' dataset typically has 'train' and 'test' splits
            eval_dataset=tokenized_dataset["test"],
        )

        trainer.train()
        return self.model, self.tokenizer

if __name__ == "__main__":
    model_name = "distilbert-base-uncased"
    dataset_name = "imdb"

    classification_model = TextClassificationModel(model_name, dataset_name)
    trained_model, tokenizer = classification_model.train(epochs=3)
