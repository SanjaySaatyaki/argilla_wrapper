from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

class TextClassificationModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer = self._initialize_model()

    def _initialize_model(self):
        # Placeholder for model initialization logic
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name).to(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(device)
        return self.model, self.tokenizer

        
    def load_classification_data(self, dataset_name, split='train'):
        """
        Load a classification dataset using the Hugging Face datasets library.

        Args:
            dataset_name (str): The name of the dataset to load.
            split (str): The split of the dataset to load (e.g., 'train', 'test', 'validation').

        Returns:
            Dataset: The loaded dataset split.
        """
        return load_dataset(dataset_name, split=split)

    def tokenize(self, batch):
        return self.tokenizer(batch['text'], padding=True, truncation=True)
    
    def train(self, dataset, epochs=3):
        """
        Train a model on the provided dataset.

        Args:
            dataset: The dataset to train the model on.
            epochs (int): The number of training epochs.

        Returns:
            The trained model.
        """


        tokenized_dataset = dataset.map(self.tokenize, batched=True)
        tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        trainer.train()
        return self.model, self.tokenizer