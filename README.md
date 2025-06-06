  # GPT-2 Fine-tuning on Story Cloze Test Dataset

This project fine-tunes a GPT-2 language model on the Story Cloze Test dataset to improve its ability to generate coherent story continuations and endings.

## Project Overview

The Story Cloze Test dataset contains four-sentence stories with two possible endings. This project:
- Preprocesses the dataset to create complete five-sentence stories
- Fine-tunes GPT-2 on these stories using the Hugging Face Transformers library
- Produces a specialized model for story generation and completion

## Dataset

**Source:** Story Cloze Test (Winter 2018)
- **Validation set:** `cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv`
- **Test set:** `cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv`

**Format:** Each example contains:
- `InputSentence1-4`: Four-sentence story context
- `RandomFifthSentenceQuiz1/2`: Two possible endings
- `AnswerRightEnding`: Correct ending (1 or 2)

## Setup

### Requirements
```bash
pip install transformers datasets torch
```

### Data Preparation
1. Place the CSV files in your project directory
2. Run the preprocessing script to tokenize the data

## Usage

### 1. Data Preprocessing and Training

```python
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load datasets
dataset = load_dataset("csv", data_files="cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv")
datasettest = load_dataset("csv", data_files="cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv")

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Preprocess and train (see full code in repository)
```

### 2. Using the Fine-tuned Model

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load your fine-tuned model
checkpoint_dir = "./results/checkpoint-1414"  # or your checkpoint path
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)
model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)

# Generate text
def generate_story(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Sarah walked into the library and"
story = generate_story(prompt)
print(story)
```

## Model Architecture

- **Base Model:** GPT-2 (117M parameters)
- **Fine-tuning:** Language modeling objective on complete stories
- **Training Configuration:**
  - 2 epochs
  - Batch size: 2 per device
  - Max sequence length: 256 tokens
  - Save strategy: Every epoch

## File Structure

```
├── results/
│   └── checkpoint-1414/          # Fine-tuned model checkpoint
│       ├── config.json           # Model configuration
│       ├── generation_config.json # Generation settings
│       ├── model.safetensors     # Model weights
│       ├── optimizer.pt          # Optimizer state
│       ├── scheduler.pt          # Scheduler state
│       ├── trainer_state.json    # Training state
│       └── training_args.bin     # Training arguments
├── logs/                         # Training logs
├── cloze_test_val_*.csv         # Validation dataset
├── cloze_test_test_*.csv        # Test dataset
└── fine_tuning_script.py        # Main training script
```

## Key Features

- **Data Preprocessing:** Automatically concatenates story context with correct endings
- **Tokenization:** Handles padding and truncation for consistent input lengths
- **Language Modeling:** Uses standard LM objective with labels = input_ids
- **Evaluation Ready:** Supports both perplexity evaluation and story generation

## Training Process

1. **Data Loading:** CSV files loaded using Hugging Face Datasets
2. **Preprocessing:** Stories assembled from components with correct endings
3. **Tokenization:** GPT-2 tokenizer with padding to max_length=256
4. **Training:** Standard language modeling with Hugging Face Trainer
5. **Checkpointing:** Model saved every epoch for evaluation

## Evaluation

The model can be evaluated in two ways:
- **Perplexity:** Standard language modeling metrics
- **Story Cloze:** Multiple-choice ending selection accuracy

## Applications

- **Story Generation:** Generate creative continuations from prompts
- **AI Agents:** Use as reasoning/generation component in agent frameworks
- **Interactive Fiction:** Power storytelling applications
- **Educational Tools:** Help with creative writing assistance

## Interview Talking Points

This project demonstrates:
- **NLP Pipeline Development:** End-to-end dataset preprocessing to model deployment
- **Transfer Learning:** Fine-tuning pre-trained models for domain-specific tasks
- **Practical Implementation:** Using industry-standard libraries (Hugging Face)
- **Problem Solving:** Handling dataset format challenges and model integration

## Future Improvements

- Experiment with larger models (GPT-2 Medium/Large)
- Implement multiple-choice evaluation metrics
- Add web interface for interactive story generation
- Deploy as API service for integration with other applications

## License

Story Cloze Test dataset creators

## Acknowledgments

- Hugging Face for the Transformers library
- Story Cloze Test dataset creators
- GPT-2 model by OpenAI

[1] https://pplx-res.cloudinary.com/image/private/user_uploads/14873364/7926c49a-11d5-4a72-8c05-34c897cfc2ee/image.jpg
