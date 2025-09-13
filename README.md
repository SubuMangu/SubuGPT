# SubuGPT 🤖
SubuGPT is a custom large language model built by replicating GPT-2 Medium architecture and extending it with evaluation, deployment, and fine-tuning experiments. The goal was to understand transformer internals and explore practical deployment of generative models.

- **App link:** https://subugpt.streamlit.app/


## ⚡ Usage
<p align="center"><img src="Images/Screen Recording 2025-09-09 160445.gif" width="750" height=""></p>


## 📊 Model Architecture

<p align="center"><img src="Images/Screenshot 2025-09-09 155249.png" width="500" height=""></p>

- **No of parameters:** 431M
- **Vocab size:** 50257
- **Embedding Dimention:** 1024
- **Attention Mechanism:** Multihead Causal Attention
- **Attention Heads:** 16
- **Number of Transformer blocks:** 24

## 🛠️ Implementation Details
1. **Tokenization** using Byte Pair Encoding(BPE) using a tokenising package of OpenAI named `tiktoken`. Futher planned  to create a BPE from scratch.
2. **Creating GPT Dataset** using the context window.
3. **Making GPT Model** including transformer blocks with Multihead Causal Attention . See in Model Architecture section.
4. **Pretraining** is done by loading the weights from gpt2 medium to our model. To pretrained the model from scratch you can train the model with GPT Dataset.
5. **Finetuning** is done with instruction dataset available in `Dataset/instruction-data.json`. See `Finetuning_GPT.ipynb` for details.

## ⚙️ Environment & Resources
- **VRAM usage for finetuning:** ~8.32 GB(to be futher reduced by quantization)  
- **Workspace:** Google Colab  
- **Framework:** PyTorch
  
## 🧪 Evaluation

- Got a score of 47.16/100 on evaluating our model with llama3(See `Evaluating_GPT.ipynb` ).
- We can further improve the model score by finetuning with more instruction datasets(See `Finetuning_GPT.ipynb`).

## 🤝 Scope for Contribution
1. Finetuning with better instruction datasets(See `Finetuning_GPT.ipynb`).
2. Quantization and integration of PEFTs like LoRA.
3. Introducing advanced features like RLHF(Reinforment Learning on Human Feedback) and reasoning.
4. Hosting model in huggingface : https://huggingface.co/subumangu2003/subugpt

## 📜 License

This project is licensed under the **Apache License** – see the [LICENSE](LICENSE) file for details.

## 📝 References and futher reading
- [LLM from Scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka

