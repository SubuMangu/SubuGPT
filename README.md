# SubuGPT 🤖
- Hello there 🖐️..... this is Subrat and I made my own LLM from scratch.
- Made a clone of GPT2 medium and loaded it's pretrained weights.
- App link: https://subugpt.streamlit.app/


## ⚡ Usage
<p align="center"><img src="Images/Screen Recording 2025-09-09 160445.gif" width="750" height=""></p>

## 📊 Model Architecture

<p align="center"><img src="Images/Screenshot 2025-09-09 155249.png" width="500" height=""></p>

- **No of parameters:** 431M
- **Vocab size:** 50257

## 🧪 Evaluation

- Got a score of 47.16/100 on evaluating our model with llama3(See `Evaluating GPT.ipynb` ).
- We can further improve the model score by finetuning with more instruction datasets(See `Finetuning GPT.ipynb`).


## 🤝 Scope for Contribution
1. Finetuning with better instruction datasets(See `Finetuning GPT.ipynb`).
2. Quantization and integration of PEFTs like LoRA.
3. Introducing advanced features like RLHF(Reinforment Learning on Human Feedback) and reasoning.
4. Hosting model in huggingface : https://huggingface.co/subumangu2003/subugpt
---

## 📜 License

This project is licensed under the **Apache License** – see the [LICENSE](LICENSE) file for details.

## 📝 References and futher reading
- [LLM from Scratch](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka

