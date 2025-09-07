import streamlit as st
from GPTModules import GPTModel,generate,text_to_token_ids,token_ids_to_text
import torch
import tiktoken
from huggingface_hub import hf_hub_download
@st.cache_resource
def load_model():
    tokenizer = tiktoken.get_encoding("gpt2")
    config={
        'vocab_size': 50257,
    'context_length': 1024,
    'drop_rate': 0.0,
    'qkv_bias': True,
    'dim': 1024,
    'num_layers': 24,
    'num_heads': 16
    }
    model_path = hf_hub_download(
    repo_id="subumangu2003/subugpt",
    filename="model.pth"
    )
    model=GPTModel(config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval();
    return model,config,tokenizer
model,config,tokenizer=load_model()

# Sidebar
st.sidebar.markdown("## Made with ❤️ by SubuMangu")

# Title (centered)
st.markdown(
    "<h1 style='text-align: center;'>SubuGPT🤖</h1>",
    unsafe_allow_html=True
)

# Parameters card (using container with background styling)
with st.container():
    temperature = st.slider("Temperature", 0.0, 5.0, 1.0, step=0.1)
    top_k = st.number_input("Top-k", min_value=1, value=10, step=1)
    max_words = st.number_input("Max word size", min_value=1, value=100, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

# Prompt input
prompt = st.text_area("Enter your prompt:", placeholder="Ask me anything...")
instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{prompt}"
    )

# Submit button
if st.button("Submit"):
    if prompt:
        # (Here you would call your LLM API with temperature, top_k, max_words, and prompt)
        # Mock response for now
        idx=generate(
            model,
            text_to_token_ids(instruction_text,tokenizer),
            max_new_tokens=max_words+len(instruction_text),
            context_size=config["context_length"],
            temperature=temperature,
            top_k=top_k,
            eos_id=50256
        )
        generated_text=token_ids_to_text(idx,tokenizer)
        response_text = (
        generated_text[len(instruction_text):]
        .replace("### Response:", "")
        .strip()
)
        answer=response_text.strip()
        st.markdown(f"### Answer:\n{answer}")

        # Evaluation button appears only after an answer is shown
        if st.button("Evaluate Model with LLaMA 3 🦙"):
            st.info("Evaluation with LLaMA 3 started... (placeholder)")
    else:
        st.warning("⚠️ Please enter a prompt before submitting.")
