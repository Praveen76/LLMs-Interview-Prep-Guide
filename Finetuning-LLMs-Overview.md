# Q1. Explain LORA and QLORA for LLMs finetuning.
Ans: 
### **LoRA** (Low-Rank Adaptation) is a widely used, parameter-efficient fine-tuning technique for training custom LLMs. It is used to adapt a pre-trained LLM to a new task by adding a small number of task-specific parameters. LoRA is based on the low-rank matrix factorization technique, which reduces the number of parameters required for fine-tuning. [This technique has been shown to be effective in reducing the number of parameters required for fine-tuning, while maintaining or improving the performance of the model ](https://lightning.ai/pages/community/lora-insights/)¹².

**QLoRA** (Quantized Low-Rank Adaptation) is an extension of LoRA that further reduces the memory requirements of fine-tuning LLMs. QLoRA first quantizes the LLM to 4-bit to reduce its size and then performs LoRA training in 32-bit precision for effectiveness. Weights temporarily revert to 32-bit. Weights are quantized and de-quantized in a stepwise manner to manage GPU memory. QLoRA enables efficient fine-tuning of giant LLMs on typical GPUs ²⁴.


Source:[1](https://lightning.ai/pages/community/lora-insights/)1/[2](https://www.predera.com/blog/llm-finetuning-with-lora-and-qlora)023
(1) Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of .... https://lightning.ai/pages/community/lora-insights/.
([2](https://www.predera.com/blog/llm-finetuning-with-lora-and-qlora)) LLM finetuning with LoRA and QLoRA - predera.com. https://www.predera.com/blog/llm-finetuning-with-lora-and-qlora.
([3](https://arxiv.org/abs/2305.14314)) [2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs - arXiv.org. https://arxiv.org/abs/2305.14314.
(4) Interactive Study with Conversation1st.ai — QLoRA: Efficient Finetuning .... https://medium.com/@tonytong.ai/interactive-study-with-conversation1st-ai-qlora-efficient-finetuning-of-quantized-llms-7b50cb31ea19.

# Q4. Why do we use LoRA, and QLoRa for LLMs?
Ans: [LoRA and QLoRA are two widely used techniques for training custom Large Language Models (LLMs) that are parameter-efficient and memory-efficient ](https://lightning.ai/pages/community/lora-insights/)¹. LoRA stands for Low Rank Adapters, which is a fine-tuning technique that helps in training LLMs with fewer parameters ¹. QLoRA, on the other hand, is a memory-efficient version of LoRA that quantizes the LLM weights to 4-bits, reducing the model's memory footprint by 8x ². QLoRA then finetunes the quantized LLM using LoRA, which enables the refined model to preserve the majority of the accuracy of the original LLM while being significantly smaller and quicker ².

Source: [1](https://lightning.ai/pages/community/lora-insights/)[1](https://lightning.ai/pages/community/lora-insights/)/[2](https://www.cloudbooklet.com/qlora-efficient-finetuning-of-quantized-llms/)0[2](https://www.cloudbooklet.com/qlora-efficient-finetuning-of-quantized-llms/)3
(1) Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of .... https://lightning.ai/pages/community/lora-insights/.
(2) QLoRA: Efficient Finetuning of Quantized LLMs - Cloudbooklet. https://www.cloudbooklet.com/qlora-efficient-finetuning-of-quantized-llms/.
(3) LoRA and QLoRA recommendations for LLMs - Google Cloud. https://cloud.google.com/vertex-ai/docs/model-garden/lora-qlora.
(4) [2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs - arXiv.org. https://arxiv.org/abs/2305.14314.


