

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd

# --- Model Loading (with quantization for local use) ---
@torch.inference_mode()
def get_model_and_tokenizer():
    """Loads the Med-Gemma model and tokenizer with 4-bit quantization."""
    model_id = "google/med-gemma-4b"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto", # Automatically uses GPU if available
    )
    return model, tokenizer

# --- Analysis Function ---
@torch.inference_mode()
def analyze_diagnosis_text(model, tokenizer, text: str) -> dict:
    """Runs Med-Gemma to analyze the diagnosis text and returns structured data."""
    prompt = f"""
    Analyze the following pathology diagnosis text. Extract the following information in a structured format:
    1.  **Classification**: Determine if the diagnosis is 'Cancer', 'Not Cancer', or 'Abnormal'.
    2.  **IHC Stains**: List all immunohistochemical (IHC) stains mentioned (e.g., CD3, CD20, BCL2).
    3.  **IHC Results**: For each stain, specify if it is 'Positive', 'Negative', or 'Equivocal/Unknown'.

    **Diagnosis Text:**
    {text}

    **Analysis Output:**
    """
    
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**input_ids, max_new_tokens=200)
    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Basic parsing of the model's output (can be improved with more robust logic)
    lines = result_text.split('\n')
    parsed_output = {
        "classification": "Unknown",
        "ihc_stains": [],
        "ihc_results": []
    }
    try:
        for line in lines:
            if "Classification:" in line:
                parsed_output["classification"] = line.split(":")[1].strip()
            if "IHC Stains:" in line:
                parsed_output["ihc_stains"] = [stain.strip() for stain in line.split(":")[1].split(",")]
            if "IHC Results:" in line:
                 parsed_output["ihc_results"] = [res.strip() for res in line.split(":")[1].split(",")]
    except IndexError:
        # Handle cases where the model output is not as expected
        pass
        
    return parsed_output

def run_batch_analysis(model, tokenizer, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Runs analysis on a DataFrame and appends the results."""
    results = []
    for index, row in df.iterrows():
        text = row[text_column]
        if pd.notna(text):
            analysis_result = analyze_diagnosis_text(model, tokenizer, text)
            results.append(analysis_result)
        else:
            results.append({"classification": "", "ihc_stains": [], "ihc_results": []})
    
    # Append results to the original DataFrame
    analysis_df = pd.DataFrame(results)
    return pd.concat([df, analysis_df], axis=1)

