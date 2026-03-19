from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# 1. SERVER SETUP & CORS
# ==========================================
app = FastAPI(title="CV Roaster API")

# Allow Next.js to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change to your frontend URL in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the incoming data structure
class RoastRequest(BaseModel):
    cv_text: str

# ==========================================
# 2. MODEL CONFIGURATION (Loads on startup)
# ==========================================
LORA_PATH = r"C:\Users\HP\Documents\GitHub\LLM_FineTuned_CV_Roast\roaster_v1"  # Ensure this points to your extracted model folder
BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit"

print("⏳ Starting server and loading the Roaster into GPU...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Snap the fine-tuned adapters onto the base model
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print("✅ Model loaded! API is ready to accept requests.")

# ==========================================
# 3. THE API ROUTE
# ==========================================
@app.post("/roast")
async def generate_roast(request: RoastRequest):
    try:
        # Prevent the context from being too massive
        selected_context = request.cv_text[:2000] 
        
        # Format EXACTLY as it was trained
        prompt = f"### CV Context:\n{selected_context}\n\n### Roast:\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,      # INCREASED: Allows for a much more detailed, longer roast
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )
            
        # Decode and strip out the prompt part
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return {"roast": response.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))