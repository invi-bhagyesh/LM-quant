def _create_huggingface_model(model_id: str) -> Callable[[str], str]:
    """Create a Hugging Face model function with enhanced retry mechanism."""
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
    except ImportError as e:
        raise ImportError(
            "Hugging Face models require 'transformers' and 'torch' libraries. "
            "Install with: pip install transformers torch"
        ) from e
    
    logger.info(f"Loading Hugging Face model: {model_id}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        logger.info(f"Successfully loaded model: {model_id}")
        
    except Exception as e:
        logger.error(f"Failed to load Hugging Face model {model_id}: {e}")
        raise
    
    def huggingface_model(prompt: str) -> str:
        """Generate response using Hugging Face model with retry mechanism."""
        max_retries = 3
        base_delay = 1.0  # Base delay in seconds
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Hugging Face generation attempt {attempt + 1}/{max_retries}")
                
                # Format prompt for instruction-tuned models
                if "instruct" in model_id.lower() or "chat" in model_id.lower():
                    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
                else:
                    formatted_prompt = prompt
                
                logger.debug(f"Formatted prompt length: {len(formatted_prompt)} characters")
                
                # Generation parameters
                result = generator(
                    formatted_prompt,
                    max_new_tokens=800,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Extract generated text
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                else:
                    generated_text = str(result)
                
                # Clean up response
                generated_text = generated_text.strip()
                if generated_text.startswith('[/INST]'):
                    generated_text = generated_text[7:].strip()
                
                logger.debug(f"Successfully generated {len(generated_text)} characters on attempt {attempt + 1}")
                return generated_text
                
            except Exception as e:
                logger.warning(f"Hugging Face generation attempt {attempt + 1} failed: {e}")
                
                # If this is the last attempt, log error and return fallback
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} Hugging Face generation attempts failed. Final error: {e}")
                    return "Error: Model generation failed after multiple retries"
                
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay:.1f} seconds... (attempt {attempt + 2}/{max_retries})")
                
                # Sleep with exponential backoff
                time.sleep(delay)
        
        # This should never be reached due to the logic above, but included for safety
        return "Error: Unexpected failure in retry mechanism"
    
    return huggingface_model

