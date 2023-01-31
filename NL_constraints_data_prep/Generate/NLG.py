from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusForConditionalGeneration, PegasusTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

para_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
para_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(device)

# summarizer_name = 'google/pegasus-reddit_tifu'
# tokenizer = PegasusTokenizer.from_pretrained(summarizer_name)
# model = PegasusForConditionalGeneration.from_pretrained(summarizer_name).to(device)


def get_summary(input_text):
    batch = tokenizer([input_text], truncation=True, padding='longest', max_length=1024, return_tensors="pt").to(
        device)
    gen_out = model.generate(**batch, max_length=1024, num_beams=5, num_return_sequences=1, temperature=1.5)
    output_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    return output_text


def get_paraphrase(sentence):
    text = "paraphrase: " + sentence + " </s>"

    encoding = para_tokenizer.encode_plus(text, padding=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    outputs = para_model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.5,
        early_stopping=True,
        temperature = 1.5,
        no_repeat_ngram_size=3,
        num_return_sequences=5
    )
    generated_outputs = []
    for output in outputs:
        line = para_tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_outputs.append(line)
    # print(generated_outputs)
    # sdfsdf
    return generated_outputs[0]
