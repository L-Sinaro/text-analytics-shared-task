import pandas as pd
import ollama
import re
import json

def extract_json_from_text(text):
    json_blocks = re.search(r'\{.*\}', text, re.DOTALL)
    if json_blocks is not None:
        return json_blocks.group(0).strip()
    
    json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    for block in json_blocks:
        block = block.strip()
        if not block.startswith('{'):
            block = '{' + block + '}'
        try:
            # json.loads(block)  # validate
            return block
        except json.JSONDecodeError:
            continue

    print("no match:", text)
    return None

label_to_id = {
    "statement": 0,
    "question": 1,
    "floorgrabber": 2, 
    "fackchannel": 3,
    "disruption": 4
}

def convert_label_to_id(label, mapping):
    return mapping.get(label.strip(), -1)  # -1 for any unexpected label

# Load and clean the dataset
df = pd.read_csv("data/mrda/test.csv")
df.columns = df.columns.str.strip().str.lower()  # standardize column names

# Validate expected columns
required_columns = {"speaker", "text", "act", "conv_id"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file must contain the following columns: {required_columns}")

# Add a context column (previous utterance in same conversation)
df["context"] = ""

# Construct context: all previous utterances in the same conversation
for conv_id, group in df.groupby("conv_id"):
    history = []
    for idx in group.index:
        df.at[idx, "context"] = " ".join(history)
        history.append(f'{df.at[idx, "speaker"]}: {df.at[idx, "text"]}')

# Function to construct the prompt
def create_prompt(current_text, context_text, speaker_id):
    return f"""Classify the dialogue act of the following utterance, considering the previous context.

Conversation so far: "{context_text}"
Now Speaker {speaker_id} says: "{current_text}"

Only output one of the following labels, without the definition:

- Statement: Conveys information or opinions.
- Question: Seeks information or clarification.
- Floorgrabber: Attempts to take or maintain the conversational floor.
- Backchannel: Provides feedback or acknowledgment without taking the floor (e.g., "uh-huh," "I see").
- Disruption: Includes interruptions, abandoned utterances, or other disruptions to the conversational

Answer with only the label using this format: 
```json
"utterance": string(label)
```
"""

def classify_with_context(text, context, speaker):
    prompt = create_prompt(text, context, speaker)
    try:
        response = ollama.chat(
            model="deepseek-r1:14b",
            messages=[
                {"role": "system", "content": "You are a dialogue act classifier."},
                {"role": "user", "content": prompt}
            ]
        )
        message = response.message.content.strip()
        print("prompt: \n", prompt)
        print("response: \n", message)
        json_res = json.loads(extract_json_from_text(message))
        # print("json:", json_res["utterance"].lower())
        # return json_res["utterance"].lower()
        return json_res["utterance"].lower()
    except Exception as e:
        print("Error:", e)
        print("answer:", message)
        return "ERROR"

# Run predictions
predicted_labels = []
for i, row in df.iterrows():
    pred = classify_with_context(row["text"], row["context"], row["speaker"])
    # print(f"[{i+1}/{len(df)}] \"{row['text']}\" => {pred}")
    predicted_labels.append(pred)

# Add predictions to DataFrame
df["predicted_act"] = predicted_labels

# Convert predicted labels
df["predicted_act_id"] = df["predicted_act"].apply(lambda x: convert_label_to_id(x, label_to_id))

# Keep only selected columns
final_columns = ["speaker", "text", "act", "conv_id", "predicted_act_id"]
df_final = df[final_columns]

# Save to CSV
df_final.to_csv("final_predicted_dialogue_acts.csv", index=False)
print("successfully exported csv results")
