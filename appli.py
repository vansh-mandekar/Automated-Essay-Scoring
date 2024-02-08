import streamlit as st
from transformers import BertTokenizer
import torch
from torch.nn.functional import softmax
import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.6):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 19)  # Adjust the output dimension (19) based on your classification task
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

model = BertClassifier()  
model.load_state_dict(torch.load('model_checkpoint.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

def get_prediction(essay):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    inputs = tokenizer(essay, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    output = model(inputs['input_ids'], inputs['attention_mask'])
    
    predicted_class = torch.argmax(output).item()
    
    return predicted_class

def main():
    st.title("Automated Essay Scoring")
    essay_input = st.text_area("Enter your essay here:")
    
    if st.button("Score Essay"):
        prediction = get_prediction(essay_input)
        st.write(f"Predicted Score: {prediction}")
        
print(model)

if __name__ == "__main__":
    main()
