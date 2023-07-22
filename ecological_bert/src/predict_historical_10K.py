from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import torch
from tqdm import tqdm
import os
from PyPDF2 import PdfReader
import re
import nltk

def extract(pdf_file:str):
	file_read = PdfReader(pdf_file)
	pdf_text=""
	for page in file_read.pages:
		content = page.extract_text()
		pdf_text += content
	return pdf_text

def cleanhtml(raw_html):
	cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
	cleantext = re.sub(cleaner, '', raw_html)
	return cleantext

def remove_numbers(text):
	result = re.sub(r'\d+', '', text)
	return result

def multiples(m, count):
	x = []
	for i in range(count):
		x = x + [i*m]
	return x

class SequenceClassificationDataset(Dataset):
	def __init__(self, x, tokenizer):
		self.examples = x
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, idx):
		return self.examples[idx]
	def collate_fn(self, batch):
		model_inputs = self.tokenizer([i for i in batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
		return {"model_inputs": model_inputs}


if __name__ == "__main__":
	company_list_names_year=[]
	#Clean and import the 10K reports (here from 2019 to 2022)
	#To change if want to import something from another file:
	company_list_dir = os.getcwd() + '/10-K_time_2019-2022'
	#To keep if want to import from another file:
	company_list_pdf = os.listdir(company_list_dir)
	for i in company_list_pdf:
		company_list_names_year=company_list_names_year+[i[4:-4]]
	count = 0
	for i in company_list_pdf:
		if True:
			document = PdfReader(open(company_list_dir + '/'+ i, 'rb'))
			pages = []
			for page in range(len(document.pages)):
				pageObj = document.pages[page]
				pages.append(pageObj.extract_text().replace('\n',''))
			text2=""
			for i in range (len(pages)):
				text2=text2+pages[i]
			x=nltk.sent_tokenize(text2)

			#Perform BERT on each
			parser = argparse.ArgumentParser()
			parser.add_argument('--model_name', type=str, default='climatebert/environmental-claims')
			parser.add_argument('--outfile_name', type=str, default='ETZH_'+company_list_names_year[count] +'.csv')
			count += 1
			args = parser.parse_args()

			device = "cuda" if torch.cuda.is_available() else "cpu"

			# load model and tokenizer
			try:
				tokenizer = AutoTokenizer.from_pretrained(args.model_name)
			except:
				tokenizer = AutoTokenizer.from_pretrained("roberta-base")

			model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)

			predict_dataset = SequenceClassificationDataset(x, tokenizer)

			outputs = []
			probs = []
			with torch.no_grad():
				model.eval()
				for batch in tqdm(DataLoader(predict_dataset, batch_size=32, collate_fn=predict_dataset.collate_fn)):
					output = model(**batch["model_inputs"])
					logits = output.logits
					outputs.extend(logits.argmax(dim=1).tolist())
					probs.extend(logits.softmax(dim=1)[:,1].tolist())

			# save to outfile
			import pandas as pd

			df = pd.DataFrame(list(zip(x, outputs, probs)), columns=["sentence", "classification", "probability"])
			df.to_csv(args.outfile_name, index=False)
		else:
			count=count+1

