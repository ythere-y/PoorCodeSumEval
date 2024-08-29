import torch 
import torch.utils.data as data
import tokenize
import io
import jsonlines
from transformers import RobertaTokenizer, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, AutoTokenizer

use_cuda = torch.cuda.is_available()


def remove_docstring(source):
    return source[:source.find('"""')] + source[source.rfind('"""') + 3:]

def remove_docs(source):
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    return out



class SummaryLlamaData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        # 1. Initialize file path or list of file names.
        self.data = []
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                self.data.append(item)
        self.tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-hf')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    def __getitem__(self, offset):
        code = remove_docs(self.data[offset]['code'])
        # code = ' '.join(self.data[offset]['code_tokens'])
        # print(code)
        idx = code.find(':') + 1
        text = code[:idx + 1] + "    \"\"\"  <FILL_ME>\n" + code[idx + 1:]
        # input = tokenizer(remove_docs(self.data[offset]['code']), return_tensors="pt", padding="max_length", truncation=True, max_length=256)
        input = self.tokenizer(text, return_tensors="pt")
        input_ids = input['input_ids'][0]
        input_mask = input['attention_mask'][0]
        docstring = ' '.join(self.data[offset]['docstring_tokens'])
        if '.' in docstring:
            docstring = docstring[:docstring.find('.')]
        target = self.tokenizer(docstring, return_tensors="pt")
        target_ids = target['input_ids'][0]
        target_mask = target['attention_mask'][0]
        return input_ids, input_mask, target_ids, target_mask

    def __len__(self):
        return len(self.data)
    
class SummaryLlamaInstrcutData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        # 1. Initialize file path or list of file names.
        self.data = []
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                self.data.append(item)
        self.data = self.data
        self.tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Instruct-hf')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    def __getitem__(self, offset):
        system = 'Generate docstring for the code in 10 words. '
        user = ' '.join(self.data[offset]['code_tokens']) + '\n"""Please fill this sentence "The goal of this function is to " in 10 words: '
        prompt = f"<s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user}[/INST]"

        input = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = input['input_ids'][0]
        input_mask = input['attention_mask'][0]
        docstring = ' '.join(self.data[offset]['docstring_tokens'])
        if '.' in docstring:
            docstring = docstring[:docstring.find('.') + 1]
        elif docstring[-1].isalpha():
            docstring = docstring + ' .'
        target = self.tokenizer(docstring, return_tensors="pt")
        target_ids = target['input_ids'][0]
        target_mask = target['attention_mask'][0]
        return input_ids, input_mask, target_ids, target_mask

    def __len__(self):
        return len(self.data)


class SummaryStarcoderData(data.Dataset):
    """
    Dataset that has binary samples.
    """
    def __init__(self, file_name):
        # 1. Initialize file path or list of file names.
        self.data = []
        with jsonlines.open(file_name, 'r') as reader:
            for item in reader:
                self.data.append(item)
        self.tokenizer = AutoTokenizer.from_pretrained('bigcode/starcoderbase', token='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')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
    def __getitem__(self, offset):
        code = remove_docs(self.data[offset]['code'])

        idx = code.find(':') + 1
        text = f"<fim_prefix>{code[:idx + 1]}\n    \"\"\"<fim_suffix>\"\"\"\n    {code[idx + 1:]}<fim_middle>"
        # input = tokenizer(remove_docs(self.data[offset]['code']), return_tensors="pt", padding="max_length", truncation=True, max_length=256)
        input = self.tokenizer(text, return_tensors="pt")
        input_ids = input['input_ids'][0]
        input_mask = input['attention_mask'][0]
        docstring = ' '.join(self.data[offset]['docstring_tokens'])
        if '.' in docstring:
            docstring = docstring[:docstring.find('.')]
        target = self.tokenizer(docstring, return_tensors="pt")
        target_ids = target['input_ids'][0]
        target_mask = target['attention_mask'][0]
        return input_ids, input_mask, target_ids, target_mask

    def __len__(self):
        return len(self.data)
