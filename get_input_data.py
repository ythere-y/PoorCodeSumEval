from tokenizer import TokeNizer

TN = TokeNizer("Python")

code = "if a.isEmpty():"
print(TN.getPureTokens(code))
