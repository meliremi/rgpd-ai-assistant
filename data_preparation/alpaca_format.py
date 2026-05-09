def to_alpaca_with_context(question, answer, context):
    return {
        "instruction": question,
        "input": f"Contexte : \n{context}",
        "output": answer
     } [cite: 73, 74, 75]