# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

def format_aime_prompt(question: str) -> str:
    return f"""Please reason step by step, and put your final answer within \\boxed{{}}

{question}
"""

def format_aime_answer(answer: str) -> str:
    return f"\\boxed{{{answer}}}"

def init_dataset():
    import random

    from datasets import load_dataset

    train_split = [{"input": format_aime_prompt(x["problem"]), "additional_context": {"solution": x["solution"]}, 
                    "answer": format_aime_answer(x["answer"])} for x in load_dataset("AI-MO/aimo-validation-aime")["train"]]
    random.Random(0).shuffle(train_split)
    test_split = [{"input": format_aime_prompt(x["problem"]), "answer": format_aime_answer(x["answer"])} for x in load_dataset("MathArena/aime_2025")["train"]]

    trainset = train_split[:len(train_split)//2]
    valset = train_split[len(train_split)//2:]
    testset = test_split * 5

    return trainset, valset, testset

if __name__ == "__main__":
    trainset, valset, testset = init_dataset()
    print(trainset[0])
    print(valset[0])
    print(testset[0])

    print("\n==============================================\n")

    print(trainset[0]["input"])
    print(valset[0]["input"])
    print(testset[0]["input"])

    print("\n==============================================\n")

    print(trainset[0]["answer"])
    print(valset[0]["answer"])
    print(testset[0]["answer"])