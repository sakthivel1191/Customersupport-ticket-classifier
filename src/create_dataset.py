from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import train_test_split
base_dir=Path(__file__).resolve().parent.parent
data_dir=base_dir/"data"
data_dir.mkdir(exist_ok=True)
random.seed(42)
messages={
    "Billing":[
            "I was charged twice for my order",
            "My refund has not arrived",
            "there is an incorrect amount on my bill",
            "my payment failed but money was deducted",
            "please explain my subscription charge"
        ],
        "Technical":[
            "The application  keep crashing ",
            "I cannot open the website",
            "the page shows an error message",
            "the application is very slow",
            "my device is not connecting"
        ],
        "Account":[
            "I cannot log in to my account"
            "Please reset my account password",
            "I want to update my email address",
            "my account has been locked",
            "I cannot verify my account"
        ],
        "Shipping":[
            "my order has not arrived",
            "the delivery is delayed",
            "I received the wrong product"
        ],
        "Cancellation":[
            "please cancel my subscription",
            "I want to my cancel my order",
            "Stop my automatic renewal"
        ]
}
priority_messages={
    "Low":[
        "when you have time",
        "this is not urgent",
        "please help me when possible"
    ],
    "Medium":[
        "please resolve this soon",
        "i need an update today",
        "this is affecting my work"
    ],
    "High":[
        "this is urgent and needs immediate attention",
        "i cannot continue using the service ",
        "please resolve this immediately"
    ]
}
rows=[]
for category,examples in messages.items():
    for priority,priority_example in priority_messages.items():
        for number in range(50):
            text=(
                f"{random.choice(examples)}."
                f"{random.choice(priority_example)}."
                f"Reference{ category[:2]}-{priority[:2]}-{number:03d}"
            )
            rows.append({
                "text":text,
                "category":category,
                "priority":priority
            })

df=pd.DataFrame(rows)
df["text"]=df["text"].str.lower().str.strip()
df=df.drop_duplicates(subset=["text"]).reset_index(drop=True)
df.to_csv(data_dir/"tickets.csv",index=False)
print("Total rows:",len(df))
print("\nCategory counts:")
print(df["category"].value_counts().sort_index())
print("\nPriority counts")
print(df["priority"].value_counts().sort_index())
combined_labels=df["category"]+"_"+df["priority"]
train_df,remaining_df=train_test_split(
    df,test_size=0.30,
    random_state=42,
    stratify=combined_labels
)
remaining_labels=(
    remaining_df["category"]+"_"+remaining_df["priority"]
)
validation_df,test_df=train_test_split(
    remaining_df,
    test_size=0.50,
    random_state=42,
    stratify=remaining_labels
)

train_df.to_csv(data_dir/"train.csv",index=False)
validation_df.to_csv(data_dir/"validation.csv",index=False)
test_df.to_csv(data_dir/"test.csv",index=False)
print("\n Dataset split")
print("Training:",len(train_df))
print("Validation",len(validation_df))
print("Test:",len(test_df))