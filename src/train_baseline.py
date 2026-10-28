from pathlib import Path
import json 
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import(
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
base_dir=Path(__file__).resolve().parent.parent
data_dir=base_dir/"data"
model_dir=base_dir/"models"
report_dir=base_dir/"reports"
model_dir.mkdir(exist_ok=True)
report_dir.mkdir(exist_ok=True)
train_df=pd.read_csv(data_dir/"train.csv")
validation_df=pd.read_csv(data_dir/"validation.csv")
test_df=pd.read_csv(data_dir/"test.csv")
results={}
tasks=["category","priority"]
for task in tasks:
    print(f"\nTraining model for {task}...")
    X_train=train_df["text"]
    y_train=train_df[task]
    X_val=validation_df["text"]
    y_val=validation_df[task]
    X_test=test_df["text"]
    y_test=test_df[task]
    pipeline=Pipeline([
        ("tfidf",TfidfVectorizer(
            strip_accents="unicode",
            sublinear_tf=True
        )),
        ("classifier",LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])
    parameters=[
        {
            "tfidf__analyzer":["word"],
            "tfidf__ngram_range":[(1,1),(1,2)],
            "classifier__C":[0.1,1,10],
        },
        {
            "tfidf__analyzer":["char_wb"],
            "tfidf__ngram_range":[(3,5)],
            "classifier__C":[0.1,1,10],
        },
    ]
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    search=GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=5,
        error_score="raise"
    )
    start=time.perf_counter()
    search.fit(X_train,y_train)
    training_time=time.perf_counter()-start
    model=search.best_estimator_
    start=time.perf_counter()
    val_predictions=model.predict(X_val)
    val_time=time.perf_counter()-start
    val_accuracy=accuracy_score(y_val,val_predictions)
    val_f1=f1_score(
        y_val,val_predictions,
        average="macro",zero_division=0
    )
    print("\nBest parameters:",search.best_params_)
    print("Cross-validation macro F1:",round(search.best_score_,4))
    print("validation accuracy:",round(val_accuracy,4))
    print("validation macro F1:",round(val_f1,4))
    test_predictions=model.predict(X_test)
    test_accuracy=accuracy_score(y_test,test_predictions)
    test_f1=f1_score(
        y_test,test_predictions,
        average="macro",zero_division=0
    )
    print("test accuracy:",round(test_accuracy,4))
    print("test macro F1:",round(test_f1,4))
    print("\nValidation classification report:")
    print(classification_report(
        y_val,val_predictions,zero_division=0
    ))
    joblib.dump(model,model_dir/f"{task}_model.joblib")
    labels=list(model.classes_)
    matrix=confusion_matrix(y_val,val_predictions,labels=labels)
    plt.figure(figsize=(7,5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues"
    )
    plt.xlabel("prediction label")
    plt.ylabel("Actual label")
    plt.title(f"{task.title()} Confusion matrix")
    plt.tight_layout()
    plt.savefig(report_dir/f"{task}_confusion_matrix.png")
    plt.close()
    probablities=model.predict_proba(X_val)
    confidence=probablities.max(axis=1)
    errors=pd.DataFrame({
        "text":X_val.to_numpy(),
        "actual":y_val.to_numpy(),
        "predicted":val_predictions,
        "confidence":confidence
    })
    errors=errors[errors["actual"]!=errors["predicted"]]
    errors=errors.sort_values("confidence",ascending=False)
    errors.to_csv(report_dir/f"{task}_errors.csv",index=False)
    results[task]={
        "best_parameters":search.best_params_,
        "cross_validation_macro_f1":float(search.best_score_),
        "validation_accuracy":float(val_accuracy),
        "validation_macro_f1":float(val_f1),
        "test_accuracy":float(test_accuracy),
        "test_macro_f1":float(test_f1),
        "training_sceonds":round(training_time,3),
        "validation_inference_ms_per_ticket":round(
            val_time/len(X_val)*1000,3
        )
    }
with open(report_dir/"baseline_results.json","w")as file:
    json.dump(results,file,indent=4)

