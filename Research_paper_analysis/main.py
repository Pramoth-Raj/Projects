from mod.classifier import predict_conference_reasoning, predict_paper_publishability

# Task1: Classifying Research Papers as Publishable or Non-Publishable Using Feature Extraction and Clustering

path = "sample.pdf"
sample_groq_api_key = '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'

publishability = predict_paper_publishability(path)

if publishability:
    conference, rationale = predict_conference_reasoning(path, sample_groq_api_key)
    print(f"The research paper is publishable and should be submitted to the {conference} conference.")
    print(f"Rationale: {rationale}")
else:
    print("The research paper is not publishable.")
