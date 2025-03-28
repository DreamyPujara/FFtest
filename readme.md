import json
from langchain.schema import Document

# Load JSON data
with open("dataset.json", "r") as file:
    data = json.load(file)

# Convert to Document format
documents = []
for entry in data:
    instruction = entry.get("instruction", "")
    output = entry.get("output", "")
    page_content = f"***question***: {instruction}\n\n***answer***: {output}"
    metadata = {"source": "dataset"}
    
    document = Document(page_content=page_content, metadata=metadata)
    documents.append(document)

serializable_docs = [
    {"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents
]

# Save to a JSON file
output_file = "documents.json"
with open(output_file, "w") as file:
    json.dump(serializable_docs, file, indent=4)

print(f"Documents saved to {output_file}")

 DEFAULT FOR PER_ASG_FTE_VALUE IS 1
 DEFAULT FOR PER_ASG_JOB_NAME IS ' '\n\n
 PRORATIONFACTOR = 1\n\n
 IF (PER_ASG_JOB_NAME = 'Wealth Management Consultant' OR PER_ASG_JOB_NAME = 'Trade Associate')\nTHEN\n 
 (PRORATIONFACTOR = ROUND(PER_ASG_FTE_VALUE,2))\n\n
 RETURN PRORATIONFACTOR