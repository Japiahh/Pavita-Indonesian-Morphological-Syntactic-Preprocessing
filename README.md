# Pavita : Indonesian Morphological Preprocessing (IMP)

Pavita is a modular Preprocessing pipeline designed specifically for the Indonesian language (Bahasa Indonesia). This project aims to provide a structured approach to text analysis by integrating tokenization, Part-of-Speech (POS) tagging, syntactic and depedency parsing into a cohesive workflow.

## Key Modules
Pavita is composed of several specialized modules, each named to represent a specific function within the pipeline:
* Chakaria (Tokenizer): A specialized tokenizer that handles Indonesian sentence structures and punctuation, preparing raw text for deeper analysis.
* Erisa (POS Tagger): The core tagging engine. It utilizes a hybrid approach (rule-based and dictionary-based) and features "Confix Fusion"—a capability to merge separated morphemes (e.g., se- + buah) into meaningful units.
* Zhyani (Parser):
  * Syntactic Parser: Constructs constituency trees to visualize sentence structure.
  * Dependency Parser: Analyzes grammatical relationships between words (Subject, Object, Root).
* Utils:
  * Sasmita (Tag Checker): A quality assurance module that monitors the tagging process. It identifies and logs "Unknown" (UNK) tokens, helping to refine the dictionary and logic over time.
