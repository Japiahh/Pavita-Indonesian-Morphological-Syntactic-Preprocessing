# Pavita : Indonesian Morphological Preprocessing (IMP)

Pavita is a modular Preprocessing pipeline designed specifically for the Indonesian language (Bahasa Indonesia). This project aims to provide a structured approach to text analysis by integrating tokenization, Part-of-Speech (POS) tagging, syntactic and depedency parsing into a cohesive workflow.

---
## Key Modules
Pavita is composed of several specialized modules, each named to represent a specific function within the pipeline:
* **Chakaria** (Tokenizer): A specialized tokenizer that handles Indonesian sentence structures and punctuation, preparing raw text for deeper analysis.
* **Erisa** (POS Tagger): The core tagging engine. It utilizes a hybrid approach (rule-based and dictionary-based) and features "Confix Fusion"a capability to merge separated morphemes (e.g., se- + buah) into meaningful units.
* **Zhyani** (Parser):
  * **Syntactic** Parser: Constructs constituency trees to visualize sentence structure.
  * **Dependency** Parser: Analyzes grammatical relationships between words (Subject, Object, Root).
* **Utils**:
  * **Sasmita** (Tag Checker): A quality assurance module that monitors the tagging process. It identifies and logs "Unknown" "UNK" tokens, helping to refine the dictionary and logic over time.

---
## Project Structure
```bash
Pavita/
|--- modules/
|   |--- tokenizer/chakaria.py       # Tokenization Logic
|   |--- postag/erisa.py             # POS Tagging & Fusion Logic
|   |    |--- data/                  # Lexicon (JSON) & Regex Rules
|   |--- parser/
|        |--- sintatic/zhyanisintatic.py
|        |--- depedency/zhyanidepedency.py
|--- utils/
|    |--- sasmita.py                 # Quality Control & Logging
|--- Pavita.py                       # Main Entry Point                      
|--- README.md                       # Documentation
```

---
## Installation
1. Clone
```bash
git clone https://github.com/Japiahh/Pavita-Indonesian-Morphological-Preprocessing.git
cd Pavita-Indonesian-Morphological-Preprocessing
```

2. Usage
You can utilize the main script, Pavita.py, to process text files. The system will read your input text and generate a clean, structured JSON output.

**Example Python Usage**:
```Python
from Pavita import PavitaIMP

# Initialize the pipeline
pavita = PavitaIMP()

# Process a text file and save the result
pavita.process_file("input.txt", "results/output.json")
```

## Output
```JSON
[
    {
        "raw_text": "Ayo, duduk dan berbincang denganku. Aku ingin tahu semua tentang harimu~",
        "token": [ "ayo", ",", "duduk", "dan", "berbincang", "denganku", ".", "aku", "ingin", "tahu", "semua", "tentang", "harimu"],
        "tagged": [["ayo", "INT-DISC"], [ ",", "SYM-COM" ], [ "duduk", "VB-ACT"], ["dan", "CON-COR"], ["berbincang", "VB-STAT"], ... ],
        "syntax_tree": [
            "S",
            [
                [
                    "PP",
                    [
                        [
                            "ayo",
                            "INT-DISC"
                        ]
                    ]
                ],
                [
                    "PUNCT",
                    [
                        [
                            ",",
                            "SYM-COM"
                        ]
                    ]
                ],
                [
                    "VP",
                    [
                        [
                            "duduk",
                            "VB-ACT"
                        ]
                    ]
                ],
                [
                    "CONJ",
                    [
                        [
                            "dan",
                            "CON-COR"
                        ]
                    ]
                ],
                [
                    "VP",
                    [
                        [
                            "berbincang",
                            "VB-STAT"
                        ],
                        [
                            "NP",
                            [
                                [
                                    "denganku",
                                    "NN-COM"
                                ]
                            ]
                        ]
                    ]
                ],
                ...
        ],
        "dependency_graph": [
            {
                "sentence_id": 1,
                "text": "ayo , duduk dan berbincang NP .",
                "dependencies": {
                    "root": [
                        "duduk",
                        "VB-ACT"
                    ],
                    "nsubj": null,
                    "dobj": [
                        "denganku",
                        "NN-COM"
                    ],
                    "xcomp": [],
                    "punct": [
                        [
                            ",",
                            "SYM-COM"
                        ],
                        [
                            ".",
                            "SYM-DOT"
                        ]
                    ]
                }
            }
        ]
    },
    ...
]
```

---
Built with Risa and coffee, thanks Gemie. 
