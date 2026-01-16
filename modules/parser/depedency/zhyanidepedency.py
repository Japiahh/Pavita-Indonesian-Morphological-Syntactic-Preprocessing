import re

from .module.find import FindDepedency

class ZhyaniDependencyParser:
    def __init__ (self):
        self.finder = FindDepedency()

    def dependency_parse(self, syntactic_tree):
        data_to_process = syntactic_tree
        if isinstance(syntactic_tree, tuple) and len(syntactic_tree) > 1:
            data_to_process = syntactic_tree[1] 

        if not data_to_process:
            return []

        sentences_list = self.sentence_split(data_to_process)
        
        final_results = []

        for idx, sentence in enumerate(sentences_list):
            if not sentence: continue
            
            dep_data = self.all_find(sentence)
            
            text_parts = []
            for t in sentence:
                if isinstance(t, tuple):
                    if isinstance(t[1], list): 
                        text_parts.extend([sub[0] for sub in t[1] if isinstance(sub, tuple)])
                    else: 
                        text_parts.append(t[0])

            sentence_output = {
                "sentence_id": idx + 1,
                "text": " ".join(text_parts),
                "dependencies": dep_data
            }
            final_results.append(sentence_output)

        return final_results

    def sentence_split(self, tokens):
        if not tokens:
            return []
        
        # Normalisasi input jadi list
        if not isinstance(tokens, list):
            if isinstance(tokens, tuple):
                tokens = list(tokens)
            else:
                return [] 

        sentence_endings = {'.', '?', '!'}
        valid_after_colon = {'PRP-PER', 'PRP-DEM', 'VB-ACT', 'VB-STAT', 'DT-DEF', 'DT-ORD'}
        
        sentences = []
        current_sentence = []

        i = 0
        while i < len(tokens):
            token = tokens[i]
            current_sentence.append(token)

            check_word = None
            check_tag = None

            if isinstance(token, tuple) and len(token) == 2 and isinstance(token[1], str):
                check_word = token[0]
                check_tag = token[1]

            elif isinstance(token, tuple) and len(token) == 2 and isinstance(token[1], list):
                label = token[0]
                children = token[1]
                if children and len(children) > 0:
                    first_child = children[0]
                    if isinstance(first_child, tuple) and len(first_child) >= 2:
                        check_word = first_child[0]
                        check_tag = first_child[1]

            if check_word and check_tag:
                tag_str = str(check_tag)

                if check_word in sentence_endings and tag_str.startswith("SYM"):
                    sentences.append(current_sentence)
                    current_sentence = []

                elif check_word == ':' and tag_str.startswith("SYM"):
                    pass

            i += 1

        if current_sentence:
            sentences.append(current_sentence)

        return sentences
    
    def all_find(self, syntactic_tree):
        finder = self.finder 
        
        dependency_components = {
            "root": None,
            "nsubj": None,
            "dobj": None,
            "xcomp": [],
            "punct": []
        }

        if hasattr(finder, 'find_root'):
            dependency_components["root"] = finder.find_root(syntactic_tree)

        if hasattr(finder, 'find_nsubj'):
            dependency_components["nsubj"] = finder.find_nsubj(syntactic_tree)

        if hasattr(finder, 'find_dobj'):
            dependency_components["dobj"] = finder.find_dobj(syntactic_tree)
        elif hasattr(finder, 'find_obj'):
            dependency_components["dobj"] = finder.find_obj(syntactic_tree)

        if hasattr(finder, 'find_xcomp'):
            dependency_components["xcomp"] = finder.find_xcomp(syntactic_tree)

        if hasattr(finder, 'find_punctuation'):
            dependency_components["punct"] = finder.find_punctuation(syntactic_tree)


        return dependency_components
