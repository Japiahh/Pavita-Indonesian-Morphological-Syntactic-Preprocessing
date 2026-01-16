import re
import math
from collections import defaultdict

from .module.handle_ambiguity import Handleambiguity

from .data import regex_patterns

class ErisaPOSTagger :
    def __init__(self, model=None, mode="", verbose=False):
        self.model = model
        self.mode = mode
        self.verbose = verbose
        self.rules = self.load_rules()
        self.ambiguity_handler = Handleambiguity()

    def load_rules(self):
        self.regex_patterns = regex_patterns
        return {
            "regex_patterns": self.regex_patterns
        }

    def posttag(self, tokens):
        regex_tags = {}
        try:
            regex_results = self.regex_tagging(tokens)
            if regex_results:
                for i, res in enumerate(regex_results):
                    if res and isinstance(res, tuple) and len(res) == 2:
                        _, tag = res
                        if tag is not None:
                            regex_tags[i] = tag
        except Exception:
            pass 

        lexicon_tags = {}
        try:
            for i, token in enumerate(tokens):
                tag = self.lookup_lexicon(token)
                if tag:
                    lexicon_tags[i] = tag
        except Exception:
            pass

        combined_tags = {}
        for i in range(len(tokens)):
            if i in lexicon_tags:
                combined_tags[i] = lexicon_tags[i]
            elif i in regex_tags:
                combined_tags[i] = regex_tags[i]

        token_tag_pairs = [(tokens[i], combined_tags.get(i)) for i in range(len(tokens))]

        try:
            token_tag_pairs = self.merge_tokens(token_tag_pairs)
        except Exception:
            pass

        for i, (token, tag) in enumerate(token_tag_pairs):
            if tag is None:
                try:
                    inferred_list = self.infer_tag([token])
                    if inferred_list and len(inferred_list) > 0:
                        first_item = inferred_list[0]
                        if isinstance(first_item, tuple) and len(first_item) > 1:
                            inferred_tag = first_item[1]
                            if inferred_tag:
                                token_tag_pairs[i] = (token, inferred_tag)
                except Exception:
                    pass

        try:
            rule_based = self.rule_based_tagging(token_tag_pairs)
            if rule_based and len(rule_based) == len(token_tag_pairs):
                for i, (token, tag) in enumerate(rule_based):
                    if tag:
                        token_tag_pairs[i] = (token, tag)
        except Exception:
            pass

        merged_tokens = [token for token, _ in token_tag_pairs]
        viterbi_result = []
        try:
            viterbi_result = self.viterbi(merged_tokens)
        except Exception:
            viterbi_result = []

        final_tags = []
        for i, (token, tag) in enumerate(token_tag_pairs):
            if tag:
                final_tags.append((token, tag))
            else:
                vt_tag = "NN-COM" 

                if i < len(viterbi_result):
                    res = viterbi_result[i]
                    if res and res != 'UNK':
                        vt_tag = res
                
                final_tags.append((token, vt_tag))
                
        try:
            final_tags = self.posthandle(final_tags)
        except Exception:
            pass

        return final_tags
    
    def posthandle(self, tokens_with_tags):
        fused_tokens_with_tags = self.handle_confix_fusion(tokens_with_tags)

        disambiguated = self.ambiguity_handler.handle(fused_tokens_with_tags)

        return disambiguated

    def rule_based_tagging(self, tokens):
        tagged = []

        if all(isinstance(t, str) for t in tokens):
            tokens = [(t, None) for t in tokens]

        for token, tag in tokens:
            if tag is not None:
                tagged.append((token, tag))
                continue

            if token == "di":
                tagged.append((token, "IN"))
            elif token.startswith("ber-") or token.startswith("me-") or token.startswith("di-"):
                tagged.append((token, "VB-ACT"))
            elif token.startswith("ter-"):
                tagged.append((token, "VB-STAT"))
            elif token.startswith("se-") and len(token) > 3:
                tagged.append((token, "DT-DEF"))
            elif token.endswith("-kan"):
                tagged.append((token, "VB-CAUS"))
            elif token.endswith("-nya"):
                tagged.append((token, "PRP-POSS"))
            elif token.endswith("-an"):
                tagged.append((token, "NN-COM"))
            elif token.endswith("-i") and len(token) > 3:
                tagged.append((token, "VB-STAT"))
            else:
                tagged.append((token, None))

        return tagged
    
    def regex_tagging(self, tokens):
        tagged = []
        for token in tokens:
            tag_assigned = False
            for pattern, tag in self.regex_patterns.items():
                if re.fullmatch(pattern, token):
                    tagged.append((token, tag))
                    tag_assigned = True
                    break
            if not tag_assigned:
                tagged.append((token, None))
        return tagged

    def infer_tag(self, tokens):
        inferred = []
        for token in tokens:
            tag = None

            if token.startswith("ber-") or token.startswith("me-") or token.startswith("di-"):
                tag = "VB-ACT"
            elif token.startswith("ter-"):
                tag = "VB-STAT"
            elif token.startswith("ke-") and token.endswith("-an"):
                tag = "NN-ABST" 
            elif token.endswith("-kan"):
                tag = "VB-CAUS"
            elif token.endswith("-nya"):
                tag = "PRP-POSS"
            elif token.endswith("-an"):
                tag = "NN-COM"
            elif token.endswith("-lah"):
                tag = "MOD-EMPH"
            elif token.endswith("-i"):
                tag = "VB-STAT"
            elif "-" in token:
                parts = token.split("-")
                if len(parts) == 2 and parts[0] == parts[1]:
                    tag = "NN-COLL"
                else:
                    tag = "NN-COM" 

            if tag is None:
                tag = "<UNK>" 

            inferred.append((token, tag)) 

        return inferred

    def get_possible_tags(self, token):
        possible_tags = set()

        for tag, patterns in self.regex_patterns.items():
            for pattern in patterns:
                if re.fullmatch(pattern, token):
                    possible_tags.add(tag)

        if not possible_tags:
            if token.startswith(("me-", "ber-", "di-", "men-", "mem-", "ter-")):
                possible_tags.add("VB-ACT")
            if token.endswith("-an"):
                possible_tags.add("NN-COM")
            if token.endswith(("-nya", "-ku")):
                possible_tags.add("PRP-POSS")
            if "-" in token:
                parts = token.split("-")
                if len(parts) == 2 and parts[0] == parts[1]:
                    possible_tags.add("NN-COLL")

        return list(possible_tags)
    
    def merge_tokens(self, token_tag_pairs):
        merged = []
        i = 0
        while i < len(token_tag_pairs):
            token, tag = token_tag_pairs[i]

            if token.lower() == "sama-sama":
                merged.append(("sama-sama", "INT-RESP"))
                i += 1
                continue

            if (i + 2) < len(token_tag_pairs):
                t1, _ = token_tag_pairs[i]
                t2, _ = token_tag_pairs[i + 1]
                t3, _ = token_tag_pairs[i + 2]
                if t1.lower() == "sama" and t2 == "-" and t3.lower() == "sama":
                    merged.append(("sama-sama", "INT-RESP"))
                    i += 3
                    continue
                elif t2 == "-" and t1 == t3:
                    merged.append((f"{t1}-{t3}", "NN-REPEAT"))
                    i += 3
                    continue

            if (i + 1) < len(token_tag_pairs):
                t1, _ = token_tag_pairs[i]
                t2, _ = token_tag_pairs[i + 1]
                if t1.lower() == "sama" and t2.lower() == "sama":
                    merged.append(("sama-sama", "INT-RESP"))
                    i += 2
                    continue

            if (i + 2) < len(token_tag_pairs):
                next_token, next_tag = token_tag_pairs[i + 1]
                next2_token, next2_tag = token_tag_pairs[i + 2]
                if next_token == '-' and token != next2_token:
                    merged.append((token, tag))
                    merged.append(('-', 'SYM-DASH'))
                    i += 2
                    continue

            merged.append((token, tag))
            i += 1

        return merged

    def viterbi(self, tokens):
        V = [{}] 
        path = {}  
        first_token = tokens[0]
        first_tags = self.get_possible_tags(first_token)
        for tag in first_tags:
            V[0][tag] = self.score("<s>", tag) 
            path[tag] = [tag]

        for t in range(1, len(tokens)):
            V.append({})
            new_path = {}
            curr_token = tokens[t]
            possible_tags = self.get_possible_tags(curr_token)

            for curr_tag in possible_tags:
                best_score = float('-inf')
                best_prev_tag = None

                for prev_tag in V[t - 1]:
                    score = V[t - 1][prev_tag] + self.score(prev_tag, curr_tag)

                    if score > best_score:
                        best_score = score 
                        best_prev_tag = prev_tag

                if best_prev_tag is not None:
                    V[t][curr_tag] = best_score
                    new_path[curr_tag] = path[best_prev_tag] + [curr_tag]

            path = new_path

        if not V[-1]:
            return ["<UNK>"] * len(tokens) 

        max_final_tag = max(V[-1], key=lambda tag: V[-1][tag])
        return path[max_final_tag]

    def handle_confix_fusion(self, tokens_with_tags):
        fused = []
        i = 0
        n = len(tokens_with_tags)

        valid_prefixes = {"di", "me", "ber", "ter", "mem", "men", "meng", "ke", "pe", "se", "pen", "pem", "per"}
        valid_suffixes = {"i", "kan", "an", "nya", "lah", "kah", "ku", "mu", "pun"}
        
        double_suffix_map = {
            ("an", "nya"): "NN-COM",
            ("kan", "nya"): "VB-ACT",
            ("i", "lah"): "VB-ACT",
            ("kan", "lah"): "VB-ACT",
            ("an", "ku"): "NN-COM",
            ("an", "mu"): "NN-COM",
        }

        while i < n:
            curr_tok, curr_tag = tokens_with_tags[i]

            if i + 3 < n:
                p_tok, p_tag = tokens_with_tags[i]
                r_tok, r_tag = tokens_with_tags[i+1]
                s1_tok, s1_tag = tokens_with_tags[i+2]
                s2_tok, s2_tag = tokens_with_tags[i+3]

                if (p_tok.endswith('-') and s1_tok.startswith('-') and s2_tok.startswith('-')):
                    c_pref = p_tok.strip('-')
                    c_suf1 = s1_tok.strip('-')
                    c_suf2 = s2_tok.strip('-')

                    if c_pref in valid_prefixes:
                        new_token = c_pref + r_tok + c_suf1 + c_suf2

                        new_tag = "VB-ACT"
                        if c_pref == "di": new_tag = "VB-PASS"
                        elif c_pref in ["ber", "ter"]: new_tag = "VB-STAT"
                        elif c_pref in ["pe", "pen", "pem", "per"]: new_tag = "NN-COM"
                        
                        fused.append((new_token, new_tag))
                        i += 4
                        continue

            if i + 2 < n:
                p_tok, p_tag = tokens_with_tags[i]
                r_tok, r_tag = tokens_with_tags[i+1]
                s_tok, s_tag = tokens_with_tags[i+2]

                if (p_tok.endswith('-') and s_tok.startswith('-')):
                    c_pref = p_tok.strip('-')
                    c_suf = s_tok.strip('-')
                    
                    if c_pref in valid_prefixes:
                        new_token = c_pref + r_tok + c_suf

                        new_tag = "VB-ACT"
                        if c_pref == "di": new_tag = "VB-PASS"
                        elif c_pref in ["ber", "ter"]: new_tag = "VB-STAT"
                        elif c_pref == "ke" and c_suf == "an": new_tag = "NN-ABST"
                        elif c_pref in ["pe", "pen", "pem", "per"]: new_tag = "NN-COM"
                        elif c_pref == "se":
                            if c_suf == "nya": new_tag = "ADV-ATT"
                            else: new_tag = "NN-COM"

                        fused.append((new_token, new_tag))
                        i += 3
                        continue

            if i + 2 < n:
                r_tok, r_tag = tokens_with_tags[i]
                s1_tok, s1_tag = tokens_with_tags[i+1]
                s2_tok, s2_tag = tokens_with_tags[i+2]

                if (s1_tok.startswith('-') and s2_tok.startswith('-')):
                    c_suf1 = s1_tok.strip('-')
                    c_suf2 = s2_tok.strip('-')

                    if (c_suf1, c_suf2) in double_suffix_map:
                        new_token = r_tok + c_suf1 + c_suf2
                        new_tag = double_suffix_map[(c_suf1, c_suf2)]
                        fused.append((new_token, new_tag))
                        i += 3
                        continue

            if i + 1 < n:
                p_tok, p_tag = tokens_with_tags[i]
                r_tok, r_tag = tokens_with_tags[i+1]

                if p_tok.endswith('-'): 
                    c_pref = p_tok.strip('-')
                    
                    if c_pref in valid_prefixes:

                        if not r_tag.startswith("SYM"):
                            new_token = c_pref + r_tok

                            new_tag = "VB-ACT"
                            if c_pref == "di": new_tag = "VB-PASS"
                            elif c_pref in ["ber", "ter"]: new_tag = "VB-STAT"
                            elif c_pref in ["pe", "pen", "pem", "per"]: new_tag = "NN-COM"
                            elif c_pref == "se":

                                if r_tok in ["buah", "orang", "ekor", "kali"]: 
                                    new_tag = "DT-NUM"
                                else:
                                    new_tag = "ADV-ATT" 

                            elif c_pref == "ke":
                                if r_tag == "DT-NUM": new_tag = "DT-ORD" 
                                else: new_tag = "NN-COM"

                            fused.append((new_token, new_tag))
                            i += 2
                            continue

            if i + 1 < n:
                r_tok, r_tag = tokens_with_tags[i]
                s_tok, s_tag = tokens_with_tags[i+1]

                if s_tok.startswith('-'):
                    c_suf = s_tok.strip('-')
                    if c_suf in valid_suffixes:
                        new_token = r_tok + c_suf

                        new_tag = r_tag 
                        if c_suf == "an": new_tag = "NN-COM"
                        elif c_suf == "nya": 
                            if r_tag.startswith("VB"): new_tag = "NN-COM" 
                            else: new_tag = "NN-COM"
                        elif c_suf in ["kan", "i"]: new_tag = "VB-ACT"
                        elif c_suf in ["ku", "mu"]: new_tag = "NN-COM"
                        
                        fused.append((new_token, new_tag))
                        i += 2
                        continue

            fused.append(tokens_with_tags[i])
            i += 1

        return fused
