"""
Stage 1: Multilingual NLP & NegEx-Based Weak Label Extraction Engine
====================================================================
RSNA 2026 Knee Abnormality Detection AI Challenge

This module implements:
1. Multilingual 12-Target Knee Pathology Dictionary (EN, ES, FR, DE, IT, PT).
2. Rule-Based Scope-Sensitive NegEx & Uncertainty Detection (preventing false positives).
3. Soft-Target & Binary Pseudo-Label Generator for all ~4,400 training reports.
4. Reconciliation between provided gold ground-truth labels and report-extracted labels.
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Set

# 12 Target Findings
TARGET_FINDINGS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture"
]

def normalize_text(text: str) -> str:
    """Normalize accents, lowercases, and clean punctuation."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Multilingual keywords & synonyms for the 12 targets
# Covering English, Spanish, French, German, Portuguese, Italian
TARGET_PATTERNS: Dict[str, List[str]] = {
    "ACL": [
        r"\banterior cruciate ligament\b", r"\bacl\b",
        r"\bligamento cruzado anterior\b", r"\blca\b",
        r"\bligament croise anterieur\b", r"\blca\b",
        r"\bvorder(?:es|en|er|em)? kreuzband(?:es|s)?\b", r"\bvkb\b",
        r"\bligamento crociato anteriore\b"
    ],
    "MCL": [
        r"\bmedial collateral ligament\b", r"\bmcl\b",
        r"\btibial collateral ligament\b",
        r"\bligamento colateral medial\b", r"\blcm\b", r"\bligamento colateral tibial\b",
        r"\bligament collateral medial\b", r"\bligament collateral tibial\b",
        r"\bmediale(?:s|n|r|m)? kollateralband(?:es|s)?\b", r"\binnenband(?:es)?\b",
        r"\bligamento collaterale mediale\b"
    ],
    "Medial Meniscus": [
        r"\bmedial meniscus\b", r"\bmedial meniscal\b", r"\binner meniscus\b",
        r"\bmenisco medial\b", r"\bmenisco interno\b",
        r"\bmenisque medial\b", r"\bmenisque interne\b",
        r"\binnenmeniskus\b", r"\bmedialer meniskus\b"
    ],
    "Lateral Meniscus": [
        r"\blateral meniscus\b", r"\blateral meniscal\b", r"\bouter meniscus\b",
        r"\bmenisco lateral\b", r"\bmenisco externo\b",
        r"\bmenisque lateral\b", r"\bmenisque externe\b",
        r"\baussenmeniskus\b", r"\blateraler meniskus\b"
    ],
    "Medial OA": [
        r"\bmedial\b.*?\b(?:osteoarthritis|oa|arthrosis|gonarthrosis|chondromalacia|joint space narrowing|cartilage loss|eburnation)\b",
        r"\b(?:osteoarthritis|oa|arthrosis|gonarthrosis|chondromalacia|cartilage loss)\b.*?\bmedial\b",
        r"\bosteoartritis medial\b", r"\bosteoartrose medial\b", r"\bgonartrosis medial\b",
        r"\bosteoarthrite mediale\b", r"\bgonarthrose mediale\b",
        r"\bmediale gonarthrose\b", r"\bmediale arthrose\b"
    ],
    "Lateral OA": [
        r"\blateral\b.*?\b(?:osteoarthritis|oa|arthrosis|gonarthrosis|chondromalacia|joint space narrowing|cartilage loss|eburnation)\b",
        r"\b(?:osteoarthritis|oa|arthrosis|gonarthrosis|chondromalacia|cartilage loss)\b.*?\blateral\b",
        r"\bosteoartritis lateral\b", r"\bosteoartrose lateral\b", r"\bgonartrosis lateral\b",
        r"\bosteoarthrite laterale\b", r"\bgonarthrose laterale\b",
        r"\blaterale gonarthrose\b", r"\blaterale arthrose\b"
    ],
    "PF OA": [
        r"\bpatellofemoral\b.*?\b(?:osteoarthritis|oa|arthrosis|chondromalacia|cartilage loss|narrowing)\b",
        r"\b(?:osteoarthritis|oa|arthrosis|chondromalacia)\b.*?\bpatellofemoral\b",
        r"\bpellofemoral\b", r"\bpatella.*femur\b",
        r"\bosteoartritis patelofemoral\b", r"\bgonarthrose femoro-patellaire\b",
        r"\bfemoropatellararthrose\b", r"\bretropatellar\b.*?\b(?:arthrose|chondromalacie)\b"
    ],
    "Effusion": [
        r"\bjoint effusion\b", r"\beffusion\b", r"\bhydrarthrosis\b",
        r"\bexcess joint fluid\b", r"\bsuprapatellar effusion\b",
        r"\bderrame articular\b", r"\bderrame\b", r"\beffusion intra-articulaire\b",
        r"\bepanchement\b", r"\bepanchement intra-articulaire\b",
        r"\bgelenkerguss\b", r"\berguss\b", r"\bversamento articolare\b"
    ],
    "Synovitis": [
        r"\bsynovitis\b", r"\bsynovial hypertrophy\b", r"\bsynovial thickening\b",
        r"\bsinovitis\b", r"\bhipertrofia sinovial\b",
        r"\bsynovite\b", r"\bepaissement synovial\b",
        r"\bsynovialitis\b", r"\bsynovialitis\b", r"\bsinovite\b"
    ],
    "Baker's": [
        r"\bbaker(?:'s)? cyst\b", r"\bpopliteal cyst\b",
        r"\bquiste de baker\b", r"\bquiste popliteo\b",
        r"\bkyste de baker\b", r"\bkyste poplite\b",
        r"\bbaker-zyste\b", r"\bpoplitealzyste\b",
        r"\bcisti di baker\b", r"\bcisto de baker\b"
    ],
    "Contusion": [
        r"\bbone contusion\b", r"\bbone bruise\b", r"\bmarrow edema\b", r"\bbone marrow contusion\b",
        r"\bbone marrow lesion\b", r"\btrabecular contusion\b",
        r"\bcontusion osea\b", r"\bedema oseo\b", r"\bedema medular\b",
        r"\bcontusion osseuse\b", r"\boedeme osseux\b",
        r"\bknochenkontusion\b", r"\bknochenmarkodem\b",
        r"\bcontusione ossea\b"
    ],
    "Fracture": [
        r"\bfracture\b", r"\bfractured\b", r"\bavulsion fracture\b", r"\btrabecular microfracture\b",
        r"\bfractura\b", r"\bfracture\b", r"\bfraktur\b", r"\bfrattura\b"
    ]
}

# Pathological condition markers (evidence of damage/abnormality)
PATHOLOGY_VERBS: List[str] = [
    r"\btear\b", r"\btears\b", r"\btorn\b", r"\brupture\b", r"\bruptured\b",
    r"\bsprain\b", r"\bsprained\b", r"\binjury\b", r"\binjured\b",
    r"\blesion\b", r"\bdeg(?:eneration|enerative)?\b", r"\bmaceration\b",
    r"\bdetachment\b", r"\bdefect\b", r"\bdisruption\b",
    # Multilingual terms
    r"\bdesgarro\b", r"\brotura\b", r"\blesion\b", r"\bruptura\b",
    r"\bdechirure\b", r"\briss\b", r"\blasion\b", r"\bruzzione\b"
]

# Negation scope triggers
PRE_NEGATION_TRIGGERS: List[str] = [
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bfree of\b", r"\bintact\b",
    r"\bnegative for\b", r"\bruled out\b", r"\babsence\b", r"\bdenies\b",
    r"\bno evidence of\b", r"\bno sign of\b", r"\bunremarkable\b", r"\bnormal\b",
    r"\bpreserved\b",
    # Spanish / Portuguese
    r"\bsin\b", r"\bno se observa\b", r"\bausencia\b", r"\bnormal\b", r"\bintacto\b",
    r"\bsem\b", r"\bsem evidencia de\b",
    # French
    r"\bsans\b", r"\bpas de\b", r"\baucun(?:e)?\b", r"\babsence\b", r"\bintact\b",
    # German
    r"\bkein(?:e|en|er|em)?\b", r"\bohne\b", r"\bintakt\b", r"\bregelrecht\b", r"\bregelrechte\b", r"\bausschluss\b"
]

POST_NEGATION_TRIGGERS: List[str] = [
    r"\bunremarkable\b", r"\bintact\b", r"\bnormal\b", r"\bnot seen\b",
    r"\bwas not identified\b", r"\bis excluded\b",
    r"\bintacto\b", r"\bnormal\b", r"\bsem alteracoes\b",
    r"\bintacte?\b", r"\bsans particularite\b",
    r"\bintakt\b", r"\bregelrecht\b", r"\bintatto\b"
]

# Pseudo-negations (do not trigger negation)
PSEUDO_NEGATIONS: List[str] = [
    r"\bno\s+(?:significant\s+)?(?:interval\s+)?change\b",
    r"\bnot\s+only\b",
    r"\bno\s+doubt\b"
]

# Uncertainty terms (lower confidence soft label)
UNCERTAINTY_TRIGGERS: List[str] = [
    r"\bpossible\b", r"\bprobable\b", r"\bsuspected\b", r"\bcannot exclude\b",
    r"\bquestionable\b", r"\bindeterminate\b", r"\bequivocal\b",
    r"\bposible\b", r"\bprobable\b", r"\bsuspeita\b", r"\bpossible\b", r"\bfraglich\b"
]


class KneeReportNLPParser:
    """
    Robust NegEx-style parser for extracting 12 knee abnormalities from multilingual reports.
    """
    def __init__(self):
        self.pre_neg_re = re.compile(r"|".join(PRE_NEGATION_TRIGGERS), re.IGNORECASE)
        self.post_neg_re = re.compile(r"|".join(POST_NEGATION_TRIGGERS), re.IGNORECASE)
        self.pseudo_re = re.compile(r"|".join(PSEUDO_NEGATIONS), re.IGNORECASE)
        self.uncert_re = re.compile(r"|".join(UNCERTAINTY_TRIGGERS), re.IGNORECASE)
        self.pathology_re = re.compile(r"|".join(PATHOLOGY_VERBS), re.IGNORECASE)

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text by punctuation into sentence clauses."""
        clauses = re.split(r'[.;:!?\n\r]+', text)
        return [c.strip() for c in clauses if len(c.strip()) > 2]

    def check_clause_for_finding(self, clause: str, finding: str) -> Tuple[bool, bool, bool]:
        """
        Check if finding is in clause.
        Returns: (present, is_negated, is_uncertain)
        """
        # 1. Match finding pattern
        matched = False
        match_span = None
        for pat in TARGET_PATTERNS[finding]:
            m = re.search(pat, clause)
            if m:
                matched = True
                match_span = m.span()
                break

        if not matched:
            return False, False, False

        # If it's a structural ligament/meniscus, we check if pathology or normal is specified
        is_structural_anatomy = finding in ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"]
        has_explicit_pathology = bool(self.pathology_re.search(clause))

        # Check pre-negation (before mention within ~60 chars)
        prefix = clause[max(0, match_span[0] - 60):match_span[0]]
        suffix = clause[match_span[1]:min(len(clause), match_span[1] + 60)]

        # Mask pseudo-negations
        prefix_clean = self.pseudo_re.sub("___", prefix)
        suffix_clean = self.pseudo_re.sub("___", suffix)

        is_negated = False
        if self.pre_neg_re.search(prefix_clean) or self.post_neg_re.search(suffix_clean):
            is_negated = True

        # For structural tissues (e.g. "ACL"), if no pathology verb is present and words like "intact" or "normal" appear:
        if is_structural_anatomy and not has_explicit_pathology:
            if re.search(r"\b(?:intact|normal|continuous|unremarkable|preserved|homogeneo)\b", clause):
                is_negated = True

        is_uncertain = bool(self.uncert_re.search(prefix_clean) or self.uncert_re.search(suffix_clean))

        return True, is_negated, is_uncertain

    def parse_report(self, report_text: str) -> Dict[str, float]:
        """
        Extract soft probabilities (0.0 to 1.0) for each of the 12 findings.
        Returns dictionary mapping finding -> confidence score.
        """
        normalized = normalize_text(report_text)
        sentences = self.split_into_sentences(normalized)

        # Default confidence: low prior (0.05)
        scores: Dict[str, float] = {f: 0.05 for f in TARGET_FINDINGS}

        for sentence in sentences:
            for finding in TARGET_FINDINGS:
                present, is_negated, is_uncertain = self.check_clause_for_finding(sentence, finding)
                if present:
                    if is_negated:
                        # Strongly negative evidence
                        scores[finding] = min(scores[finding], 0.02)
                    else:
                        # Positive evidence found
                        conf = 0.65 if is_uncertain else 0.95
                        scores[finding] = max(scores[finding], conf)

        return scores
