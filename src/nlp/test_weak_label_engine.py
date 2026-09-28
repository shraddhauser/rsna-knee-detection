"""
Test cases for Multilingual NegEx Weak Label Extractor
"""
from weak_label_engine import KneeReportNLPParser

def run_tests():
    parser = KneeReportNLPParser()
    
    test_cases = [
        # 1. Obvious English positive ACL tear
        ("Complete tear of the anterior cruciate ligament with joint effusion.", 
         {"ACL": 0.95, "Effusion": 0.95}),
         
        # 2. Negated finding (MedCLIP & Das et al. failure mode avoided!)
        ("No evidence of fracture or bone contusion. The anterior cruciate ligament is completely intact.",
         {"Fracture": 0.02, "Contusion": 0.02, "ACL": 0.02}),
         
        # 3. Pseudo-negation ("No significant change in meniscal tear")
        ("No significant interval change in previously noted complex medial meniscus tear.",
         {"Medial Meniscus": 0.95}),
         
        # 4. Uncertainty
        ("Possible mild lateral compartment osteoarthritis with questionable joint space narrowing.",
         {"Lateral OA": 0.65}),
         
        # 5. Multilingual - Spanish
        ("Sin derrame articular ni lesiones osteocondrales. Desgarro del menisco interno.",
         {"Effusion": 0.02, "Medial Meniscus": 0.95}),
         
        # 6. Multilingual - German
        ("Kein Erguss. Regelrechte Darstellung des vorderen Kreuzbandes. Baker-Zyste medial.",
         {"Effusion": 0.02, "ACL": 0.02, "Baker's": 0.95}),
         
        # 7. Multilingual - French
        ("Absence d'epanchement intra-articulaire. Rupture du ligament croise anterieur.",
         {"Effusion": 0.02, "ACL": 0.95})
    ]
    
    passed = 0
    for idx, (text, expected_checks) in enumerate(test_cases, 1):
        scores = parser.parse_report(text)
        print(f"\n--- Test Case {idx}: {text} ---")
        case_passed = True
        for target, expected_val in expected_checks.items():
            actual_val = scores.get(target, 0.0)
            diff = abs(actual_val - expected_val)
            print(f"Target '{target}': Expected ~{expected_val}, Got {actual_val}")
            if diff > 0.15:
                print(f"FAILED on {target}")
                case_passed = False
        if case_passed:
            passed += 1
            print("Status: PASSED")
        else:
            print("Status: FAILED")
            
    print(f"\nSummary: {passed}/{len(test_cases)} tests passed.")

if __name__ == "__main__":
    run_tests()
