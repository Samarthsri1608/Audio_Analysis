# Copy this into your code and run it:
from language_tool_python import LanguageTool
import numpy as np

def debug_grammar_pipeline(transcript):
    tool = LanguageTool('en-US')
    matches = tool.check(transcript)
    
    print(f"Errors found: {len(matches)}")
    
    if len(matches) == 0:
        print("⚠️ NO ERRORS - LanguageTool can't find them!")
        print("→ Solution: Add punctuation before grammar check")
    else:
        print(f"Errors found: {len(matches)}")
        rules = {}
        for m in matches:
            rules[m.ruleId] = rules.get(m.ruleId, 0) + 1
        for rule, count in sorted(rules.items(), key=lambda x: -x[1]):
            print(f"  {rule}: {count}")

# Run it
debug_grammar_pipeline("i think the candidates are good they speak english well")