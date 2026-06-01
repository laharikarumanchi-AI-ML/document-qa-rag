## What changed
<!-- 1–2 lines summarizing the change -->

## Why
<!-- The problem / motivation. Link to spec section if relevant. -->

## How I verified
<!-- e.g. "pytest passes locally", "ran ragqa ask on the LLaMA paper and got a sensible answer with citations". -->

## Notes for future me
<!-- Anything weird I want to remember: shortcuts taken, follow-ups, decisions. -->

## Pre-merge checklist
- [ ] All tests pass locally (`pytest -v`)
- [ ] If retrieval params changed, I re-ran the smoke set and noted the precision@k
- [ ] If the prompt changed, I tested it against an "answer not in docs" case to verify it abstains
- [ ] No secrets or API keys committed
