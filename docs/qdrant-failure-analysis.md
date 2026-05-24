## Qdrant Failure Analysis

**Dataset:** Natural Questions dev split, 100 examples
**Token budget:** 4000

This file lists benchmark failures directly. It is meant for engineering triage, not marketing.

### Qdrant Hybrid Evidence Lost After ContextForge

| Example | Question | Answers | Hybrid Hit / ContextForge Hit | Hybrid Tokens / ContextForge Tokens |
|---------|----------|---------|-------------------------------|-----------------------------------|
| `1474401183034409997` | when does the last episode of adventure time air | TBA | True / False | 3209 / 815 |
| `-8526247506624400769` | who won oscar for best director this month | Guillermo del Toro | True / False | 12759 / 359 |
| `-4135209844918483842` | who carried the us flag in the 2014 olympics | Julie Chu, Todd Lodwick | True / False | 1601 / 763 |
| `5543510584551366341` | when did the celebrities enter the big brother house | 2017, February 7, 2018 | True / False | 4326 / 3877 |
| `-8022345911863395279` | the father son and holy spirit in latin | in nomine Patris et Filii et Spiritus Sancti, Patris et F... | True / False | 670 / 662 |
| `1065612251914840415` | who is the girl in the stone sour video say you'll haunt me | Joanna Moskawa | True / False | 704 / 615 |
| `6009212502620981150` | what's the biggest nfl stadium in the united states | MetLife Stadium, Michigan Stadium | True / False | 11521 / 402 |
| `-1003552412210538439` | who has the most all ireland hurling medals | Henry Shefflin | True / False | 20577 / 754 |
| `-8311261765349453370` | when did the smoking ban in public places start | 1995, August 2, 1990 | True / False | 7111 / 1296 |

### Vector Evidence Lost After ContextForge

| Example | Question | Answers | Vector Hit / ContextForge Hit | Vector Tokens / ContextForge Tokens |
|---------|----------|---------|-------------------------------|-----------------------------------|
| `-8526247506624400769` | who won oscar for best director this month | Guillermo del Toro | True / False | 12758 / 335 |
| `5543510584551366341` | when did the celebrities enter the big brother house | 2017, February 7, 2018 | True / False | 4326 / 3865 |
| `-8022345911863395279` | the father son and holy spirit in latin | in nomine Patris et Filii et Spiritus Sancti, Patris et F... | True / False | 670 / 632 |
| `6009212502620981150` | what's the biggest nfl stadium in the united states | MetLife Stadium, Michigan Stadium | True / False | 11521 / 384 |
| `-1003552412210538439` | who has the most all ireland hurling medals | Henry Shefflin | True / False | 20577 / 724 |
| `-4366283268910846199` | where was the ark of the covenant built | the foot of biblical Mount Sinai, at the foot of biblical... | True / False | 631 / 721 |
| `-8311261765349453370` | when did the smoking ban in public places start | 1995, August 2, 1990 | True / False | 1106 / 1266 |

### ContextForge Token Budget Violations

| Example | Strategy | Question | Tokens | Utilization |
|---------|----------|----------|--------|-------------|
| none | none | none | none | none |
