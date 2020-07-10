# Automation Planning

These are the main components, which we need for a full automation:
1. “localisation”, i.e., the attribution to find relevant weights, as well maybe the repair layer selection
2. test input execution and path constraint extraction
3. repair constraint collection/creation and constraint solving
4. expert generation and repaired/combined network synthesis
5. analysis/statistics after repair

Module (2) involves SPF.
Module (3) handles combining the results of SPF and calling z3
Module (4) takes the generated z3 model and synthesizes a new repaired network 
