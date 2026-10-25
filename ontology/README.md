# IABM Ontology-aligned contextualization schema

This folder contains the lightweight ontology-aligned contextualization schema used to support the industrial asset behavioral monitoring workflow.

## Contents

- `iabm.ttl`  
  Core schema describing the main classes and relations used to contextualize observations, operational states, behavioral sequences, operating modes, working modes, and abnormality evidence.

- `examples/wheel_washer.ttl`  
  Minimal instance-level example based on the wheel-washing industrial asset used in the paper.

- `queries/example_queries.rq`  
  Example SPARQL queries showing how observations, states, sequences, modes, and abnormality evidence can be retrieved from the schema and example instance data.

## Scope

The current semantic layer should be understood as a lightweight ontology-aligned contextualization scheme rather than as a complete ontology-driven reasoning system. Its purpose is to make the contextualization explicit, inspectable, and reusable.

## Suggested usage

A reader may inspect the schema first (`iabm.ttl`), then the instantiated example (`examples/wheel_washer_example.ttl`), and finally the query examples (`queries/example_queries.rq`) to understand how the semantic layer relates to the outputs of `Model_A`, `Model_B`, and `Model_C`.
