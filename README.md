# GIVE: Graph-based Interpretable Verification Engine

An implementation of the GIVE framework from the paper ["GIVE: Graph-based Interpretable Verification Engine for Enhancing Large Language Model Reasoning"](https://arxiv.org/pdf/2410.08475).

## Overview

GIVE is a framework designed to enhance the reasoning capabilities of Large Language Models (LLMs) by leveraging knowledge graphs and multi-hop reasoning. The framework aims to provide interpretable and verifiable answers by integrating structured knowledge from knowledge graphs with the reasoning power of LLMs. It combines:

- Knowledge graph integration
- Entity similarity matching
- Progressive knowledge refinement
- Multi-hop reasoning paths

## Features

- 🔍 Knowledge graph loading and querying
- 🤖 OpenAI GPT integration for reasoning
- 🔗 Entity similarity using sentence transformers
- 📊 Graph-based knowledge extraction
- 🔄 Progressive answer refinement

## How It Works

1. **Query Processing**: The framework starts by extracting key entities and relations from user queries using a Large Language Model (LLM). This step ensures that the query is broken down into its fundamental components for further processing.
2. **Entity Grouping**: Similar entities are grouped together using sentence transformers. This step helps in identifying and clustering entities that are semantically similar, which is crucial for effective knowledge integration.
3. **Knowledge Extraction**: The framework combines knowledge from both the Knowledge Graph (KG) and the LLM reasoning. This dual-source approach ensures that the extracted knowledge is both comprehensive and contextually relevant.
4. **Multi-hop Reasoning**: GIVE discovers intermediate connections between entities through multi-hop reasoning. This step allows the framework to uncover deeper insights and relationships that are not immediately apparent.
5. **Answer Generation**: Finally, the framework progressively refines the answers using the extracted knowledge. This iterative refinement process ensures that the final answers are accurate, interpretable, and verifiable.

## Architecture

The framework consists of several key components:

- `GIVE` class: Main interface for the framework
- Knowledge graph integration using NetworkX
- Entity encoding using sentence transformers
- LLM integration (OpenAI GPT) for reasoning
- Progressive knowledge refinement pipeline