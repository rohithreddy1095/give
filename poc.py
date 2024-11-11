import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Set, Tuple
import networkx as nx
import json
from typing import List, Tuple
from pydantic import BaseModel
from openai import OpenAI

model_name = "gpt-4o"


class QueryExtraction(BaseModel):
    Entities: List[str]
    Relations: List[str]

class GIVE:
    def __init__(self, llm_model: str = "gpt-3.5-turbo", kg_path: str = None):
        """
        Initialize GIVE framework
        Args:
            llm_model: Name of the LLM model to use
            kg_path: Path to knowledge graph data
        """
        self.llm = self._initialize_llm(llm_model)
        self.kg = self._load_knowledge_graph(kg_path)
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    def _initialize_llm(self, model_name: str):
        """Initialize LLM interface"""
        import openai
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        if "gpt" in model_name.lower():
            try:
                client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
                # Test the client with a simple request
                client.models.list()
                return client
            except openai.AuthenticationError:
                raise ValueError("Invalid OpenAI API key")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _load_knowledge_graph(self, kg_path: str) -> nx.Graph:
        """Load knowledge graph from file"""
        if kg_path is None:
            return nx.Graph()
            
        import rdflib
        graph = rdflib.Graph()
        file_format = kg_path.split('.')[-1]
        
        try:
            if file_format == 'ttl':
                graph.parse(kg_path, format='turtle') 
            elif file_format in ['jsonld', 'json']:
                graph.parse(kg_path, format='json-ld')
            elif file_format == 'owl':
                graph.parse(kg_path, format='xml')
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
            # Convert RDF graph to NetworkX graph
            nx_graph = nx.Graph()
            for s, p, o in graph:
                nx_graph.add_edge(str(s), str(o), relation=str(p))
                
            return nx_graph
            
        except Exception as e:
            print(f"Error loading knowledge graph: {str(e)}")
            return nx.Graph()

    def extract_query_info(self, query: str) -> Tuple[List[str], List[str]]:
        prompt = """Extract key entities and relations from this query in JSON format:
        {
            "Entities": ["entity1", "entity2"],
            "Relations": ["relation1", "relation2"]
        }"""
        
        client = OpenAI()
        completion = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt + "\n\nQuery: " + query}
            ],
            response_format=QueryExtraction,
        )
        
        response = completion.choices[0].message.parsed
        entities = response.Entities
        relations = response.Relations
        
        return entities, relations

    def _get_similar_entities(self, encoded_query: torch.Tensor, top_p: int = 5) -> List[str]:
        """Find similar entities in knowledge graph using cosine similarity"""
        similar_entities = []
        
        # Get all entities from the knowledge graph
        entities = list(self.kg.nodes())
        if not entities:
            return similar_entities
            
        # Encode all entities
        with torch.no_grad():
            entity_encodings = torch.cat([self._encode_text(e) for e in entities])
            
        # Calculate cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            encoded_query, entity_encodings
        )
        
        # Get top-p similar entities
        top_indices = torch.argsort(similarities, descending=True)[:top_p]
        similar_entities = [entities[i] for i in top_indices]
        
        return similar_entities

    def construct_entity_groups(self, entities: List[str], top_p: int = 5) -> Dict[str, Set[str]]:
        """
        Construct groups of similar entities
        Args:
            entities: List of query entities
            top_p: Number of similar entities to retrieve
        Returns:
            Dictionary mapping query entities to sets of similar entities
        """
        entity_groups = {}
        for entity in entities:
            # Encode query entity
            encoded = self._encode_text(entity)
            
            # Find similar entities in KG
            similar_entities = self._get_similar_entities(encoded, top_p)
            entity_groups[entity] = set(similar_entities)
            
        return entity_groups

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text using pretrained encoder"""
        tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            output = self.encoder(**tokens)
        return output.last_hidden_state.mean(dim=1)

    def induce_inner_group_connections(self, entity_groups: Dict[str, Set[str]]) -> List[Tuple]:
        """
        Induce connections within entity groups
        Args:
            entity_groups: Dictionary of entity groups
        Returns:
            List of (entity1, relation, entity2) triplets
        """
        connections = []
        for entity, group in entity_groups.items():
            for similar_entity in group:
                relation = self._get_llm_relation(entity, similar_entity)
                if relation:
                    connections.append((entity, relation, similar_entity))
        return connections

    def discover_intermediate_groups(self, 
                                  source_group: Set[str], 
                                  target_group: Set[str]) -> Set[str]:
        """
        Discover intermediate node groups for multi-hop reasoning
        Args:
            source_group: Source entity group
            target_group: Target entity group
        Returns:
            Set of intermediate entities
        """
        # Find length-2 paths between groups
        paths = self._find_paths(source_group, target_group, max_length=2)
        
        # Use LLM to select most relevant intermediate nodes
        intermediate_nodes = self._select_intermediate_nodes(paths)
        
        # Construct intermediate group
        return self.construct_entity_groups([intermediate_nodes])[intermediate_nodes]

    def _extract_kg_knowledge(self, entity_groups: Dict[str, Set[str]]) -> List[Tuple]:
        """Extract knowledge from knowledge graph"""
        knowledge = []
        
        # Flatten all entities from groups
        all_entities = {e for group in entity_groups.values() for e in group}
        
        # Get all edges between entities in groups
        for e1 in all_entities:
            for e2 in all_entities:
                if self.kg.has_edge(e1, e2):
                    edge_data = self.kg.get_edge_data(e1, e2)
                    knowledge.append((e1, edge_data['relation'], e2))
        
        return knowledge

    def _extrapolate_llm_knowledge(self, entity_groups: Dict[str, Set[str]], 
                                 potential_relations: Set[str]) -> List[Tuple]:
        """Use LLM to extrapolate additional knowledge"""
        knowledge = []
        
        for entity, similar_entities in entity_groups.items():
            for similar_entity in similar_entities:
                prompt = f"What is the relationship between {entity} and {similar_entity}?"
                response = self.llm.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                relation = response.choices[0].message.content.strip()
                if relation in potential_relations:
                    knowledge.append((entity, relation, similar_entity))
        
        return knowledge

    def extrapolate_knowledge(self, 
                            entity_groups: Dict[str, Set[str]], 
                            potential_relations: Set[str]) -> List[Tuple]:
        """
        Extrapolate knowledge using LLM and KG structure
        Args:
            entity_groups: Dictionary of entity groups
            potential_relations: Set of potential relations
        Returns:
            List of (subject, relation, object) triplets
        """
        knowledge = []
        
        # Extract knowledge from KG
        kg_knowledge = self._extract_kg_knowledge(entity_groups)
        knowledge.extend(kg_knowledge)
        
        # Extrapolate using LLM
        extrapolated = self._extrapolate_llm_knowledge(entity_groups, potential_relations)
        knowledge.extend(extrapolated)
        
        return knowledge

    def _is_affirmative(self, knowledge_tuple: Tuple) -> bool:
        """Check if knowledge tuple represents affirmative knowledge"""
        # Simple implementation - could be enhanced with more sophisticated logic
        return True  # For now, treat all knowledge as affirmative
    
    def _generate_with_knowledge(self, query: str, knowledge: List[Tuple]) -> str:
        """Generate answer using provided knowledge"""
        # Format knowledge into readable text
        knowledge_text = "\n".join([f"{s} {r} {o}" for s, r, o in knowledge])
        
        prompt = f"""Given this knowledge:
{knowledge_text}

Answer this question: {query}"""
        
        response = self.llm.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.strip()

    def generate_answer(self, query: str, knowledge: List[Tuple]) -> str:
        """
        Generate final answer using progressive refinement
        Args:
            query: Input query
            knowledge: List of knowledge triplets
        Returns:
            Generated answer
        """
        # Split knowledge into affirmative and counter-factual
        affirmative = [k for k in knowledge if self._is_affirmative(k)]
        counterfactual = [k for k in knowledge if not self._is_affirmative(k)]
        
        # Progressive refinement
        initial_answer = self._generate_with_knowledge(query, affirmative)
        refined_answer = self._generate_with_knowledge(query, affirmative + counterfactual)
        final_answer = self._generate_with_knowledge(query, knowledge)
        
        return final_answer


if __name__ == "__main__":
    # Initialize GIVE
    give = GIVE(llm_model="gpt-4o", kg_path="data/CustomerInfo.owl")

    # Process a query
    query = "What is the relationship between John Doe and Jane Smith?"

    # Extract query information
    entities, relations = give.extract_query_info(query)

    # Construct entity groups
    entity_groups = give.construct_entity_groups(entities)

    # Generate knowledge
    knowledge = give.extrapolate_knowledge(entity_groups, set(relations))

    # Generate answer
    answer = give.generate_answer(query, knowledge)

    print(answer)
