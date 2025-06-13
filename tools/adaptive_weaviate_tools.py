# tools/adaptive_weaviate_tools.py - PURE AI ADAPTIVE PIPELINE

import os
import json
import logging
import weaviate
import weaviate.classes.query as wq
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from crewai.tools import tool
import boto3

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Weaviate client (reuse your existing singleton)
try:
    from tools.weaviate_tools import WeaviateClientSingleton
except ImportError:
    # Fallback if import fails
    class WeaviateClientSingleton:
        @classmethod
        def get_instance(cls):
            return None

class AdaptiveSearchEngine:
    """Pure AI-driven adaptive search engine for Weaviate"""
    
    def __init__(self):
        # Initialize Claude for AI decision making
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.model_id = os.getenv('BEDROCK_MODEL_ID_CLAUDE', 'anthropic.claude-3-5-haiku-20241022-v1:0')
    
    def ai_query_triage(self, query: str) -> Dict[str, Any]:
        """AI-driven query complexity assessment"""
        
        prompt = f"""
        Analyze this user query and determine if it's SIMPLE or COMPLEX:
        
        Query: "{query}"
        
        SIMPLE queries:
        - Contain specific identifiers (names, IDs, exact values)
        - Direct lookup requests
        - Clear, single-purpose questions
        
        COMPLEX queries:
        - Conceptual or analytical requests
        - Contain words like "analyze", "compare", "trend", "lifecycle"
        - Vague or multi-step questions
        - Require business context understanding
        
        Respond with JSON only:
        {{
            "complexity": "SIMPLE" or "COMPLEX",
            "reasoning": "brief explanation",
            "requires_context_enrichment": true/false,
            "key_entities": ["list", "of", "key", "entities", "mentioned"]
        }}
        """
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 300,
                    "temperature": 0.1
                })
            )
            
            result = json.loads(response['body'].read())
            content = result['content'][0]['text']
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback
                return {
                    "complexity": "COMPLEX",
                    "reasoning": "AI parsing failed, defaulting to complex",
                    "requires_context_enrichment": True,
                    "key_entities": []
                }
                
        except Exception as e:
            logger.warning(f"AI triage failed: {e}")
            return {
                "complexity": "COMPLEX",
                "reasoning": "AI service unavailable, defaulting to complex",
                "requires_context_enrichment": True,
                "key_entities": []
            }
    
    def ai_context_enrichment(self, query: str, business_context_results: List[Dict]) -> str:
        """AI-driven query enrichment using BusinessContext"""
        
        if not business_context_results:
            return query
        
        # Prepare context for AI - SCHEMA-ALIGNED
        context_info = ""
        for ctx in business_context_results[:3]:  # Top 3 contexts
            context_info += f"""
            Business Term: {ctx.get('term', 'Unknown')}
            Definition: {ctx.get('definition', 'No definition')}
            Strategic Context: {ctx.get('context', 'No context')}
            SQL Examples: {ctx.get('examples', 'No examples')}
            ---
            """
        
        prompt = f"""
        Original Query: "{query}"
        
        Relevant Business Context:
        {context_info}
        
        Create an enriched search query that:
        1. Incorporates relevant keywords from the business context
        2. Maintains the original intent
        3. Adds specific table/column terminology when available
        4. Makes the query more precise for database schema discovery
        
        Return only the enriched query string, no explanation.
        """
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 200,
                    "temperature": 0.1
                })
            )
            
            result = json.loads(response['body'].read())
            enriched_query = result['content'][0]['text'].strip()
            
            # Clean up the response (remove quotes if present)
            enriched_query = enriched_query.strip('"\'')
            
            logger.info(f"Query enriched: '{query}' -> '{enriched_query}'")
            return enriched_query
            
        except Exception as e:
            logger.warning(f"AI enrichment failed: {e}")
            return query  # Fallback to original
    
    def ai_column_relevance_scorer(self, query: str, columns: List[Dict]) -> List[Dict]:
        """AI-driven column relevance scoring"""
        
        if not columns:
            return columns
        
        # Prepare column info for AI
        column_info = ""
        for i, col in enumerate(columns[:20]):  # Limit to avoid token limits
            column_info += f"{i}: {col.get('column_name', 'Unknown')} ({col.get('data_type', 'Unknown')}) - {col.get('description', 'No description')}\n"
        
        prompt = f"""
        Query: "{query}"
        
        Available Columns:
        {column_info}
        
        Rate each column's relevance to the query on a scale of 0-10.
        Consider:
        - Direct semantic match to query intent
        - Data type appropriateness
        - Column name relevance
        - Description alignment
        
        Respond with JSON only:
        {{
            "0": 8.5,
            "1": 3.2,
            "2": 9.1,
            ...
        }}
        """
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0.1
                })
            )
            
            result = json.loads(response['body'].read())
            content = result['content'][0]['text']
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                
                # Apply scores to columns
                for i, col in enumerate(columns):
                    score = scores.get(str(i), 5.0)  # Default to 5.0 if not scored
                    col['ai_relevance_score'] = float(score)
                
                # Sort by AI relevance score
                columns.sort(key=lambda x: x.get('ai_relevance_score', 0), reverse=True)
                
                return columns
                
        except Exception as e:
            logger.warning(f"AI column scoring failed: {e}")
        
        return columns  # Return original if AI fails

# Initialize the adaptive search engine
adaptive_engine = AdaptiveSearchEngine()

@tool("Adaptive Business Context Analyzer")
def adaptive_business_context_analyzer(query: str) -> str:
    """
    STAGE 1: Adaptive Context & Keyword Enrichment
    
    Pure AI-driven triage and enrichment:
    1. AI determines if query is SIMPLE or COMPLEX
    2. For COMPLEX queries: searches BusinessContext and enriches query
    3. For SIMPLE queries: passes through directly
    
    This is the adaptive entry point to the pipeline.
    """
    weaviate_client = WeaviateClientSingleton.get_instance()
    logger.info(f"🧠 STAGE 1: Adaptive Context Analysis for: '{query}'")
    
    # STEP 1: AI-driven query triage
    triage_result = adaptive_engine.ai_query_triage(query)
    logger.info(f"📊 AI Triage: {triage_result['complexity']} - {triage_result['reasoning']}")
    
    result = {
        "original_query": query,
        "triage_result": triage_result,
        "final_search_query": query,
        "contextual_warnings": [],
        "enrichment_applied": False
    }
    
    # STEP 2: Conditional BusinessContext enrichment
    if triage_result["requires_context_enrichment"] and weaviate_client:
        try:
            logger.info("🔍 Searching BusinessContext for enrichment...")
            business_collection = weaviate_client.collections.get("BusinessContext")
            
            # Search BusinessContext with original query - SCHEMA-ALIGNED
            import weaviate.classes.query as wq
            
            context_response = business_collection.query.near_text(
                query=query,
                limit=3,
                return_properties=[
                    # ✅ EXACT MATCH to your BusinessContext schema
                    "term",        # ✅ Exists in your schema
                    "definition",  # ✅ Exists (vectorized)
                    "context",     # ✅ Exists (vectorized)
                    "examples"     # ✅ Exists (not vectorized, SQL examples)
                ],
                return_metadata=wq.MetadataQuery(distance=True)
            )
            
            if context_response.objects:
                # Extract context results - SCHEMA-ALIGNED
                context_results = []
                for obj in context_response.objects:
                    if obj.metadata.distance < 0.4:  # Only high-relevance contexts
                        context_results.append({
                            "term": obj.properties.get('term', ''),
                            "definition": obj.properties.get('definition', ''),
                            "context": obj.properties.get('context', ''),
                            "examples": obj.properties.get('examples', ''),  # SQL examples from your schema
                            "relevance": 1 - obj.metadata.distance
                        })
                
                if context_results:
                    # AI-driven query enrichment
                    enriched_query = adaptive_engine.ai_context_enrichment(query, context_results)
                    
                    result.update({
                        "final_search_query": enriched_query,
                        "business_contexts_found": context_results,
                        "enrichment_applied": True
                    })
                    
                    # Extract contextual warnings from context
                    for ctx in context_results:
                        context_text = ctx.get('context', '')
                        if any(warning_word in context_text.lower() for warning_word in ['warning', 'issue', 'caution', 'problem']):
                            result["contextual_warnings"].append(f"Context warning from {ctx.get('term', 'Unknown')}: Check context for data quality considerations")
                    
                    logger.info(f"✅ Query enriched and warnings extracted")
                else:
                    logger.info("ℹ️ No relevant BusinessContext found")
            else:
                logger.info("ℹ️ No BusinessContext results")
                
        except Exception as e:
            logger.warning(f"BusinessContext enrichment failed: {e}")
    
    elif triage_result["complexity"] == "SIMPLE":
        logger.info("⚡ SIMPLE query - skipping BusinessContext enrichment for efficiency")
    
    logger.info(f"🎯 Final search query: '{result['final_search_query']}'")
    return json.dumps(result, indent=2)

@tool("Adaptive Schema Discovery Engine")
def adaptive_schema_discovery_engine(context_analysis_json: str) -> str:
    """
    STAGE 2: Multi-Layered Schema Discovery with Precision Filtering
    
    Uses the adaptive context analysis to perform:
    1. Core dataset discovery (DatasetMetadata)
    2. Precision column discovery with mandatory filtering
    3. Conditional relationship discovery
    
    This implements the 4-step search strategy with AI-driven precision.
    """
    weaviate_client = WeaviateClientSingleton.get_instance()
    
    # Parse context analysis
    try:
        context_data = json.loads(context_analysis_json)
    except:
        logger.error("Failed to parse context analysis JSON")
        return json.dumps({"error": "Invalid context analysis input"})
    
    final_search_query = context_data.get("final_search_query", context_data.get("original_query", ""))
    contextual_warnings = context_data.get("contextual_warnings", [])
    
    logger.info(f"🔍 STAGE 2: Adaptive Schema Discovery for: '{final_search_query}'")
    
    if not weaviate_client:
        logger.warning("🔄 Weaviate not available, using adaptive fallback")
        return _get_adaptive_fallback_schema(final_search_query, context_data)
    
    results = {
        "search_query": final_search_query,
        "context_analysis": context_data,
        "datasets": [],
        "columns_by_dataset": {},
        "relationships": [],
        "contextual_warnings": contextual_warnings,
        "optimization_metrics": {
            "queries_executed": 0,
            "precision_filters_applied": 0,
            "results_filtered": 0
        }
    }
    
    try:
        # STEP 2A: Core Dataset Discovery
        logger.info("1️⃣ Discovering core datasets...")
        datasets_found = _discover_core_datasets(weaviate_client, final_search_query, results["optimization_metrics"])
        results["datasets"] = datasets_found
        
        if not datasets_found:
            logger.warning("No datasets found")
            return json.dumps(results, indent=2)
        
        # STEP 2B: Precision Column Discovery (THE EFFICIENCY ENGINE)
        logger.info("2️⃣ Precision column discovery with mandatory filtering...")
        columns_found = _discover_columns_with_precision_filtering(
            weaviate_client, final_search_query, datasets_found, results["optimization_metrics"]
        )
        
        # CRITICAL: Check if column discovery actually succeeded
        if not columns_found or all(len(cols) == 0 for cols in columns_found.values()):
            error_msg = f"CRITICAL ERROR: Column discovery failed for all tables. This indicates a schema mismatch between the tool code and your Weaviate database. Check that property names in the tool match your actual Weaviate schema."
            logger.error(error_msg)
            results["error"] = error_msg
            results["columns_by_dataset"] = {}
            return json.dumps(results, indent=2)
        
        results["columns_by_dataset"] = columns_found
        
        # STEP 2C: Conditional Relationship Discovery
        if len(datasets_found) > 1:
            logger.info("3️⃣ Discovering join relationships...")
            relationships_found = _discover_join_relationships(
                weaviate_client, datasets_found, results["optimization_metrics"]
            )
            results["relationships"] = relationships_found
        
        logger.info(f"✅ Schema discovery complete - {results['optimization_metrics']['queries_executed']} queries executed")
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Schema discovery error: {e}", exc_info=True)
        return _get_adaptive_fallback_schema(final_search_query, context_data)

def _discover_core_datasets(weaviate_client, search_query: str, metrics: Dict) -> List[Dict]:
    """Step 2A: Discover core datasets using hybrid search"""
    
    try:
        dataset_collection = weaviate_client.collections.get("DatasetMetadata")
        
        # Import MetadataQuery for proper v4 client usage
        import weaviate.classes.query as wq
        
        # Use hybrid search as specified (alpha=0.75) - SCHEMA-ALIGNED
        response = dataset_collection.query.hybrid(
            query=search_query,
            alpha=0.75,  # Weight semantic search higher than keyword search
            limit=5,
            return_properties=[
                # ✅ EXACT MATCH to your DatasetMetadata schema (optimized version)
                "tableName",                      # ✅ Exists
                "athenaTableName",               # ✅ Exists
                "description",                   # ✅ Exists (vectorized)
                "businessPurpose",               # ✅ Exists (vectorized)
                "tags",                          # ✅ Exists (TEXT_ARRAY, vectorized)
                "recordCount",                   # ✅ Exists
                "columnSemanticsConcatenated",   # ✅ Exists (vectorized)
                "dataOwner",                     # ✅ Exists
                "sourceSystem"                   # ✅ Exists
            ],
            return_metadata=wq.MetadataQuery(score=True, distance=True)
        )
        
        metrics["queries_executed"] += 1
        
        datasets = []
        for obj in response.objects:
            props = obj.properties
            table_name = props.get('tableName', '')
            athena_table_name = props.get('athenaTableName', '')
            
            if table_name and athena_table_name:
                # Ensure proper schema prefix
                if "." not in athena_table_name:
                    athena_table_name = f"amspoc3test.{athena_table_name}"
                
                # Get relevance score
                score = getattr(obj.metadata, 'score', 0.5)
                
                datasets.append({
                    "table_name": table_name,
                    "athena_table_name": athena_table_name,
                    "description": props.get('description', ''),
                    "business_purpose": props.get('businessPurpose', ''),
                    "tags": props.get('tags', []),
                    "record_count": props.get('recordCount', 0),
                    "relevance_score": score
                })
                
                logger.info(f"   ✅ {table_name} -> {athena_table_name} (score: {score:.3f})")
        
        # Sort by relevance score
        datasets.sort(key=lambda x: x['relevance_score'], reverse=True)
        return datasets
        
    except Exception as e:
        logger.error(f"Dataset discovery failed: {e}")
        return []

# In your _discover_columns_with_precision_filtering function
# Fix the filter to use the actual parentAthenaTableName values

def _discover_columns_with_precision_filtering(weaviate_client, search_query: str, datasets: List[Dict], metrics: Dict) -> Dict[str, List[Dict]]:
    """Step 2B: THE EFFICIENCY ENGINE - Precision column discovery with mandatory filtering"""
    
    column_collection = weaviate_client.collections.get("Column")
    columns_by_dataset = {}
    
    from weaviate.classes.query import Filter
    import weaviate.classes.query as wq
    
    # For each dataset, perform FILTERED column search
    for dataset in datasets:
        table_name = dataset["table_name"]
        athena_table_name = dataset["athena_table_name"]
        
        logger.info(f"   🎯 Precision search for {table_name} columns...")
        
        try:
            # Use table_name (not athena_table_name) for the filter
            precision_filter = Filter.by_property("parentAthenaTableName").equal(table_name)
            
            logger.info(f"   🔍 Using filter: parentAthenaTableName = '{table_name}'")
            
            # Perform filtered semantic search
            response = column_collection.query.near_text(
                query=search_query,
                limit=20,
                filters=precision_filter,
                return_properties=[
                    "columnName", 
                    "parentAthenaTableName",
                    "athenaDataType",
                    "pandasDataType", 
                    "description", 
                    "businessName",
                    "semanticType", 
                    "isPrimaryKey", 
                    "foreignKeyInfo",
                    "sampleValues",
                    "sqlUsagePattern",
                    "usageHints",
                    "nullCount",
                    "dataClassification",
                    "commonFilters",
                    "aggregationPatterns",
                    "parentDatasetContext"
                ],
                return_metadata=wq.MetadataQuery(distance=True)
            )
            
            metrics["queries_executed"] += 1
            metrics["precision_filters_applied"] += 1
            
            columns = []
            for col_obj in response.objects:
                col_props = col_obj.properties
                col_name = col_props.get('columnName', '')
                
                if col_name:
                    distance = col_obj.metadata.distance
                    relevance = 1 - distance
                    
                    # Get the primary data type
                    athena_data_type = col_props.get('athenaDataType', '')
                    pandas_data_type = col_props.get('pandasDataType', '')
                    primary_data_type = athena_data_type or pandas_data_type or 'UNKNOWN'
                    
                    # Handle foreign key info correctly
                    foreign_key_info = col_props.get('foreignKeyInfo', '')
                    is_foreign_key = bool(foreign_key_info and foreign_key_info.strip())
                    
                    column_data = {
                        "column_name": col_name,
                        "data_type": primary_data_type,
                        "athena_data_type": athena_data_type,
                        "pandas_data_type": pandas_data_type,
                        "description": col_props.get('description', ''),
                        "business_name": col_props.get('businessName', ''),
                        "semantic_type": col_props.get('semanticType', ''),
                        "is_primary_key": col_props.get('isPrimaryKey', False),
                        "is_foreign_key": is_foreign_key,
                        "foreign_key_info": foreign_key_info,
                        "sample_values": col_props.get('sampleValues', []),
                        "sql_usage_pattern": col_props.get('sqlUsagePattern', ''),
                        "usage_hints": col_props.get('usageHints', []),
                        "null_count": col_props.get('nullCount', 0),
                        "data_classification": col_props.get('dataClassification', ''),
                        "common_filters": col_props.get('commonFilters', []),
                        "aggregation_patterns": col_props.get('aggregationPatterns', []),
                        "parent_dataset_context": col_props.get('parentDatasetContext', ''),
                        "relevance": relevance
                    }
                    
                    columns.append(column_data)
            
            if columns:
                # AI-driven relevance scoring (sorts by highest score first)
                columns = adaptive_engine.ai_column_relevance_scorer(search_query, columns)
                
                # ✅ TOP 5 COLUMNS ONLY - UNCOMMENT TO SEND ALL COLUMNS
                top_columns = columns[:5]  # Take only top 5 highest scoring columns
                columns_by_dataset[athena_table_name] = top_columns
                
                # 🚫 COMMENTED OUT: SEND ALL COLUMNS
                # Uncomment the line below and comment out the two lines above to send all columns
                # columns_by_dataset[athena_table_name] = columns
                
                logger.info(f"      📊 Found {len(columns)} columns, sending top {len(top_columns)} to SQL agent")
                logger.info(f"      🎯 Top columns selected by AI score:")
                for i, col in enumerate(top_columns):
                    ai_score = col.get('ai_relevance_score', 0)
                    rank_emoji = ["🥇", "🥈", "🥉", "🏅", "🏅"][i] if i < 5 else "📍"
                    logger.info(f"         {rank_emoji} {col['column_name']} (AI: {ai_score:.1f})")
                
                # Log filtered out columns count
                if len(columns) > 5:
                    filtered_out = len(columns) - 5
                    logger.info(f"      🗑️  Filtered out {filtered_out} lower-scoring columns")
                    
            else:
                logger.warning(f"      ❌ No columns found for {table_name}")
                logger.info(f"         Filter used: parentAthenaTableName = '{table_name}'")
                
        except Exception as e:
            logger.warning(f"Column discovery failed for {table_name}: {e}")
            import traceback
            logger.debug(f"Full error traceback: {traceback.format_exc()}")
    
    return columns_by_dataset
def _discover_join_relationships(weaviate_client, datasets: List[Dict], metrics: Dict) -> List[Dict]:
    """Step 2C: Discover join relationships using compound filtering"""
    
    if len(datasets) < 2:
        return []
    
    try:
        relationship_collection = weaviate_client.collections.get("DataRelationship")
        relationships = []
        
        # Check all pairs of tables
        for i, table1 in enumerate(datasets):
            for table2 in datasets[i+1:]:
                
                table1_name = table1["table_name"]
                table2_name = table2["table_name"]
                
                logger.info(f"   🔗 Checking relationship: {table1_name} <-> {table2_name}")
                
                # Compound filter to find relationship in either direction
                from weaviate.classes.query import Filter
                
                relationship_filter = Filter.any_of([
                    # Direction 1: table1 -> table2
                    Filter.all_of([
                        Filter.by_property("fromTableName").equal(table1_name),
                        Filter.by_property("toTableName").equal(table2_name)
                    ]),
                    # Direction 2: table2 -> table1
                    Filter.all_of([
                        Filter.by_property("fromTableName").equal(table2_name),
                        Filter.by_property("toTableName").equal(table1_name)
                    ])
                ])
                
                response = relationship_collection.query.fetch_objects(
                    filters=relationship_filter,  # v4 client uses 'filters' not 'where'
                    limit=1,
                    return_properties=[
                        # ✅ EXACT MATCH to your DataRelationship schema
                        "fromTableName",           # ✅ Exists
                        "fromColumn",              # ✅ Exists
                        "toTableName",             # ✅ Exists
                        "toColumn",                # ✅ Exists
                        "relationshipType",        # ✅ Exists (vectorized)
                        "cardinality",             # ✅ Exists
                        "suggestedJoinType",       # ✅ Exists
                        "businessMeaning"          # ✅ Exists (vectorized)
                    ]
                )
                
                metrics["queries_executed"] += 1
                
                if response.objects:
                    rel_props = response.objects[0].properties
                    relationships.append({
                        "from_table": rel_props.get('fromTableName'),
                        "from_column": rel_props.get('fromColumn'),
                        "to_table": rel_props.get('toTableName'),
                        "to_column": rel_props.get('toColumn'),
                        "relationship_type": rel_props.get('relationshipType', ''),  # ✅ Your actual schema
                        "cardinality": rel_props.get('cardinality', ''),             # ✅ Your actual schema
                        "join_type": rel_props.get('suggestedJoinType', 'INNER'),
                        "business_meaning": rel_props.get('businessMeaning', '')     # ✅ Your actual schema
                    })
                    
                    logger.info(f"      ✅ {rel_props.get('fromTableName')}.{rel_props.get('fromColumn')} -> {rel_props.get('toTableName')}.{rel_props.get('toColumn')}")
        
        return relationships
        
    except Exception as e:
        logger.warning(f"Relationship discovery failed: {e}")
        return []

def _get_adaptive_fallback_schema(search_query: str, context_data: Dict) -> str:
    """Adaptive fallback when Weaviate is unavailable"""
    
    # Use AI to determine likely tables needed
    triage_result = context_data.get("triage_result", {})
    key_entities = triage_result.get("key_entities", [])
    
    datasets = []
    columns_by_dataset = {}
    
    # Adaptive table selection based on entities
    if any(entity in str(key_entities).lower() for entity in ['customer', 'contact', 'email', 'name']):
        datasets.append({
            "table_name": "customer",
            "athena_table_name": "amspoc3test.customer",
            "relevance_score": 0.9,
            "fallback_reason": "Customer entity detected in query"
        })
        
        columns_by_dataset["amspoc3test.customer"] = [
            {"column_name": "ID", "data_type": "INTEGER", "athena_data_type": "INTEGER", "ai_relevance_score": 9.0, "is_primary_key": True},
            {"column_name": "FirstName", "data_type": "VARCHAR", "athena_data_type": "VARCHAR", "ai_relevance_score": 8.5, "is_primary_key": False},
            {"column_name": "LastName", "data_type": "VARCHAR", "athena_data_type": "VARCHAR", "ai_relevance_score": 8.5, "is_primary_key": False},
            {"column_name": "emailaddress", "data_type": "VARCHAR", "athena_data_type": "VARCHAR", "ai_relevance_score": 9.2, "is_primary_key": False}
        ]
    
    if any(entity in str(key_entities).lower() for entity in ['move', 'booking', 'service', '2024']):
        datasets.append({
            "table_name": "moves",
            "athena_table_name": "amspoc3test.moves",
            "relevance_score": 0.9,
            "fallback_reason": "Move/booking entity detected in query"
        })
        
        columns_by_dataset["amspoc3test.moves"] = [
            {"column_name": "ID", "data_type": "INTEGER", "athena_data_type": "INTEGER", "ai_relevance_score": 9.0, "is_primary_key": True},
            {"column_name": "CustomerID", "data_type": "INTEGER", "athena_data_type": "INTEGER", "ai_relevance_score": 9.5, "is_foreign_key": True},
            {"column_name": "BookedDate", "data_type": "VARCHAR", "athena_data_type": "VARCHAR", "ai_relevance_score": 8.0, "is_primary_key": False},
            {"column_name": "Status", "data_type": "VARCHAR", "athena_data_type": "VARCHAR", "ai_relevance_score": 7.5, "is_primary_key": False}
        ]
    
    relationships = []
    if len(datasets) > 1:
        relationships.append({
            "from_table": "moves",
            "from_column": "CustomerID",
            "to_table": "customer", 
            "to_column": "ID",
            "join_type": "LEFT JOIN",
            "business_purpose": "Link moves to customer information"
        })
    
    return json.dumps({
        "search_query": search_query,
        "context_analysis": context_data,
        "datasets": datasets,
        "columns_by_dataset": columns_by_dataset,
        "relationships": relationships,
        "contextual_warnings": context_data.get("contextual_warnings", []),
        "fallback_mode": True,
        "adaptive_fallback_applied": True,
        "warning": "🚨 ADAPTIVE FALLBACK MODE - Connect to Weaviate for full precision"
    }, indent=2)