# tools/athena_tools.py - PURE AI-DRIVEN ATHENA TOOLS
# No static methods - Pure AI decision making for SQL generation

import os
import json
import time
import logging
from typing import Dict, Any, List
import boto3
from dotenv import load_dotenv
from crewai.tools import tool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PureAIQueryEngine:
    """
    Pure AI-driven SQL query engine using Claude 4 for all decision making
    - No hardcoded rules or static methods
    - AI analyzes question and context to generate optimal SQL
    - AI determines query type, table selection, and column usage
    - AI applies Athena best practices automatically
    """
    
    def __init__(self):
        """Initialize Claude 4 for AI-driven SQL generation"""
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.model_id = os.getenv('BEDROCK_MODEL_ID_CLAUDE', 'anthropic.claude-sonnet-4-20250514-v1:0')
    
    def generate_sql_with_ai(self, question: str, context: Dict) -> str:
        """
        Use Claude 4 to generate optimized SQL based on question and discovered schema
        
        Args:
            question: Original business question
            context: Complete adaptive pipeline context with datasets and columns
            
        Returns:
            AI-generated SQL query optimized for Athena
        """
        # Extract schema information from adaptive pipeline
        stage_2 = context.get('stage_2_precision_discovery', {})
        datasets = stage_2.get('datasets', [])
        columns_by_dataset = stage_2.get('columns_by_dataset', {})
        relationships = stage_2.get('relationships', [])
        
        # Prepare comprehensive context for AI
        schema_context = self._prepare_schema_context(datasets, columns_by_dataset, relationships)
        
        # Create AI prompt for SQL generation
        prompt = f"""
You are an expert AWS Athena SQL generator. Generate optimized SQL following these requirements:

QUESTION: "{question}"

AVAILABLE SCHEMA:
{schema_context}

AWS ATHENA BEST PRACTICES TO FOLLOW:
1. Use specific column names (never SELECT *)
2. Use TRY_CAST for safe type conversion
3. Use DATE literals like DATE '2024-01-01' for date filtering
4. Use proper Trino/Presto date functions: date_parse(column, '%Y-%m-%d %H:%i:%s')
5. Always add LIMIT clause for performance
6. Use LOWER(TRIM()) for string comparisons
7. Filter out NULL values explicitly
8. Use efficient join order (larger table first)
9. Use CTEs for complex queries
10. Add proper WHERE clauses for partition pruning

COLUMN PRIORITIZATION:
- Use columns with highest ai_relevance_score first
- Prioritize columns with score >= 9.0 for main SELECT
- Use columns with score >= 8.0 for WHERE clauses
- Consider is_primary_key and is_foreign_key for joins

GENERATE OPTIMIZED SQL:
1. Analyze the question to understand what data is needed
2. Select the most relevant dataset(s) based on relevance_score
3. Choose optimal columns based on ai_relevance_score
4. Apply appropriate filters and conditions
5. Use proper Athena syntax and best practices
6. Add meaningful column aliases
7. Include performance optimizations

Return ONLY the SQL query, no explanation.
        """
        
        try:
            # Generate SQL using Claude 4
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0.1  # Low temperature for consistent SQL generation
                })
            )
            
            result = json.loads(response['body'].read())
            ai_generated_sql = result['content'][0]['text'].strip()
            
            # Clean up the SQL (remove any markdown formatting)
            ai_generated_sql = ai_generated_sql.replace('```sql', '').replace('```', '').strip()
            
            logger.info("✅ AI generated SQL successfully")
            return ai_generated_sql
            
        except Exception as e:
            logger.error(f"❌ AI SQL generation failed: {e}")
            # Fallback to simple query if AI fails
            return self._get_fallback_query(datasets, columns_by_dataset)
    
    def _prepare_schema_context(self, datasets: List[Dict], columns_by_dataset: Dict, relationships: List[Dict]) -> str:
        """Prepare comprehensive schema context for AI"""
        context_parts = []
        
        # Add datasets information
        context_parts.append("DATASETS (ordered by relevance_score):")
        for i, dataset in enumerate(datasets, 1):
            table_name = dataset.get('table_name', 'unknown')
            athena_name = dataset.get('athena_table_name', 'unknown')
            relevance = dataset.get('relevance_score', 0)
            purpose = dataset.get('business_purpose', 'No description')
            
            context_parts.append(f"""
{i}. TABLE: {table_name}
   Athena Name: {athena_name}
   Relevance Score: {relevance:.3f}
   Business Purpose: {purpose[:200]}...
            """)
        
        # Add columns information
        context_parts.append("\nCOLUMNS (ordered by AI relevance score):")
        for athena_table, columns in columns_by_dataset.items():
            context_parts.append(f"\nTable: {athena_table}")
            for col in columns[:10]:  # Top 10 columns only
                col_name = col.get('column_name', 'unknown')
                data_type = col.get('data_type', 'unknown')
                ai_score = col.get('ai_relevance_score', 0)
                description = col.get('description', 'No description')
                is_pk = col.get('is_primary_key', False)
                is_fk = col.get('is_foreign_key', False)
                
                context_parts.append(f"""
   - {col_name} ({data_type})
     AI Score: {ai_score:.1f}
     Description: {description[:100]}...
     Primary Key: {is_pk}, Foreign Key: {is_fk}
                """)
        
        # Add relationships information
        if relationships:
            context_parts.append("\nRELATIONSHIPS:")
            for rel in relationships:
                from_table = rel.get('from_table', 'unknown')
                from_col = rel.get('from_column', 'unknown')
                to_table = rel.get('to_table', 'unknown')
                to_col = rel.get('to_column', 'unknown')
                join_type = rel.get('join_type', 'JOIN')
                
                context_parts.append(f"""
   - {from_table}.{from_col} -> {to_table}.{to_col} ({join_type})
                """)
        
        return "\n".join(context_parts)
    
    def _get_fallback_query(self, datasets: List[Dict], columns_by_dataset: Dict) -> str:
        """Simple fallback query if AI fails"""
        if not datasets:
            return "SELECT 'No datasets available' as message"
        
        primary_table = datasets[0].get('athena_table_name', 'amspoc3test.customer')
        columns = columns_by_dataset.get(primary_table, [])
        
        if columns:
            # Use top 3 columns
            select_cols = [col['column_name'] for col in columns[:3]]
            return f"""
SELECT {', '.join(select_cols)}
FROM {primary_table}
WHERE ID IS NOT NULL
LIMIT 10
            """.strip()
        else:
            return f"SELECT * FROM {primary_table} LIMIT 10"


class PureAIAnalysisEngine:
    """
    Pure AI-driven result analysis engine
    - No hardcoded analysis patterns
    - AI analyzes results in context of original question
    - AI provides business insights and recommendations
    """
    
    def __init__(self):
        """Initialize Claude 4 for AI-driven analysis"""
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.model_id = os.getenv('BEDROCK_MODEL_ID_CLAUDE', 'anthropic.claude-sonnet-4-20250514-v1:0')
    
    def analyze_results_with_ai(self, question: str, results: Dict, context: Dict = None) -> Dict:
        """
        Use Claude 4 to analyze query results and provide business insights
        
        Args:
            question: Original business question
            results: Athena query results
            context: Optional context from adaptive pipeline
            
        Returns:
            AI-generated analysis with insights and recommendations
        """
        # Prepare results context for AI
        results_context = self._prepare_results_context(results)
        
        # Create AI prompt for result analysis
        prompt = f"""
You are an expert business data analyst. Analyze these query results and provide comprehensive business insights.

ORIGINAL QUESTION: "{question}"

QUERY RESULTS:
{results_context}

ANALYSIS REQUIREMENTS:
1. Provide a direct answer to the original question
2. Extract key business insights from the data
3. Assess the business impact and implications
4. Identify any risks or opportunities
5. Provide strategic recommendations
6. Evaluate data quality and completeness
7. Assign a confidence level (HIGH/MEDIUM/LOW)

RESPONSE FORMAT (JSON):
{{
    "direct_answer": "Clear, concise answer to the original question",
    "key_insights": [
        "Insight 1 with specific data points",
        "Insight 2 with quantitative details",
        "Insight 3 with trend information"
    ],
    "strategic_analysis": {{
        "business_impact": "Assessment of business significance",
        "confidence_level": "HIGH/MEDIUM/LOW",
        "risk_factors": ["Risk 1", "Risk 2"],
        "opportunities": ["Opportunity 1", "Opportunity 2"],
        "strategic_recommendations": [
            "Actionable recommendation 1",
            "Actionable recommendation 2"
        ]
    }},
    "data_quality_assessment": "Evaluation of data completeness and reliability",
    "performance_insights": "Analysis of query performance and optimization opportunities"
}}

Return ONLY valid JSON, no additional text.
        """
        
        try:
            # Generate analysis using Claude 4
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1500,
                    "temperature": 0.2  # Slightly higher for creative insights
                })
            )
            
            result = json.loads(response['body'].read())
            ai_analysis = result['content'][0]['text'].strip()
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', ai_analysis, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback if JSON extraction fails
                return self._get_fallback_analysis(question, results)
                
        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            return self._get_fallback_analysis(question, results)
    
    def _prepare_results_context(self, results: Dict) -> str:
        """Prepare results context for AI analysis"""
        if results.get('status') == 'error':
            error_details = results.get('error', {})
            return f"""
ERROR: Query execution failed
Error Code: {error_details.get('code', 'Unknown')}
Error Message: {error_details.get('message', 'No details')}
            """
        
        rows = results.get('rows', [])
        columns = results.get('columns', [])
        performance = results.get('performance_metrics', {})
        
        context_parts = [
            f"COLUMNS: {', '.join(columns)}",
            f"ROW COUNT: {len(rows)}",
            f"EXECUTION TIME: {performance.get('execution_time_ms', 0)}ms",
            f"DATA SCANNED: {performance.get('data_scanned_mb', 0)} MB"
        ]
        
        if rows:
            context_parts.append("\nSAMPLE DATA:")
            for i, row in enumerate(rows[:5]):  # Show first 5 rows
                row_data = dict(zip(columns, row))
                context_parts.append(f"Row {i+1}: {row_data}")
            
            if len(rows) > 5:
                context_parts.append(f"... and {len(rows) - 5} more rows")
        
        return "\n".join(context_parts)
    
    def _get_fallback_analysis(self, question: str, results: Dict) -> Dict:
        """Fallback analysis if AI fails"""
        if results.get('status') == 'error':
            return {
                "direct_answer": "Query execution failed - unable to provide analysis",
                "key_insights": ["Query execution error prevented analysis"],
                "strategic_analysis": {
                    "business_impact": "Data access issues preventing business insights",
                    "confidence_level": "LOW",
                    "risk_factors": ["Data availability", "Query execution"],
                    "opportunities": ["Resolve technical issues"],
                    "strategic_recommendations": ["Fix query execution problems", "Verify data access"]
                },
                "data_quality_assessment": "Unable to assess - query failed",
                "performance_insights": "Query execution failed"
            }
        
        rows = results.get('rows', [])
        return {
            "direct_answer": f"Analysis completed: {len(rows)} records found",
            "key_insights": [f"Query returned {len(rows)} records"],
            "strategic_analysis": {
                "business_impact": "Data retrieved successfully",
                "confidence_level": "MEDIUM",
                "risk_factors": ["AI analysis unavailable"],
                "opportunities": ["Data available for further analysis"],
                "strategic_recommendations": ["Review results manually", "Apply business context"]
            },
            "data_quality_assessment": f"Retrieved {len(rows)} records successfully",
            "performance_insights": "Query executed successfully"
        }


# Initialize AI engines
ai_query_engine = PureAIQueryEngine()
ai_analysis_engine = PureAIAnalysisEngine()


# --- TOOL 1: PURE AI SQL GENERATION ---
@tool
def sql_generation_tool(question: str, context_json: str) -> str:
    """
    Pure AI-driven SQL generation using Claude 4 for all decision making.
    
    No hardcoded rules or static methods - AI analyzes the question and context
    to generate optimized Athena SQL queries following best practices.
    
    Args:
        question: The business question to answer
        context_json: JSON string containing adaptive pipeline results
    
    Returns:
        AI-generated SQL query optimized for AWS Athena
    """
    logger.info("🤖 Pure AI SQL Generation Tool started")
    
    try:
        # Parse context from adaptive pipeline
        if isinstance(context_json, str):
            context = json.loads(context_json)
        else:
            context = context_json
        
        logger.info(f"🧠 Question: {question}")
        
        # Let AI generate SQL based on question and context
        ai_generated_sql = ai_query_engine.generate_sql_with_ai(question, context)
        
        logger.info("✅ AI SQL generation completed")
        return ai_generated_sql
        
    except Exception as e:
        logger.error(f"❌ SQL generation error: {e}")
        return "SELECT 'SQL generation error' as error_message"


# --- TOOL 2: ENHANCED ATHENA EXECUTION ---
@tool  
def athena_execution_tool(sql_query: str) -> str:
    """
    Enhanced Athena execution with comprehensive monitoring and error handling.
    
    Args:
        sql_query: The SQL query to execute
    
    Returns:
        JSON string with results, performance metrics, or error information
    """
    logger.info("🚀 Enhanced Athena Execution Tool started")
    
    try:
        # Initialize Athena client
        athena_client = boto3.client(
            'athena',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        # Configure output location
        account_id = os.getenv('AWS_ACCOUNT_ID', 'default')
        region = os.getenv('AWS_REGION', 'us-east-1')
        output_location = os.getenv(
            'ATHENA_OUTPUT_LOCATION',
            f"s3://aws-athena-query-results-{account_id}-{region}/"
        )
        
        # Start query execution
        start_time = time.time()
        start_response = athena_client.start_query_execution(
            QueryString=sql_query,
            ResultConfiguration={
                'OutputLocation': output_location,
                'EncryptionConfiguration': {
                    'EncryptionOption': 'SSE_S3'
                }
            },
            QueryExecutionContext={
                'Database': os.getenv('ATHENA_DATABASE', 'default')
            },
            WorkGroup=os.getenv('ATHENA_WORKGROUP', 'primary')
        )
        
        query_execution_id = start_response['QueryExecutionId']
        logger.info(f"🆔 Query ID: {query_execution_id}")
        
        # Poll for completion with exponential backoff
        max_attempts = 60
        poll_interval = 1
        
        for attempt in range(max_attempts):
            response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
            execution = response['QueryExecution']
            status = execution['Status']['State']
            
            if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            
            time.sleep(min(poll_interval * (1.2 ** (attempt // 10)), 5))
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        if status == 'SUCCEEDED':
            # Get results
            results_response = athena_client.get_query_results(
                QueryExecutionId=query_execution_id,
                MaxResults=1000
            )
            
            # Process results
            result_set = results_response['ResultSet']
            column_info = result_set['ResultSetMetadata']['ColumnInfo']
            columns = [col['Label'] for col in column_info]
            column_types = [col['Type'] for col in column_info]
            
            rows = []
            result_rows = result_set['Rows']
            data_rows = result_rows[1:] if len(result_rows) > 1 else result_rows
            
            for row in data_rows:
                row_data = []
                for cell in row['Data']:
                    value = cell.get('VarCharValue')
                    row_data.append(value)
                rows.append(row_data)
            
            # Extract performance statistics
            statistics = execution.get('Statistics', {})
            data_scanned_bytes = statistics.get('DataScannedInBytes', 0)
            data_scanned_mb = round(data_scanned_bytes / (1024 * 1024), 2)
            
            response_data = {
                "status": "success",
                "columns": columns,
                "column_types": column_types,
                "rows": rows,
                "row_count": len(rows),
                "performance_metrics": {
                    "execution_time_ms": execution_time_ms,
                    "engine_execution_time_ms": statistics.get('EngineExecutionTimeInMillis', execution_time_ms),
                    "data_scanned_mb": data_scanned_mb,
                    "data_scanned_bytes": data_scanned_bytes,
                    "query_queue_time_ms": statistics.get('QueryQueueTimeInMillis', 0),
                    "query_planning_time_ms": statistics.get('QueryPlanningTimeInMillis', 0)
                },
                "query_id": query_execution_id
            }
            
        else:
            error_reason = execution['Status'].get('StateChangeReason', 'Query failed')
            response_data = {
                "status": "error",
                "error": {
                    "code": "EXECUTION_ERROR",
                    "message": error_reason,
                    "query_id": query_execution_id,
                    "execution_time_ms": execution_time_ms
                }
            }
        
        logger.info(f"✅ Athena execution completed: {response_data['status']}")
        return json.dumps(response_data, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Athena execution error: {e}")
        return json.dumps({
            "status": "error",
            "error": {
                "code": "TOOL_ERROR",
                "message": str(e),
                "type": type(e).__name__
            }
        }, indent=2)


# --- TOOL 3: PURE AI DATA ANALYSIS ---
@tool
def data_analysis_tool(question: str, athena_results_json: str) -> str:
    """
    Pure AI-driven data analysis using Claude 4 for comprehensive insights.
    
    No hardcoded analysis patterns - AI analyzes results in context of the
    original question to provide strategic business insights.
    
    Args:
        question: The original business question
        athena_results_json: JSON string with Athena execution results
    
    Returns:
        JSON string with AI-generated analysis and strategic insights
    """
    logger.info("🤖 Pure AI Data Analysis Tool started")
    
    try:
        # Parse Athena results
        if isinstance(athena_results_json, str):
            results = json.loads(athena_results_json)
        else:
            results = athena_results_json
        
        logger.info(f"🧠 Analyzing results for: {question}")
        
        # Let AI analyze results and provide insights
        ai_analysis = ai_analysis_engine.analyze_results_with_ai(question, results)
        
        logger.info("✅ AI analysis completed")
        return json.dumps(ai_analysis, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Data analysis error: {e}")
        return json.dumps({
            "direct_answer": f"Analysis error: {str(e)}",
            "key_insights": ["AI analysis could not be completed"],
            "strategic_analysis": {
                "business_impact": "Unable to provide insights due to analysis error",
                "confidence_level": "LOW",
                "risk_factors": ["Technical analysis issues"],
                "opportunities": ["Resolve analysis errors"],
                "strategic_recommendations": ["Review data format", "Check analysis requirements"]
            },
            "data_quality_assessment": "Analysis failed due to technical error",
            "performance_insights": "Unable to assess performance"
        }, indent=2)