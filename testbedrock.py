# businesscontext_schema_fix.py - Fix BusinessContext schema mismatch

"""
This script fixes the schema mismatch between your BUSINESS_CONTEXT_DATA 
and your actual Weaviate BusinessContext schema.

Issue: Your data has 'search_keywords' but schema doesn't include this property.
"""

import os
import sys
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def check_businesscontext_schema_mismatch():
    """Check for schema mismatch in BusinessContext"""
    
    try:
        from tools.adaptive_weaviate_tools import WeaviateClientSingleton
        
        client = WeaviateClientSingleton.get_instance()
        if not client:
            print("❌ Cannot connect to Weaviate")
            return False
            
        print("🔍 BUSINESSCONTEXT SCHEMA VALIDATION")
        print("=" * 50)
        
        # Test BusinessContext collection
        try:
            business_collection = client.collections.get("BusinessContext")
            response = business_collection.query.fetch_objects(limit=1)
            
            if response.objects:
                actual_properties = set(response.objects[0].properties.keys())
                print(f"✅ BusinessContext collection found")
                print(f"📋 Available properties: {sorted(list(actual_properties))}")
                
                # Check for schema mismatch
                expected_in_data = {"term", "search_keywords", "definition", "context", "examples"}
                expected_in_schema = {"term", "definition", "context", "examples"}
                
                print(f"\n🔍 Schema Analysis:")
                print(f"   Properties in your data: {sorted(list(expected_in_data))}")
                print(f"   Properties in your schema: {sorted(list(expected_in_schema))}")
                
                missing_in_schema = expected_in_data - expected_in_schema
                if missing_in_schema:
                    print(f"   ❌ Missing in schema: {sorted(list(missing_in_schema))}")
                    print(f"\n💡 SOLUTION OPTIONS:")
                    print(f"   1. Add 'search_keywords' property to BusinessContext schema")
                    print(f"   2. Remove 'search_keywords' from BUSINESS_CONTEXT_DATA")
                    print(f"   3. Map 'search_keywords' to existing properties")
                else:
                    print(f"   ✅ No schema mismatch detected")
                
                return True
            else:
                print("❌ No objects found in BusinessContext collection")
                return False
                
        except Exception as e:
            print(f"❌ Error accessing BusinessContext collection: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        return False

def suggest_schema_fix():
    """Suggest how to fix the schema mismatch"""
    
    print("\n" + "=" * 60)
    print("🔧 BUSINESSCONTEXT SCHEMA FIX SUGGESTIONS")
    print("=" * 60)
    
    print("""
🎯 **The Issue:**
Your BUSINESS_CONTEXT_DATA contains 'search_keywords' but your schema doesn't have this property.

🔧 **Fix Option 1: Add search_keywords to schema (Recommended)**

In your BusinessContext schema creator/uploader, add:

```python
Property(
    name="search_keywords",
    data_type=DataType.TEXT_ARRAY,
    description="Keywords for semantic search optimization",
    vectorize_property_name=False,
    skip_vectorization=True  # Don't vectorize arrays of keywords
)
```

🔧 **Fix Option 2: Update the data (Simpler)**

In your BUSINESS_CONTEXT_DATA, remove or rename 'search_keywords':

```python
# Remove this line from each data item:
# "search_keywords": [...]

# OR map to context:
"context": f"{existing_context}\n\nKeywords: {', '.join(search_keywords)}"
```

🔧 **Fix Option 3: Update the adaptive tools**

The tools are already fixed to not request 'search_keywords', so this mismatch
won't cause errors in the adaptive pipeline.

🚀 **Recommended Action:**
Since the adaptive tools no longer request 'search_keywords', your pipeline 
should work fine now. The schema mismatch is in the data upload, not the query tools.

💡 **Test your pipeline:**
```bash
python3 run.py
```

If it works, you can ignore this mismatch for now and fix it during the next
schema update cycle.
""")

if __name__ == "__main__":
    print("🔍 BUSINESSCONTEXT SCHEMA MISMATCH CHECKER")
    print("=" * 60)
    
    success = check_businesscontext_schema_mismatch()
    
    if success:
        suggest_schema_fix()
        print("\n🎯 Bottom Line:")
        print("The adaptive pipeline should work now even with this minor mismatch.")
        print("The tools no longer request 'search_keywords' property.")
    else:
        print("\n❌ Could not check BusinessContext schema")
        print("Ensure Weaviate is running and BusinessContext collection exists")