import streamlit as st
import json
from neo4j import GraphDatabase
import google.genai as genai
from google.genai.types import FunctionDeclaration, GenerateContentConfig, Part, Tool, Content
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="LVMH SupplyChain Superbrain",
    page_icon="🏥",
    layout="wide"
)

def setup_connections():
    logger.info("Setting up connections to Neo4j and Google AI")
    
    try:
        neo4j_uri = st.secrets["NEO4J_URI"]
        neo4j_user = st.secrets["NEO4J_USER"] 
        neo4j_password = st.secrets["NEO4J_PASSWORD"]
        google_api_key = st.secrets["api_key"]
        
        logger.info(f"Successfully loaded secrets from Streamlit secrets")
    except Exception as e:
        logger.warning(f"Could not load from secrets: {e}. Using backup values.")
    
    return neo4j_uri, neo4j_user, neo4j_password, google_api_key

def run_cypher_query(query, neo4j_uri, neo4j_user, neo4j_password):
    logger.info(f"Executing Neo4j query: {query}")
    
    driver = GraphDatabase.driver(
        neo4j_uri, 
        auth=(neo4j_user, neo4j_password)
    )

    try:
        with driver.session() as session:
            result = session.run(query)
            result_list = [
                {key: (dict(value) if hasattr(value, "keys") else value) for key, value in record.items()}
                for record in result
            ]
        
        logger.info(f"Query completed successfully. Result count: {len(result_list)}")
        return json.dumps(result_list, indent=2)
    except Exception as e:
        logger.error(f"Error executing Neo4j query: {e}")
        return json.dumps({"error": str(e)})
    finally:
        driver.close()

def main():
    st.title("LVMH SupplyChain Superbrain")
    
    st.markdown("### Ask a question about LVMH Vendors, Supply Chain or operations")
    st.markdown("""
    Example questions:
    - Can you give me executives/managers names in LVMH?
    - What were the issues found within Zenith Textiles?
    - Where and which product has HS code risk?
    """)
    
    neo4j_uri, neo4j_user, neo4j_password, google_api_key = setup_connections()
    user_query = st.text_input("Enter your question:", key="user_query")
    
    if st.button("Get Answer", key="process_button"):
        if not user_query:
            st.warning("Please enter a question first")
            return
        
        with st.spinner("Processing your question..."):
            logger.info(f"Processing user query: {user_query}")
            
            try:
                logger.info("Initializing Google AI client")
                client = genai.Client(api_key=google_api_key)
                
                run_query = FunctionDeclaration(
                    name="run_cypher_query",
                    description="Run a Cypher query against the Neo4j database.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The Cypher query to run.",
                            },
                        },
                        "required": ["query"],
                    },
                )
                
                data_tool = Tool(
                    function_declarations=[
                        run_query,
                    ]
                )
                
                system_prompt = """
                  # Objective:
                  Query a Neo4j Knowledge Graph to extract, analyze, and synthesize answers strictly based on graph database (referred as Knowledge graph).

                  # Graph Structure
                  Nodes: Community, SubCommunity, Entity, Chunk (referred as Document Data of Knowledge graph), EntityType
                  Subcommunity -> Community  has "BELONGS_TO"
                  Entity -> SubCommunity has "BELONGS_TO"
                  Chunk -> Entity has "RELATED_TO"
                  EntityType -> Entity has "RELATED_TO"
                  Available entity types - ("HS Code"/"Organization"/"Business Model"/"Industry"/"Product Category"/"Business Process"/"Geographic Region"/"Brand"/"Product"/"Platform"/"Communication Channel"/"Business Metric"/"Person"/"Location"/"Event")
                  Relationships: "BELONGS_TO"/"OPERATES"/"OWNS"/"RELATED_TO"/"RELATEDTO"/"MANAGES"/"USES"/"MONITORS"/"COVERS"/"SPANS"/"SOURCESFROM"/"CONTAINS"/"VENDOR"/"ALTERNATEVENDORLOCATION"/"SUBJECTTO"/"HASHSCODE"/"REQUIRES"/"BRAND"/"COMPOSEDOF"

                  # Workflow
                  Step 1: Global Search — Decompose and Search Intelligently
                  Break down the user query into individual, atomic keywords.
                  Example: "claimed rights and quotation" → ["claim", "claimed", "quote", "quotation", "rights"].
                  Search iteratively:
                  DO NOT search using all keywords at once.
                  Start with the most meaningful/central keyword first.
                  Perform a CONTAINS search over the following fields:
                  Chunk.summary
                  Community.comm_name
                  Community.comm_description
                  Subcommunity.comm_name
                  Subcommunity.comm_description
                  Subcommunity.keywords
                  Subcommunity.insights
                  Important:
                  After searching with a keyword, analyze the results.
                  If the results are sufficient for understanding, immediately proceed to Step 2.
                  Only if the information is insufficient, then pick the next most relevant keyword and search again.
                  Stop as soon as useful data is found. Do not exhaustively search all keywords.

                  Step 2: Local Search — Expand via Related Entities
                  For each matched Chunk, Community, or Subcommunity:
                  Retrieve linked Entities through the BELONGS_TO relationship using **IDs** (Chunk.id, Community.id etc) of matched Chunk, Community, or Subcommunity.
                  For each retrieved Entity:
                  Extract:
                  Entity descriptions(entity_description) and Relationship(relationship_description) descriptions between them
                  Goal:
                  Understand how different Chunks/Communities/Subcommunities are interconnected via Entities.
                  Entities can appear across multiple parts of the graph and create deeper insights.

                  Step 3 (Optional): Global Search — Leverage Entity Types if Needed
                  If required information is still missing, then:
                  Explore connected Entity Types through the RELATED_TO relationship.
                  Example: if you need all "PERSON" nodes linked to the context.
                  Only perform this step if a specific type of entity needs to be searched across the graph.

                  Step 4: Synthesize the Answer
                  Merge:
                  Chunk text
                  Entity facts
                  Entity relationship facts
                  SubCommunity insights
                  Entity types (if explored)
                  Then synthesize a coherent, complete answer based on all collected data.

                  # Response Rules
                  **Only the final synthesized response must be in bold.**
                  *All intermediate notes must be italicized.*
                  Always conclude with a positive or forward-looking remark.
                  If you are performing global search, output it as "Performing Global Search..." else "Performing Local Search" or "performing Optional Global Search" and so on
                  Do not include use term - chunk , use document Database.

                  # Strict Rules
                  YOu need to perform both global and local search compulsarily.
                  Refer Graph as Knowledgr graph and chunk as document database
                  No combining keywords unless individually searched.
                  No cross-node assumptions unless supported by explicit relations.
                  No invented or assumed facts — only synthesize from observed graph patterns.
                  Never include any IDs in your response, Just output without any specification of them. Ex : "I found it in a document database"

                  Example Workflows
                  Service Availability:
                  Break into keywords → Search Chunks → Find Entities → Extract OFFEREDBY links → Gather SubCommunity/Community insights → Synthesize.

                  Claims Resolution:
                  Break into claims, underserved → Search Chunks → Extract Entities → Trace claim-processing relationships → Contextualize via Communities → Synthesize.
                """
                
                logger.info("Creating chat instance with Gemini")
                chat = client.chats.create(
                    model="gemini-2.0-flash",
                    config=GenerateContentConfig(
                        temperature=0,
                        tools=[data_tool],
                        system_instruction=system_prompt
                    ),
                )
                
                logger.info("Sending user query to Gemini")
                response = chat.send_message(user_query)
                
                main_container = st.container()
                
                with main_container:
                    st.subheader("Your Question")
                    st.write(user_query)
                    process_steps = st.container()
                    
                    answer_container = st.container()
                
                step_content = []
                final_answer_text = ""
                
                
                logger.info("Starting processing loop for function calls")
                while True:
                    logger.info(response.candidates)
                    if not hasattr(response, "candidates") or not response.candidates or not response.candidates[0].content.parts:
                        break
                    logger.info(response.candidates)
                        
                    function_calls = []
                    for part in response.candidates[0].content.parts:
                        logger.info(response.candidates)
                        if hasattr(part, "function_call") and part.function_call:
                            function_calls.append(part.function_call)
                    
                    if not function_calls:
                        text_response = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, "text") and part.text)
                        if text_response.strip():
                            final_answer_text = text_response
                        break
                    
                    function_responses = []
                    interim_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, "text") and part.text)
                    
                    if interim_text.strip():
                        step_content.append(interim_text)
                        
                        with process_steps:
                            st.markdown("\n\n".join(step_content))
                        
                        step_content = []  
                        logger.info(f"Interim text: {interim_text}")
                    
                    for func_call in function_calls:
                        function_name = func_call.name
                        args = func_call.args
                        
                        logger.info(f"Function call detected: {function_name}")
                        
                        if function_name == "run_cypher_query" and "query" in args:
                            query = args["query"]
                            
                            data = run_cypher_query(query, neo4j_uri, neo4j_user, neo4j_password)
                            print(data)
                            
                            function_responses.append(
                                Part.from_function_response(
                                    name=function_name,
                                    response={"results": data}
                                )
                            )
                             
                    if function_responses:
                        logger.info(f"Sending function responses back to the model")
                        response = chat.send_message(function_responses)
                    else:
                        break
                
                if hasattr(response, "candidates") and response.candidates and response.candidates[0].content.parts:
                    final_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, "text") and part.text)
                    if final_text.strip():
                        final_answer_text = final_text
                        logger.info(f"Final answer obtained: {len(final_text)} characters")
                
                with answer_container:
                    st.subheader("Answer")
                    st.markdown(final_answer_text)
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
