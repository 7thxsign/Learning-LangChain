from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
import requests

@tool
def pokemon_lookup(pokemon_name: str) -> str:
    """Query the PokeAPI database for Pokemon information.
    
    Args:
        pokemon_name: The name or ID of the Pokemon to look up
    
    Returns:
        Pokemon data if found, or error message if not found
    """
    try:
        pokemon_name = pokemon_name.lower().strip()
        url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name}"
        
        print(f"\n🔍 Searching PokeAPI for: {pokemon_name}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 404:
            return f"ERROR: No Pokemon data found for '{pokemon_name}' in the PokeAPI database."
        
        if response.status_code != 200:
            return f"ERROR: Database query failed with status {response.status_code}"
        
        data = response.json()
        name = data['name'].capitalize()
        height = data['height'] / 10
        weight = data['weight'] / 10
        abilities = [ability['ability']['name'].replace('-', ' ').title() 
                     for ability in data['abilities']]
        
        result = f"""Pokemon: {name}
Height: {height} m
Weight: {weight} kg
Abilities: {', '.join(abilities)}"""
        
        print(f"✅ Data retrieved from PokeAPI\n")
        return result
        
    except Exception as e:
        return f"ERROR: {str(e)}"

# Initialize Ollama
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.3,
    base_url="http://localhost:11434"
)

# Strict Pokédex system prompt
strict_system_prompt = """You are a Pokédex - a digital encyclopedia device for Pokemon data.

MANDATORY RULES:
1. ALWAYS use the pokemon_lookup tool for ANY query that could be a Pokemon name
2. NEVER make assumptions - check the database first
3. ONLY respond to Pokemon-related queries
4. If asked about non-Pokemon topics: "Error: Query outside Pokédex database scope."
5. Report tool results exactly as returned
6. Stay in character as a scientific device

RESPONSE FORMAT:
- If tool returns data: Present it as a Pokédex entry
- If tool returns ERROR: State "No data available in database."
- If non-Pokemon query: "Error: Query outside Pokédex database scope."
"""

# Create agent WITHOUT verbose parameter
agent = create_agent(
    model=llm,
    tools=[pokemon_lookup],
    system_prompt=strict_system_prompt
)

def run_pokedex():
    """Run Pokédex interface"""
    print("🔴 Pokédex System Online")
    print("=" * 60)
    
    conversation_history = []
    
    while True:
        user_input = input("\nTrainer: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Pokédex shutting down...")
            break
        
        if not user_input:
            continue
        
        try:
            conversation_history.append({"role": "user", "content": user_input})
            
            result = agent.invoke({"messages": conversation_history})
            response = result["messages"][-1].content
            
            conversation_history = result["messages"]
            
            print(f"\nPokédex: {response}")
            
        except Exception as e:
            print(f"\n⚠️ Error: {e}")

if __name__ == "__main__":
    run_pokedex()
