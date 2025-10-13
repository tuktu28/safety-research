# 🚨 IMPORTANT: If you encounter "cannot import genai from google", please run this command in your terminal:
# pip install google-genai

import streamlit as st
import json
import uuid
from typing import List, Dict
import time
import urllib.parse 
from google import genai
from google.genai import types
from google.genai.errors import APIError 

# --- Configuration and Helpers ---

# Set a wide page layout and title
st.set_page_config(
    page_title="Work Safety Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- GEMINI CLIENT SETUP ---
# Use genai.Client() for configuration, which is the most robust SDK method.
try:
    # Use the specific key name requested by the user
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # IMPORTANT: The user must ensure their secrets.toml looks like this:
    # GEMINI_API_KEY = "YOUR_ACTUAL_KEY_HERE"
    # Note the double quotes around the key value.
    st.error("""
        **Configuration Error:** `GEMINI_API_KEY` not found in Streamlit secrets. 
        Please configure it in your `.streamlit/secrets.toml` file.
        
        **Fix for the current error:** Ensure your file looks exactly like this:
        ```toml
        GEMINI_API_KEY = "AIzaSyBEgaJjSdxvshpAhRsuRgO6WKCJj-JpJE" 
        ```
        (The value must be enclosed in double quotes for Streamlit to parse it as a string.)
    """)
    st.stop() # Stop the script execution if the key is missing

# Initialize the Gemini client and model name globally
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# --- New Helper Function for URL Cleaning ---
def clean_google_redirect_url(url: str) -> str:
    """
    Checks if a URL is a Google redirect (e.g., google.com/url?q=...) 
    and extracts the final, clean destination URL.
    """
    if "google.com/url?q=" in url:
        # Parse the query string parameters
        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        # The actual destination is usually in the 'q' parameter
        if 'q' in query_params and query_params['q']:
            return query_params['q'][0]
    return url


# --- Gemini API Utilities (Refactored for Dynamic Fetching) ---

# REMOVED @st.cache_data to allow repeated fetching
def fetch_more_articles() -> Dict:
    """Fetches a new batch of 10 articles using Google Search grounding and structured output (via SDK)."""

    # --- PROMPT UPDATED TO REQUEST 10 ARTICLES FOR BATCH LOADING ---
    SYSTEM_PROMPT = """You are an AI research assistant specializing in workplace safety, workers' compensation, and workplace technology. Your goal is to provide real-time, academic, and well-sourced information. You must perform a Google Search. Based on the search results (articles/papers), extract the top three, concise, high-level insights. Each insight must be 10 words or less. Also, provide a list of 10 search results containing their title, source (must be a plausible journal, organization, or news source), the **publicly browsable, full URL from the search result (MUST be the final destination link and must NOT be a Google redirect link like google.com/url?q=...)**, a recent publication date, and the **first 1-2 sentences of the article/paper's content (the snippet)**.

    CRITICAL INSTRUCTION: Your entire response must be ONLY a raw JSON object following this exact schema. DO NOT include any explanatory text, markdown formatting (like ```json), or wrapping text.

    JSON Schema:
    {
        "insights": [
            "insight 1 (10 words max)",
            "insight 2 (10 words max)",
            "insight 3 (10 words max)"
        ],
        "articles": [
            {
                "id": "A unique, short identifier (e.g., 'art-1', though we will overwrite this for uniqueness)",
                "title": "The full title of the article or paper.",
                "source": "The publication or source name.",
                "url": "The publicly browsable, full URL.",
                "date": "A recent publication date (e.g., 'Oct 2025').",
                "snippet": "The first 1-2 sentences of the article content."
            },
            ... 9 more articles ... 
        ]
    }
    """
    # NOTE: The query remains the same, but the model is expected to find different results over time.
    USER_QUERY = "Recent papers, articles, and research on workers compensation insurance, workplace safety, AI in the workplace, and technology integration."
    
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[{"google_search": {}}], # Search grounding tool MUST be present
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=USER_QUERY,
            config=config
        )
        
        json_text = response.candidates[0].content.parts[0].text
        
        # Clean up potential leading/trailing non-JSON characters if model fails strict compliance
        if json_text.strip().startswith("```"):
            json_text = json_text.strip().lstrip("```json").rstrip("```")
            
        parsed_data = json.loads(json_text)
        
        # Ensure articles have unique IDs, summary is null, and clean the URL
        for article in parsed_data.get('articles', []):
            # FIX: Force a unique UUID for the article ID regardless of what the model returned
            # This prevents the StreamlitDuplicateElementKey error by guaranteeing uniqueness.
            article['id'] = str(uuid.uuid4())
            article['summary'] = None
            
            # Clean up the URL (to fix broken links from Google redirects)
            if 'url' in article and article['url']:
                article['url'] = clean_google_redirect_url(article['url'])
            
        return parsed_data

    except APIError as e:
        st.error(f"Gemini API Error during article fetch (Status 400/403/429): {e}")
        return None
    except json.JSONDecodeError as e: 
        st.error(f"LLM returned malformed JSON (JSONDecodeError: {e}). Please retry or check prompt instruction compliance.")
        st.code(json_text, language='json')
        return None
    except Exception as e:
        st.error(f"Failed to fetch initial data: {e}")
        return None


def generate_summary(article: Dict) -> str:
    """Generates a concise summary for a single article (via SDK)."""
    
    SYSTEM_PROMPT = """You are an expert research analyst. Summarize the following article title/topic concisely in 3-4 sentences. Focus on the main finding and its relevance to workplace safety or workers' compensation. Do not use markdown formatting."""
    
    # Including the snippet for better grounding and context
    USER_QUERY = f"""
    Article Title: {article['title']}. 
    Snippet: {article['snippet']}.
    Based on the title and snippet, provide a summary.
    """

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=USER_QUERY,
            config=config
        )
        return response.text
    except APIError as e:
        st.error(f"Gemini API Error during summarization: {e}")
        return "Summary generation failed due to API error."
    except Exception as e:
        st.error(f"Failed to generate summary: {e}")
        return "Summary generation failed."


# --- State Initialization ---

def init_session_state():
    """Initializes necessary session state variables and loads initial data."""
    if 'articles' not in st.session_state:
        st.session_state.articles = []
    if 'insights' not in st.session_state:
        st.session_state.insights = []
    if 'display_count' not in st.session_state:
        st.session_state.display_count = 0 
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False 

    # Check if data needs initial loading
    if not st.session_state.articles and not st.session_state.is_loading:
        st.session_state.is_loading = True
        try:
            with st.spinner("Searching the web for the initial 10 articles..."):
                data = fetch_more_articles() 
            
            if data and data.get('articles'):
                st.session_state.articles.extend(data['articles'])
                # Only set insights once
                if not st.session_state.insights:
                    st.session_state.insights = data.get('insights', []) 
                
                # Display all initially fetched articles (up to 10)
                st.session_state.display_count = len(data['articles']) 
            else:
                # Fallback data if API call fails
                if not st.session_state.articles:
                    st.session_state.insights = [
                        'AI predicts workplace hazards 80%.', 
                        'Hybrid work increases stress claims.', 
                        'Wearable tech drastically cuts injuries.'
                    ]
                    # FIX: Use generated UUIDs for fallback data to prevent key errors
                    st.session_state.articles = [
                        {'id': str(uuid.uuid4()), 'title': 'AI-Powered Risk Assessment in Manufacturing', 'source': 'Safety Journal', 'url': 'https://research-source-f1.com', 'date': 'Jan 2026', 'summary': None, 'snippet': 'This paper explores how artificial intelligence can significantly enhance risk detection.'},
                        {'id': str(uuid.uuid4()), 'title': 'The Psychological Cost of Remote Monitoring', 'source': 'Future Work Review', 'url': 'https://research-source-f2.com', 'date': 'Dec 2025', 'summary': None, 'snippet': 'New research suggests that constantly monitored remote workers face higher mental health claims.'},
                        {'id': str(uuid.uuid4()), 'title': 'Blockchain for Workers Comp Fraud Detection', 'source': 'FinTech Quarterly', 'url': 'https://research-source-f3.com', 'date': 'Nov 2025', 'summary': None, 'snippet': 'A new method using distributed ledger technology aims to dramatically reduce fraudulent workers compensation claims.'},
                    ]

                    # FIX: Apply URL cleaning to fallback data to ensure link robustness
                    for article in st.session_state.articles:
                        if 'url' in article:
                            article['url'] = clean_google_redirect_url(article['url'])
                            
                    st.session_state.display_count = len(st.session_state.articles)
                    st.warning("Could not fetch real-time data. Displaying fallback articles.")

        except Exception as e:
            st.error(f"An unexpected error occurred during initialization: {e}")
        finally:
            st.session_state.is_loading = False
            
# --- Action Handlers ---

def handle_summarize(article_id: str):
    """Generates and updates the summary for an article."""
    
    # Find the article in the main list
    article = next((a for a in st.session_state.articles if a['id'] == article_id), None)

    if not article:
        st.error("Article not found.")
        return

    with st.spinner(f"Generating summary for {article['title']}..."):
        # Call the SDK-based summary function
        summary = generate_summary(article)

    if summary and summary != "Summary generation failed.":
        # Update the article in the main list
        for i, a in enumerate(st.session_state.articles):
            if a['id'] == article_id:
                st.session_state.articles[i]['summary'] = summary
                break
        
        st.toast("Summary complete!", icon='✨')
        # Rerun to update the UI immediately
        st.rerun()

# --- Load More Handler (Now fetches new data) ---
def handle_load_more():
    """Fetches a new batch of 10 articles and increments the display count."""
    
    # Simple check to prevent double clicking during fetch
    if st.session_state.is_loading:
        st.warning("Already loading articles. Please wait.")
        return

    st.session_state.is_loading = True
    
    try:
        with st.spinner("Searching the web for 10 more articles..."):
            new_data = fetch_more_articles()
        
        if new_data and new_data.get('articles'):
            new_articles = new_data['articles']
            
            # Use a set of existing URLs/Titles to check for duplicates
            # Since URLs are cleaned in fetch_more_articles, this comparison is robust
            existing_identifiers = set(
                (a['title'], a['url']) for a in st.session_state.articles
            )
            
            # Filter out duplicates
            unique_new_articles = [
                a for a in new_articles 
                if (a['title'], a['url']) not in existing_identifiers
            ]
            
            if unique_new_articles:
                st.session_state.articles.extend(unique_new_articles)
                # Increment the display count by the number of unique articles loaded
                st.session_state.display_count += len(unique_new_articles)
                st.toast(f"Loaded {len(unique_new_articles)} new articles!", icon='📰')
            else:
                st.toast("Could not find any new unique articles right now. Try again in a moment!", icon='🔎')
        
    except Exception as e:
        st.error(f"Error loading new articles: {e}")
    finally:
        st.session_state.is_loading = False
        st.rerun() # Ensure UI updates


# --- UI Components ---

def render_article_card(article: Dict): 
    """Renders a single detailed article card with buttons, using global CSS."""
    article_id = article['id']
    
    st.markdown(
        f"""
        <div class="article-card-base">
            <p style="font-size: 0.85rem; color: #8b949e; margin-bottom: 5px;">Source: {article.get('source', 'N/A')} | Date: {article.get('date', 'Recent')}</p>
            <a class="article-title-link" href="{article['url']}" target="_blank">{article['title']}</a>
            <p class="article-snippet">{article.get('snippet', 'Snippet not available.')}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display summary if available
    if article.get('summary'):
        st.markdown(
            f"""
            <div class="summary-box">
                <p style="font-size: 0.9rem; color: #e6edf3;">{article['summary']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Use Streamlit columns for buttons (only Summarize remains)
    col_sum, col_spacer = st.columns([0.4, 0.6]) 
    
    with col_sum:
        st.button(
            "✨ Summarize",
            # The key uses the globally unique UUID to prevent the previous StreamlitDuplicateElementKey error
            key=f"sum_{article_id}",
            on_click=handle_summarize,
            args=(article_id,), 
            type="primary"
        )
        
    st.markdown("---") # Separator for neatness

def render_home_tab(articles: List[Dict], insights: List[str]):
    """Renders the Home tab content in a single-column, integrated layout."""
    
    st.markdown(f"""
        <h2 style="color: #e6edf3; font-size: 1.5rem; font-weight: 700; margin-bottom: 15px; text-align: center;">Work Safety Intelligence Dashboard</h2>
    """, unsafe_allow_html=True) 

    # 1. Insights Block 
    st.markdown("""
        <div style="background-color: #00587c; padding: 5px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4); margin-bottom: 10px;">
            <h3 style="color: white; font-size: 1.3rem; font-weight: 700; margin-bottom: 10px; text-align: center;">🔥 Top 3 Immediate Insights</h3>
        """, unsafe_allow_html=True)
            
    # CSS for insights (inline for specific styling)
    st.markdown(
        """
        <style>
        .insight-item-new {
            margin-bottom: 8px;
            font-size: 1rem;
            color: white; 
            text-align: left; 
            padding: 0 10px; 
            display: block;
            max-width: 600px; 
            margin-left: auto;
            margin-right: auto;
        }
        .insight-number-new {
            color: #d1d5db; 
            font-weight: bold;
            margin-right: 5px;
        }
        </style>
        """, unsafe_allow_html=True
    )
    for i, insight in enumerate(insights):
        st.markdown(f'<p class="insight-item-new"><span class="insight-number-new">{i+1}.</span> {insight}</p>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


    # 2. Articles List
    st.markdown(f"""
        <h3 style="color: #e6edf3; font-size: 1.3rem; font-weight: 700; margin-bottom: 15px; border-top: 1px solid #2e3b4a; padding-top: 20px; text-align: center;">Recent Research & News</h3>
        <!-- REMOVED max-height and overflow-y to allow page to grow infinitely -->
        <div style="padding: 0px 10px 10px 0px;"> 
    """, unsafe_allow_html=True) 
        
    # --- Display only the articles up to the current display_count ---
    articles_to_show = articles[:st.session_state.display_count]

    if not articles_to_show:
        st.info("No articles loaded yet. Click the button below to load data.")
    else:
        for article in articles_to_show:
            render_article_card(article)
    
    # Close the articles list div
    st.markdown("</div>", unsafe_allow_html=True) 

    # --- Load More Button Logic ---
    col_load, col_spacer = st.columns([0.4, 0.6]) 
    with col_load:
        st.button(
            "Load 10 more articles",
            key="load_more_key",
            on_click=handle_load_more,
            type="secondary",
            # This disabled check is now purely based on the loading state, since there is no hard limit on article count.
            disabled=st.session_state.is_loading 
        )
    if st.session_state.is_loading:
        st.empty() # Remove placeholder if loading
        st.markdown("<p style='text-align: center; color: #8b949e;'>Searching for more results...</p>", unsafe_allow_html=True)


# --- Main App Execution ---

if __name__ == "__main__":
    
    # --- Global Style Injection (for Dark Theme and Custom Font) ---
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
            
            /* Custom Streamlit theming to set font and dark background */
            html, body, .stApp {
                font-family: 'Inter', sans-serif !important;
                background-color: #0d1117; /* Dark slate background */
                color: #e6edf3; /* Light text color */
            }
            /* Styling for Streamlit text elements */
            h1, h2, h3, h4, .stMarkdown, .stTabs [role="tab"] {
                color: #e6edf3 !important;
                font-family: 'Inter', sans-serif !important;
            }
            /* Override Streamlit primary color for buttons/tabs to be the new deep teal */
            .stButton>button, .stTabs [aria-selected="true"] {
                background-color: #00587c; /* Primary button color */
                color: white;
            }
            /* Adjust the background of the main column containers */
            [data-testid="stVerticalBlock"] > [style*="background-color: rgb(240, 242, 246)"] {
                background-color: #0d1117;
            }
            /* Streamlit container borders */
            .stContainer {
                background-color: #161b22 !important; /* Slightly lighter card background */
            }
            [data-testid="stVerticalBlock"] > [style*="border: 1px"] {
                border-color: #2e3b4a !important; 
            }
            
            /* Hide the tabs menu now that there's only one view */
            .stTabs {
                display: none;
            }

            /* FIX: Ensure the main Streamlit container doesn't cause horizontal overflow */
            [data-testid="stAppViewContainer"] {
                overflow-x: hidden !important;
                width: 100% !important;
            }

            .stApp header {
                display: none; /* Hide default Streamlit header */
            }
            
            /* --- EXTRACTED ARTICLE CARD STYLES --- */
            .article-card-base {
                padding: 1rem;
                border-radius: 12px;
                background-color: #161b22; /* Dark card background */
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                margin-bottom: 15px;
            }
            .summary-box {
                margin-top: 10px;
                padding: 10px;
                border-radius: 8px;
                background-color: #0b213b; /* Darker blue background for summary */
                border: 1px solid #00587c; /* Summary box border color */
                color: #e6edf3;
            }
            .article-title-link {
                font-size: 1.25rem;
                font-weight: 600;
                color: #58a6ff; /* Lighter blue link for contrast */
                text-decoration: none;
            }
            .article-title-link:hover {
                color: #79c0ff;
                text-decoration: underline;
            }
            .article-snippet {
                font-size: 0.95rem;
                color: #d1d5db; /* Light gray for snippet text */
                margin-top: 5px;
                margin-bottom: 10px;
            }
            /* --- END EXTRACTED STYLES --- */

        </style>
        """, unsafe_allow_html=True)
    # --- End Global Style Injection ---
    
    init_session_state()
    
    render_home_tab(st.session_state.articles, st.session_state.insights)
