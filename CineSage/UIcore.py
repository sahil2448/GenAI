import streamlit as st
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Movie Extractor",
    page_icon="🍿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Modern Look ---
st.markdown("""
<style>
    /* Styling for the main title */
    .movie-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Styling for the summary box */
    .summary-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: rgba(255, 75, 75, 0.05);
        border-left: 5px solid #FF4B4B;
        margin-top: 1rem;
        margin-bottom: 2rem;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    /* Badges for Genres and Cast */
    .badge {
        display: inline-block;
        padding: 0.3em 0.8em;
        margin: 0.2em;
        font-size: 0.85em;
        font-weight: 600;
        border-radius: 12px;
        color: white;
    }
    .badge-genre { background-color: #4CAF50; }
    .badge-cast { background-color: #2196F3; }
</style>
""", unsafe_allow_html=True)

# --- Pydantic Model ---
class Movie(BaseModel):
    title: str 
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

# --- Core Extraction Logic ---
def extract_movie_info(paragraph: str, api_key: str) -> Movie:
    # Initialize Model
    model = ChatMistralAI(
        model='mistral-small-2506', 
        mistral_api_key=api_key
    )
    
    # Setup Parser
    parser = PydanticOutputParser(pydantic_object=Movie)
    
    # Setup Prompt
    prompt = ChatPromptTemplate.from_messages([
        ('system', """
        Extract movie information from the paragraph.
        If a piece of information is missing, use null/None.
        {format_instructions}
        """),
        ("human", "{paragraph}")
    ])
    
    # Create final prompt and invoke
    final_prompt = prompt.invoke({
        "paragraph": paragraph,
        "format_instructions": parser.get_format_instructions()
    })
    
    response = model.invoke(final_prompt)
    return parser.parse(response.content)

# --- Main App UI ---
def main():
    # Load env variables (if available)
    load_dotenv()
    
    # Sidebar: Configuration
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3172/3172568.png", width=80)
        st.header("⚙️ Configuration")
        
        # API Key Input
        env_key = os.getenv("MISTRAL_API_KEY", "")
        api_key = st.text_input(
            "Mistral API Key", 
            value=env_key, 
            type="password",
            help="Enter your Mistral API Key. If set in .env, it will load automatically."
        )
        
        st.markdown("---")
        st.markdown(
            "**Powered by:**\n"
            "- 🦜🔗 LangChain\n"
            "- 🧠 Mistral AI\n"
            "- 👑 Pydantic\n"
            "- 🎈 Streamlit"
        )

    # Main Content Area
    st.title("🎬 AI Movie Data Extractor")
    st.markdown("Paste a review, synopsis, or article about a movie, and let AI structure the data for you in seconds.")

    # Input Section
    default_text = "In 2010, Christopher Nolan directed the mind-bending sci-fi thriller Inception. The film stars Leonardo DiCaprio as a professional thief who steals information by infiltrating the subconscious of his targets. The incredible cast also includes Joseph Gordon-Levitt, Elliot Page, and Tom Hardy. It was wildly successful and holds an impressive 8.8 rating on IMDB."
    
    user_input = st.text_area(
        "Enter Movie Paragraph:", 
        value=default_text, 
        height=180,
        placeholder="Type or paste the movie description here..."
    )

    # Action Button
    extract_btn = st.button("✨ Extract Information", type="primary", use_container_width=True)

    if extract_btn:
        if not api_key:
            st.error("🚨 Please enter a valid Mistral API Key in the sidebar.")
            st.stop()
            
        if not user_input.strip():
            st.warning("⚠️ Please enter a paragraph to extract data from.")
            st.stop()

        with st.spinner("🤖 Analyzing text and extracting entities..."):
            try:
                # Call extraction logic
                movie_data = extract_movie_info(user_input, api_key)
                
                # --- Success Display ---
                st.success("Extraction Complete!")
                st.divider()
                
                # Header & Summary
                st.markdown(f'<div class="movie-title">{movie_data.title}</div>', unsafe_allow_html=True)
                
                # Director & Metrics
                st.caption(f"**Directed by:** {movie_data.director if movie_data.director else 'Unknown'}")
                
                col1, col2 = st.columns(2)
                with col1:
                    year = movie_data.release_year if movie_data.release_year else "N/A"
                    st.metric(label="🗓️ Release Year", value=year)
                with col2:
                    rating = f"{movie_data.rating} / 10" if movie_data.rating else "N/A"
                    st.metric(label="⭐ Rating", value=rating)

                # Summary Box
                st.markdown("### 📝 Summary")
                st.markdown(f'<div class="summary-box">{movie_data.summary}</div>', unsafe_allow_html=True)
                
                # Tags: Genres & Cast
                col_tags1, col_tags2 = st.columns(2)
                
                with col_tags1:
                    st.markdown("### 🎭 Genres")
                    if movie_data.genre:
                        genres_html = "".join([f'<span class="badge badge-genre">{g}</span>' for g in movie_data.genre])
                        st.markdown(genres_html, unsafe_allow_html=True)
                    else:
                        st.write("No genres found.")
                        
                with col_tags2:
                    st.markdown("### 🌟 Cast")
                    if movie_data.cast:
                        cast_html = "".join([f'<span class="badge badge-cast">{c}</span>' for c in movie_data.cast])
                        st.markdown(cast_html, unsafe_allow_html=True)
                    else:
                        st.write("No cast members found.")

                # Optional: Show Raw JSON inside an expander for developers
                with st.expander("🛠️ View Raw JSON Output"):
                    st.json(movie_data.model_dump())

            except Exception as e:
                st.error("❌ An error occurred during extraction.")
                st.exception(e)

if __name__ == "__main__":
    main()