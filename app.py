import gradio as gr
import pandas as pd
from supabase import create_client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

# --- 1. CONFIGURATION ---
URL = "https://wgenfhmrusxrlglhhaei.supabase.co"
KEY = "sb_publishable_T2rvlvHV6C29ZICjIm_5Tw_qwAoIAfq"
supabase = create_client(URL, KEY)

# --- 2. THE UI FIX (CSS) ---
css_code = """
.fixed-height-table table {
    table-layout: fixed !important;
    width: 100% !important;
    border-collapse: collapse !important;
}
.fixed-height-table th, .fixed-height-table td {
    white-space: normal !important;
    word-wrap: break-word !important;
    padding: 8px !important;
    vertical-align: top !important;
    text-align: left !important;
}
.fixed-height-table th:nth-child(1), .fixed-height-table td:nth-child(1) { width: 120px !important; }
.fixed-height-table th:nth-child(2), .fixed-height-table td:nth-child(2) { width: 140px !important; }
.fixed-height-table th:nth-child(3), .fixed-height-table td:nth-child(3) { width: 150px !important; }
.fixed-height-table th:nth-child(4), .fixed-height-table td:nth-child(4) { width: 150px !important; }
.fixed-height-table th:nth-child(5), .fixed-height-table td:nth-child(5) { width: auto !important; }
"""

def get_market_pulse_data():
    response = supabase.table("comments").select("*").execute()
    df = pd.DataFrame(response.data)
    
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['comment_text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
    
    def classify_sentiment(score):
        if score >= 0.05: return "🟢 Positive"
        elif score <= -0.05: return "🔴 Negative"
        else: return "⚪ Neutral"
        
    df['classification'] = df['sentiment_score'].apply(classify_sentiment)
    df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d')
    return df[['created_at', 'author', 'sentiment_score', 'classification', 'comment_text']]

def build_dashboard():
    df = get_market_pulse_data()
    df_daily = df.groupby('created_at')['sentiment_score'].mean().reset_index()
    fig = px.line(df_daily, x='created_at', y='sentiment_score', title='Brand Sentiment Pulse')
    top_pos = df.sort_values(by='sentiment_score', ascending=False).head(2)
    top_neg = df.sort_values(by='sentiment_score', ascending=True).head(2)
    raw_data_view = df.head(100)
    return fig, top_pos, top_neg, raw_data_view

# --- 3. BUILD INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft(), css=css_code) as demo:
    gr.Markdown("# 📈 The Market Pulse Engine")
    
    with gr.Tab("Sentiment Trends"):
        plot = gr.Plot()
        refresh_btn = gr.Button("🔄 Refresh Data")
        
    with gr.Tab("Top Signals"):
        gr.Markdown("### 🌟 Highest Sentiment")
        pos_df = gr.DataFrame(elem_classes="fixed-height-table")
        gr.Markdown("### ⚠️ Lowest Sentiment")
        neg_df = gr.DataFrame(elem_classes="fixed-height-table")

    with gr.Tab("All Comments (Raw Signals)"):
        gr.Markdown("### 🕵️ Bottom-Up Audit")
        full_df = gr.DataFrame(interactive=False, elem_classes="fixed-height-table")

    def update_ui():
        return build_dashboard()

    refresh_btn.click(update_ui, outputs=[plot, pos_df, neg_df, full_df])
    demo.load(update_ui, outputs=[plot, pos_df, neg_df, full_df])

# --- 4. THE STANDALONE ENTRY POINT ---
if __name__ == "__main__":
    # share=True ensures a public link is generated even when running outside Colab
    demo.launch(share=True)
