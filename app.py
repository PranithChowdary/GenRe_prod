import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import json
import os
import math


# CORE ARCHITECTURES

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

# GenRe_V3
class GenReV3(nn.Module):
    def __init__(self, cont_vocabs, cat_vocabs, emb_dim=64, layers=6, heads=4, ff_dim=256):
        super().__init__()
        self.num_cont, self.num_cat = len(cont_vocabs), len(cat_vocabs)
        self.seq_len = self.num_cont + self.num_cat
        self.sos_emb = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.cont_embs = nn.ModuleList([nn.Embedding(v, emb_dim) for v in cont_vocabs])
        self.cat_embs = nn.ModuleList([nn.Embedding(v, emb_dim) for v in cat_vocabs])
        self.pos_encoder = PositionalEncoding(emb_dim, self.seq_len + 1)
        self.transformer = nn.Transformer(d_model=emb_dim, nhead=heads, num_encoder_layers=layers, 
                                          num_decoder_layers=layers, dim_feedforward=ff_dim, batch_first=True)
        self.cont_heads = nn.ModuleList([nn.Linear(emb_dim, v) for v in cont_vocabs])
        self.cat_heads = nn.ModuleList([nn.Linear(emb_dim, v) for v in cat_vocabs])

    def _embed(self, x_cont, x_cat):
        c_e = torch.stack([self.cont_embs[i](x_cont[:, i]) for i in range(self.num_cont)], dim=1)
        cat_e = torch.stack([self.cat_embs[i](x_cat[:, i]) for i in range(self.num_cat)], dim=1)
        return torch.cat([c_e, cat_e], dim=1)

    @torch.no_grad()
    def sample_algorithm2(self, src_bins, src_cat, temp=0.1):
        bs = src_bins.size(0)
        memory = self.transformer.encoder(self.pos_encoder(self._embed(src_bins, src_cat)))
        curr = self.sos_emb.expand(bs, -1, -1)
        s_cont, s_cat = [], []
        for i in range(self.seq_len):
            tgt = self.pos_encoder(curr)
            mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            h = self.transformer.decoder(tgt, memory, tgt_mask=mask)[:, -1, :]
            if i < self.num_cont:
                logits = self.cont_heads[i](h) / temp
                choice = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                s_cont.append(choice)
                next_emb = self.cont_embs[i](choice.squeeze(-1)).unsqueeze(1)
            else:
                idx = i - self.num_cont
                logits = self.cat_heads[idx](h) / temp
                choice = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                s_cat.append(choice)
                next_emb = self.cat_embs[idx](choice.squeeze(-1)).unsqueeze(1)
            curr = torch.cat([curr, next_emb], dim=1)
        return torch.cat(s_cont, dim=1), torch.cat(s_cat, dim=1)

# Flexible ANN Proxy for Fast Evaluation
class FlexibleANNProxy(nn.Module):
    def __init__(self, num_cont, cat_vocabs, emb_dim=32, h1=256, h2=128):
        super().__init__()
        self.cont_proj = nn.Linear(num_cont, emb_dim)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(v, emb_dim) for v in cat_vocabs])
        input_dim = (1 + len(cat_vocabs)) * emb_dim
        self.classifier = nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1), nn.Sigmoid())

    def forward(self, x_cont, x_cat):
        c = self.cont_proj(x_cont).unsqueeze(1)
        cat = torch.stack([self.cat_embeddings[i](x_cat[:, i]) for i in range(len(self.cat_embeddings))], dim=1)
        combined = torch.cat([c, cat], dim=1).view(x_cont.size(0), -1)
        return self.classifier(combined)


# Data Utilities
def safe_inverse_transform(binner, X_bins, meta):
    results = []
    for i, b in enumerate(X_bins):
        edges = binner[i]
        max_valid_bin = len(edges) - 2
        safe_b = int(np.clip(b, 0, max_valid_bin))
        val = (edges[safe_b] + edges[safe_b+1]) / 2.0
        name = meta['continuous_features'][i]
        if any(k in name.lower() for k in ["acc", "inq", "pub", "delinq", "num", "mort", "collections"]):
            val = round(val)
        results.append(val)
    return results

# Domain-specific formatting
def format_financial(name, val):
    unit = "$" if any(k in name.lower() for k in ["amnt", "inc", "bal", "lim", "pymnt", "installment"]) else ""
    pct = "%" if any(k in name.lower() for k in ["rate", "util", "pct"]) else ""
    return f"{unit}{val:,.2f}{pct}"


# Asset loading with caching
BASE_DIR = "v3"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

@st.cache_resource
def load_assets():
    with open(os.path.join(DATA_DIR, "meta.json")) as f: meta = json.load(f)
    bin_edges = np.load(os.path.join(DATA_DIR, "bin_edges_v3.npy"), allow_pickle=True)
    scaler = joblib.load(os.path.join(DATA_DIR, "num_scaler.joblib"))
    X_cont = np.load(os.path.join(DATA_DIR, "X_cont.npy"))
    X_cat = np.load(os.path.join(DATA_DIR, "X_cat.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    
    cat_vocabs = [meta['categorical_vocab_sizes'][f] for f in meta['categorical_features']]
    genre = GenReV3([len(e)-1 for e in bin_edges], cat_vocabs)
    genre.load_state_dict(torch.load(os.path.join(MODEL_DIR, "genre_v3.pt"), map_location="cpu")["model_state"])
    genre.eval()
    
    ann = FlexibleANNProxy(meta['num_cont'], cat_vocabs, emb_dim=32)
    ann.load_state_dict(torch.load(os.path.join(MODEL_DIR, "ann_flexible.pt"), map_location="cpu")["model_state"])
    ann.eval()
    return meta, bin_edges, scaler, genre, ann, X_cont, X_cat, y

meta, bin_edges, scaler, genre, ann, X_cont, X_cat, y_all = load_assets()


# Recourse Generation
@torch.no_grad()
def generate_demo_recourse(fact_bins, fact_cat, k=100, lam=0.01):
    best_bins, best_cat = fact_bins.clone(), fact_cat.clone()
    f_norm = torch.tensor([safe_inverse_transform(bin_edges, fact_bins[0].tolist(), meta)], dtype=torch.float32)
    p_init = ann(f_norm, fact_cat).item()
    best_prob, best_score = p_init, -1e9

    for i_sample in range(k):
        # Temperature scheduling for maximum exploration
        temp = 0.1 + (i_sample / k) * 0.6
        gen_bins, gen_cats = genre.sample_algorithm2(fact_bins, fact_cat, temp=temp)
        
        # DEMO MANIFOLD RULE: Strict -5 to +5 bin clamp for realism
        search_radius = 3 # changed from 5 to 3 for a more focused demo
        diff = gen_bins - fact_bins
        clamped_diff = torch.clamp(diff, -search_radius, search_radius)
        
        for i in range(meta['num_cont']):
            if meta['continuous_features'][i] in meta['immutable_features']:
                gen_bins[:, i] = fact_bins[:, i]
            else:
                gen_bins[:, i] = fact_bins[:, i] + clamped_diff[:, i]
                gen_bins[:, i] = torch.clamp(gen_bins[:, i], 0, len(bin_edges[i]) - 2)

        for i, name in enumerate(meta['categorical_features']):
            if name in meta['immutable_features']:
                gen_cats[:, i] = fact_cat[:, i]

        # Evaluation
        gn_norm_vals = safe_inverse_transform(bin_edges, gen_bins[0].tolist(), meta)
        gn_norm = torch.tensor([gn_norm_vals], dtype=torch.float32)
        probs = ann(gn_norm, gen_cats).item()
        
        n_cost = torch.abs(gen_bins - fact_bins).float().sum()
        c_cost = (gen_cats != fact_cat).float().sum() * 5.0
        total_cost = (n_cost + c_cost) * 0.05

        # DEMO SCORING: Aggressively target the 90%+ probability threshold
        if probs < 0.90:
            score = probs * 100.0 - (lam * total_cost) # Heavily favor probability
        else:
            score = 1000.0 + (probs * 10.0) - total_cost # Found 90%+, now minimize cost

        if score > best_score:
            best_bins, best_cat, best_score, best_prob = gen_bins, gen_cats, score, probs
            
    return best_bins, best_cat, best_prob

# Streamlit UI
st.set_page_config(page_title="SBI Decision Intelligence", layout="wide")
st.title("🏦 SBI Smart Recourse Dashboard")

# Initialize Session State
if 'view_mode' not in st.session_state: st.session_state.view_mode = "Individual"
if 'user_idx' not in st.session_state: st.session_state.user_idx = None
if 'bulk_results' not in st.session_state: st.session_state.bulk_results = None
if 'bulk_recourses' not in st.session_state: st.session_state.bulk_recourses = None

with st.sidebar:
    st.header("Navigation")
    col_nav1, col_nav2 = st.columns(2)
    if col_nav1.button("👤 Individual"): st.session_state.view_mode = "Individual"
    if col_nav2.button("📊 Bulk Mode"): st.session_state.view_mode = "Bulk"
    
    st.divider()
    
    if st.session_state.view_mode == "Individual":
        st.header("Applicant Selection")
        if st.button("🎲 Load Random Rejected User"):
            with st.spinner("Finding highly rejected applicant..."):
                neg_indices = np.where(y_all == 0)[0]
                subset = np.random.choice(neg_indices, min(100000, len(neg_indices)))
                sub_xc, sub_xcat = torch.tensor(X_cont[subset], dtype=torch.float32), torch.tensor(X_cat[subset], dtype=torch.long)
                with torch.no_grad():
                    probs = ann(sub_xc, sub_xcat).detach().squeeze().cpu().numpy()
                truly_rejected = subset[probs < 0.3]
                if len(truly_rejected) > 0:
                    st.session_state.user_idx = int(np.random.choice(truly_rejected))
                else:
                    st.session_state.user_idx = int(np.random.choice(subset[probs < 0.5]))
    
    else:
        st.header("Bulk Controls")
        batch_size = st.slider("Batch Size", 2, 100, 5)
        if st.button("⚡ Generate Bulk Recourse"):
            neg_indices = np.where(y_all == 0)[0]
            # Select random rejected users
            subset = np.random.choice(neg_indices, batch_size, replace=False)
            
            results = []
            recourses = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uid in enumerate(subset):
                status_text.text(f"Running on CPU... \nProcessing User {i+1}/{batch_size} (ID: {uid})...")
                
                f_c_s = X_cont[uid:uid+1]
                f_c_t = torch.tensor(f_c_s, dtype=torch.float32)
                f_cat_t = torch.tensor(X_cat[uid:uid+1], dtype=torch.long)
                
                # Initial Prob
                with torch.no_grad():
                    p_start = ann(f_c_t, f_cat_t).item()
                
                # Binning
                f_b = [np.clip(np.digitize(f_c_s[0, j], bin_edges[j]) - 1, 0, len(bin_edges[j])-2) for j in range(len(f_c_s[0]))]
                f_b_t = torch.tensor([f_b], dtype=torch.long)
                
                # Generate
                r_b, r_cat, p_end = generate_demo_recourse(f_b_t, f_cat_t, k=100) # Slightly lower k for bulk speed
                
                results.append({
                    "User ID": uid,
                    "Initial Prob": f"{p_start:.2%}",
                    "Final Prob": f"{p_end:.2%}",
                    "Status": "✅ APPROVED" if p_end > 0.5 else "❌ IMPROVED",
                    "Score": p_end
                })
                
                # store complete recourse details for later display
                r_cont_vals = safe_inverse_transform(bin_edges, r_b[0].tolist(), meta)
                r_real = scaler.inverse_transform([r_cont_vals])[0]
                recourses.append({
                    "User ID": uid,
                    "current_profile": {meta['continuous_features'][j]: format_financial(meta['continuous_features'][j], scaler.inverse_transform(f_c_t)[0, j]) for j in range(len(f_c_s[0]))},
                    "suggested_profile": {meta['continuous_features'][j]: format_financial(meta['continuous_features'][j], r_real[j]) for j in range(len(f_c_s[0]))}
                })
                
                progress_bar.progress((i + 1) / batch_size)
            
            st.session_state.bulk_results = pd.DataFrame(results)
            st.session_state.bulk_recourses = recourses
            status_text.text("Bulk Generation Complete!")
            
if st.session_state.view_mode == "Individual" and st.session_state.user_idx is not None:
    idx = st.session_state.user_idx
    f_cont_scaled, f_cat = X_cont[idx:idx+1], X_cat[idx:idx+1]
    f_real = scaler.inverse_transform(f_cont_scaled)[0]
    
    with torch.no_grad():
        p_fact = ann(torch.tensor(f_cont_scaled, dtype=torch.float32), torch.tensor(f_cat, dtype=torch.long)).item()
    
    # add column descriptions for clarity
    col_descriptions = meta["descriptions"]
    
    st.write(f"### Applicant ID: {idx} - Current Approval Probability: **{p_fact:.2%}**")
    
    # Step 1: Current Profile Overview
    st.subheader(f"Step 1: Current Profile Overview")
    
    prof_data = []
    for i, name in enumerate(meta['continuous_features']):
        prof_data.append({
            "Feature": name, 
            "Value": format_financial(name, f_real[i]), 
            "Type": "Numerical",
            "description": col_descriptions.get(name, ""),
            "Status": "🔒 Immutable" if name in meta['immutable_features'] else "✏️ Mutable"
        })
    for i, name in enumerate(meta['categorical_features']):
        v_map = meta["categorical_value_maps"][name]
        inv_map = {v: k for k, v in v_map.items()}
        prof_data.append({
            "Feature": name, 
            "Value": inv_map[f_cat[0, i]], 
            "Type": "Categorical",
            "description": col_descriptions.get(name, ""), 
            "Status": "🔒 Immutable" if name in meta['immutable_features'] else "✏️ Mutable"
        })
    
    with st.expander("🔍 View Full Profile & Feature Status", expanded=False):
        st.dataframe(pd.DataFrame(prof_data), width="stretch")

    
    # Step 2: Advisory Configuration
    with st.expander("🛠️ STEP 2: Advisory Configuration", expanded=True):
        st.info("Define the allowable financial deltas based on time and effort.")
        
        # 1. Global Strategic Controls
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            timeframe = st.selectbox(
                "Target Actionable Timeframe", 
                options=[3, 6, 12], 
                format_func=lambda x: f"{x} Months", 
                index=1,
                key="ui_timeframe"
            )
        with col_ctrl2:
            base_effort = st.slider(
                "Relative Effort Level (Allowable % change)", 
                min_value=1.0, max_value=25.0, value=5.0, step=1.0,
                key="ui_effort"
            )
        
        # Calculate time-weighted percentage change for visual sliders
        # Logic: (Base Effort) * (Multiplier based on 3-month blocks)
        total_allowed_pct = (base_effort / 100.0) * (timeframe / 3.0)
        
        st.info(f"💡 **Strategic Goal:** Seeking approval with a **{total_allowed_pct:.1%}** allowable variance over a **{timeframe}-month** horizon.")
        
        tab_num1, tab_num2, tab_cat = st.tabs(["Financials & Loan Ranges", "Credit Indicator Ranges", "Categorical Transitions"])

        # High-impact features for the first tab
        priority_feats = ["loan_amnt", "annual_inc", "dti", "installment", "int_rate", "revol_util", "delinq_2yrs", "inq_last_6mths"]
        
        with tab_num1:
            cols = st.columns(2)
            for i, name in enumerate(priority_feats):
                if name in meta["continuous_features"] and name not in meta["immutable_features"]:
                    f_idx = meta["continuous_features"].index(name)
                    curr_val = float(f_real[f_idx])
                    
                    # Calculate -5/+5 Bin Manifold boundaries
                    b_idx = np.clip(np.digitize(f_cont_scaled[0, f_idx], bin_edges[f_idx]) - 1, 0, len(bin_edges[f_idx]) - 2)
                    b_min, b_max = np.clip(b_idx - 5, 0, len(bin_edges[f_idx]) - 2), np.clip(b_idx + 5, 0, len(bin_edges[f_idx]) - 2)
                    
                    s_min = float((bin_edges[f_idx][b_min] + bin_edges[f_idx][b_min+1]) / 2.0)
                    s_max = float((bin_edges[f_idx][b_max] + bin_edges[f_idx][b_max+1]) / 2.0)
                    
                    # Ensure min < max and current value is within bounds
                    s_min = min(s_min, curr_val)
                    s_max = max(s_max, curr_val + 0.01)

                    with cols[i % 2]:
                        label = f"{name} (Current: {format_financial(name, curr_val)})"
                        st.slider(label, s_min, s_max, (s_min, s_max), key=f"tab1_{name}")
        
        with tab_num2:
            cols = st.columns(2)
            # Filter remaining features to avoid Duplicate ID error
            remaining_feats = [f for f in meta["continuous_features"][:35] if f not in priority_feats and f not in meta["immutable_features"]]

            for i, name in enumerate(remaining_feats):
                f_idx = meta["continuous_features"].index(name)
                curr_val = float(f_real[f_idx])
                
                b_idx = np.clip(np.digitize(f_cont_scaled[0, f_idx], bin_edges[f_idx]) - 1, 0, len(bin_edges[f_idx]) - 2)
                b_min, b_max = np.clip(b_idx - 5, 0, len(bin_edges[f_idx]) - 2), np.clip(b_idx + 5, 0, len(bin_edges[f_idx]) - 2)
                
                s_min = float((bin_edges[f_idx][b_min] + bin_edges[f_idx][b_min+1]) / 2.0)
                s_max = float((bin_edges[f_idx][b_max] + bin_edges[f_idx][b_max+1]) / 2.0)
                
                s_min = min(s_min, curr_val)
                s_max = max(s_max, curr_val + 0.01)

                with cols[i % 2]:
                    label = f"{name} (Current: {format_financial(name, curr_val)})"
                    st.slider(label, s_min, s_max, (s_min, s_max), key=f"tab2_{name}")

        with tab_cat:
            for i, name in enumerate(meta["categorical_features"]):
                if name not in meta["immutable_features"]:
                    v_map = meta["categorical_value_maps"][name]
                    inv_map = {v: k for k, v in v_map.items()}
                    curr_val = inv_map[f_cat[0, i]]
                    options = list(v_map.keys())
                    with st.expander(f"{name} (Current: {curr_val})", expanded=False):
                        st.multiselect(f"Allowed values for {name}", options, default=curr_val, key=f"cat_{name}")

    # Step 3: Generate Recourse
    if st.button("🚀 Generate Recourse"):
        f_bins = [np.clip(np.digitize(f_cont_scaled[0, i], bin_edges[i]) - 1, 0, len(bin_edges[i])-2) for i in range(len(f_real))]
        f_bins_t = torch.tensor([f_bins], dtype=torch.long)
        
        with st.spinner("Generating recourse recommendations..."):
            r_bins, r_cat, p_end = generate_demo_recourse(f_bins_t, torch.tensor(f_cat, dtype=torch.long), k=150)
        
        r_cont_vals = safe_inverse_transform(bin_edges, r_bins[0].tolist(), meta)
        r_real = scaler.inverse_transform([r_cont_vals])[0]

        # REPORTING
        st.subheader("✅ Recommended Actions Summary")
        m1, m2 = st.columns(2)
        m1.metric("Target Prob", f"{p_end:.2%}", delta=f"{(p_end-p_fact):.2%}")
        m2.metric("Result", "STRONG APPROVAL" if p_end > 0.95 else ("APPROVED" if p_end > 0.90 else "IMPROVED"), delta_color="normal")
        
        
        comp_rows = []
        for i, name in enumerate(meta["continuous_features"]):
            if abs(r_real[i] - f_real[i]) > 1e-3:
                comp_rows.append({
                    "Feature": name, 
                    "Current": format_financial(name, f_real[i]), 
                    "Suggested": format_financial(name, r_real[i]), 
                    "Delta": f"{r_real[i]-f_real[i]:+,.2f}",
                    "Description": col_descriptions.get(name, "")
                })
        
        for i, name in enumerate(meta["categorical_features"]):
            if r_cat[0, i] != f_cat[0, i]:
                inv_map = {v: k for k, v in meta["categorical_value_maps"][name].items()}
                comp_rows.append({
                    "Feature": name, 
                    "Current": inv_map[f_cat[0, i]], 
                    "Suggested": inv_map[r_cat[0, i].item()], 
                    "Delta": "🔄 Transition",
                    "Description": col_descriptions.get(name, "")
                })
        
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), width="stretch")
        else:
            st.warning("No changes found within constraints. Try adjusting your boundaries.")

elif st.session_state.view_mode == "Bulk":
    st.header("📊 Bulk Recourse Analysis Report")
    if st.session_state.bulk_results is not None:
        df = st.session_state.bulk_results
        u_data = st.session_state.bulk_recourses
        
        # Summary Metrics
        avg_prob = df['Score'].mean()
        success_rate = (df['Score'] > 0.5).mean()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Processed", len(df))
        m2.metric("Approval Success Rate", f"{success_rate:.0%}")
        m3.metric("Avg. Target Probability", f"{avg_prob:.2%}")
        
        st.divider()
        st.subheader("Batch Results Detail")
        st.dataframe(df.drop(columns=['Score']), use_container_width=True)
        st.dataframe(u_data, width="stretch")
        
        # Download
        csv = u_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Report (CSV)", csv, "bulk_recourse_report.csv", "text/csv")
    else:
        st.info("Configure batch settings in the sidebar and click 'Generate Bulk Recourse' to begin.")

else:
    st.info("Pick a user from the sidebar to begin the Demo.")