import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(layout="wide", page_title="Market Basket Analysis App")

# ---------------------------
# Helper functions
# ---------------------------
def detect_format(df):
    """Return 'row' if row-per-item (has TransactionID & Product),
       'basket' if each row is a basket,
       else None."""
    cols = [c.lower() for c in df.columns]
    if ('transactionid' in cols or 'transaction_id' in cols or 'transaction' in cols) and any('product' in c for c in cols):
        return 'row'
    # If single column or many columns with many repeated values -> basket-style
    if df.shape[1] == 1:
        return 'basket'
    # if many columns and mostly NaN => maybe basket with item columns
    nan_frac = df.isna().mean().mean()
    if nan_frac > 0.5:
        return 'basket'
    # fallback
    return None

def rows_to_baskets(df, transaction_col, product_col):
    """Convert row-per-item dataframe to basket-format (list of lists)."""
    df = df[[transaction_col, product_col]].dropna()
    df[product_col] = df[product_col].astype(str).str.strip()
    baskets = df.groupby(transaction_col)[product_col].apply(list)
    return baskets

def basket_list_to_onehot(baskets):
    """Convert Series or list of baskets (list of item strings) to one-hot encoded DataFrame."""
    # baskets: iterable of lists
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    onehot = pd.DataFrame(te_ary, columns=te.columns_)
    return onehot

def compute_item_frequencies(baskets):
    """Return item counts and relative frequencies."""
    flat = [item for basket in baskets for item in basket]
    counts = pd.Series(flat).value_counts()
    rel = counts / counts.sum()
    df = pd.DataFrame({'item': counts.index, 'count': counts.values, 'freq': rel.values})
    return df

def cooccurrence_matrix(onehot):
    """Return co-occurrence matrix (counts) of items."""
    items = onehot.columns
    co = pd.DataFrame(np.dot(onehot.T, onehot), index=items, columns=items)
    return co

def plot_network_from_rules(rules_df, top_n=30):
    """Create networkx graph from association rules and return Plotly figure."""
    G = nx.DiGraph()
    # choose top rules by lift or confidence
    rules_sorted = rules_df.sort_values(by=['lift','confidence'], ascending=False).head(top_n)
    for _, row in rules_sorted.iterrows():
        antecedent = tuple(sorted(row['antecedents']))
        consequent = tuple(sorted(row['consequents']))
        a = ', '.join(list(antecedent))
        c = ', '.join(list(consequent))
        G.add_node(a)
        G.add_node(c)
        G.add_edge(a, c, weight=row['confidence'], lift=row['lift'])
    pos = nx.spring_layout(G, seed=42, k=0.8)
    edge_x = []
    edge_y = []
    weights = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        weights.append(edge[2].get('weight', 0.1))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        textposition="bottom center",
        text=text,
        hoverinfo='text',
        marker=dict(size=20)
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    return fig

def estimate_aov_uplift(baskets, rules_df, price_lookup=None, min_confidence=0.5):
    """
    Simple AOV uplift estimator:
    - For each basket, find applicable rules where antecedent subset of basket
    - If rule triggers, assume recommended consequent is added to the order with probability = rule.confidence
    - If price_lookup provided (dict item->price), compute dollar uplift; otherwise compute item count uplift
    Returns baseline AOV and estimated new AOV (percentage uplift).
    NOTE: This is a simulation/estimate only.
    """
    # Prepare rules list
    rules = []
    for _, r in rules_df[rules_df['confidence'] >= min_confidence].iterrows():
        antecedent = set(r['antecedents'])
        consequent = set(r['consequents'])
        conf = r['confidence']
        rules.append((antecedent, consequent, conf))

    if len(rules) == 0:
        return None

    baseline_order_values = []
    new_order_values = []

    for basket in baskets:
        basket_set = set(basket)
        # baseline value: sum of prices if price_lookup, else number of items
        if price_lookup:
            base_val = sum(price_lookup.get(it, 0) for it in basket)
        else:
            base_val = len(basket)
        added_expected = 0.0
        for antecedent, consequent, conf in rules:
            if antecedent.issubset(basket_set):
                # expected number of items added = conf * (#consequents not already in basket)
                new_items = len([i for i in consequent if i not in basket_set])
                if price_lookup:
                    added_expected += conf * sum(price_lookup.get(i,0) for i in consequent if i not in basket_set)
                else:
                    added_expected += conf * new_items
        baseline_order_values.append(base_val)
        new_order_values.append(base_val + added_expected)

    baseline_aov = np.mean(baseline_order_values)
    new_aov = np.mean(new_order_values)
    pct_uplift = (new_aov - baseline_aov) / baseline_aov * 100 if baseline_aov != 0 else None
    return {'baseline_aov': baseline_aov, 'new_aov': new_aov, 'pct_uplift': pct_uplift}

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🧺 Market Basket Analysis — Automated Data Scientist")
st.markdown("""
Upload a transaction dataset and the app will automatically:
- detect the format,
- preprocess,
- run Apriori + association rules,
- show visual dashboards and a simple AOV uplift estimate using discovered rules.
""")

with st.sidebar:
    st.header("Upload & Parameters")
    uploaded = st.file_uploader("Upload CSV (row-per-item or basket format)", type=["csv", "txt"])
    st.markdown("**Apriori params**")
    min_support = st.slider("Minimum support", 0.001, 0.5, 0.01, step=0.001, help="Relative support for frequent itemsets")
    min_confidence = st.slider("Minimum confidence (for rules)", 0.01, 1.0, 0.3, step=0.01)
    min_lift = st.slider("Minimum lift (for rules filter)", 0.0, 10.0, 1.0, step=0.1)
    max_len = st.slider("Max length of itemsets", 1, 5, 3)
    run_analysis = st.button("Run analysis")

st.write("---")

if uploaded is None:
    st.info("Upload a CSV to begin. See sidebar for parameters.")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read file as CSV: {e}")
    st.stop()

st.write("### Preview of uploaded data")
st.dataframe(df.head())

fmt = detect_format(df)
st.write(f"**Detected format:** `{fmt}`")

# Attempt to normalize column names
cols_low = {c.lower(): c for c in df.columns}

if fmt == 'row':
    # heuristics to find transaction and product column names
    tr_col = None
    prod_col = None
    for k in cols_low:
        if 'transaction' in k or 'transactionid' in k or 'basket' in k or 'order' in k:
            tr_col = cols_low[k]
        if 'product' in k or 'item' in k or 'sku' in k or 'description' in k:
            prod_col = cols_low[k]
    # fallback
    if tr_col is None:
        tr_col = df.columns[0]
    if prod_col is None:
        prod_col = df.columns[1] if df.shape[1] > 1 else df.columns[0]

    st.write(f"Using Transaction column: **{tr_col}**, Product column: **{prod_col}**")
    baskets_series = rows_to_baskets(df, tr_col, prod_col)
    baskets = baskets_series.tolist()
else:
    # Basket style
    if df.shape[1] == 1:
        # single column with comma-separated items
        col = df.columns[0]
        st.write("Interpreting single-column basket CSV (comma-separated items).")
        baskets = df[col].fillna("").astype(str).apply(lambda s: [x.strip() for x in s.split(",") if x.strip()!='']).tolist()
    else:
        # multiple item columns maybe with NaNs
        st.write("Interpreting multi-column basket CSV (each column an item).")
        baskets = df.fillna('').astype(str).values.tolist()
        # convert each row list to only non-empty item strings
        baskets = [[x.strip() for x in row if x.strip()!=''] for row in baskets]

st.write(f"Number of transactions (baskets): **{len(baskets)}**")
if len(baskets) == 0:
    st.error("No baskets detected. Check your file format.")
    st.stop()

# Quick data check
item_freq_df = compute_item_frequencies(baskets)
st.write("Top 10 items by frequency:")
st.dataframe(item_freq_df.head(10).reset_index(drop=True))

# One-hot encode
onehot = basket_list_to_onehot(baskets)

# Run analysis after press
if run_analysis:
    st.header("Analysis results")

    # Frequent itemsets
    st.subheader("Frequent Itemsets (Apriori)")
    try:
        frequent_itemsets = apriori(onehot, min_support=min_support, use_colnames=True, max_len=max_len)
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        frequent_itemsets = frequent_itemsets.sort_values(by=['support','length'], ascending=[False, False])
        st.dataframe(frequent_itemsets.head(50).reset_index(drop=True))
    except Exception as e:
        st.error(f"Apriori failed: {e}")
        st.stop()

    # Association rules
    st.subheader("Association Rules")
    try:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        # filter by lift
        rules = rules[rules['lift'] >= min_lift]
        # make antecedents, consequents printable
        rules = rules[['antecedents','consequents','support','confidence','lift','leverage','conviction']]
        # convert frozensets to lists for better display
        rules['antecedents'] = rules['antecedents'].apply(lambda x: sorted(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: sorted(list(x)))
        st.write(f"Found {len(rules)} rules (after min_confidence={min_confidence} and min_lift={min_lift})")
        st.dataframe(rules.reset_index(drop=True).head(100))
    except Exception as e:
        st.error(f"Rule generation failed: {e}")
        rules = pd.DataFrame()

    # Visual: top items bar chart
    st.subheader("Item frequency (top 20)")
    fig1 = px.bar(item_freq_df.head(20), x='item', y='count', title="Top items by count")
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)

    # Co-occurrence heatmap
    st.subheader("Item co-occurrence heatmap (top 20 items)")
    top_items = item_freq_df.head(20)['item'].tolist()
    co = cooccurrence_matrix(onehot[top_items])
    # Normalize co-occurrence to show relative co-occurrence / support
    fig2, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(co, annot=False, cmap="YlGnBu", ax=ax)
    ax.set_title("Co-occurrence matrix (counts)")
    st.pyplot(fig2)

    # Network
    st.subheader("Rules network (top rules)")
    if not rules.empty:
        net_fig = plot_network_from_rules(rules, top_n=40)
        st.plotly_chart(net_fig, use_container_width=True)
    else:
        st.info("No rules to display in network (change thresholds).")

    # Allow user to download rules CSV
    if not rules.empty:
        csv = rules.to_csv(index=False)
        st.download_button("Download rules as CSV", csv, file_name="association_rules.csv", mime="text/csv")

    # Simple AOV uplift estimator
    st.subheader("Simple AOV uplift estimator (simulation)")
    st.markdown("""This simulates potential uplift by assuming:
- If a rule fires (antecedent subset of basket), the consequent is added with probability = rule confidence.
- If you have item prices (column named like 'price' or a separate upload), provide them to estimate dollar uplift.
**This is a simulation — real uplift requires an A/B test.**""")
    # Try to detect price column in original df if row format
    price_lookup = None
    if fmt == 'row' and prod_col in df.columns:
        # If original df had a price column, build lookup by average price per product
        price_cols = [c for c in df.columns if 'price' in c.lower() or 'unit_price' in c.lower() or 'amount' in c.lower()]
        if price_cols:
            chosen_price_col = price_cols[0]
            st.write(f"Detected price column `{chosen_price_col}` and will compute average prices per product for uplift estimate.")
            price_lookup_df = df[[prod_col, chosen_price_col]].dropna()
            price_lookup = price_lookup_df.groupby(prod_col)[chosen_price_col].mean().to_dict()
    # Allow manual price CSV upload as optional
    price_upload = st.file_uploader("Optional: upload CSV with item prices (columns: Product, Price)", type=["csv"], key="price")
    if price_upload is not None:
        try:
            price_df = pd.read_csv(price_upload)
            # find product & price columns
            pl_cols = [c for c in price_df.columns if 'product' in c.lower() or 'item' in c.lower() or 'sku' in c.lower()]
            pr_cols = [c for c in price_df.columns if 'price' in c.lower() or 'amount' in c.lower() or 'unit_price' in c.lower()]
            if pl_cols and pr_cols:
                price_lookup = price_df.set_index(pl_cols[0])[pr_cols[0]].to_dict()
                st.write(f"Using {pl_cols[0]} / {pr_cols[0]} from uploaded price file.")
            else:
                st.warning("Price file detected but couldn't find product/price columns. Expected names like 'Product' and 'Price'.")
        except Exception as e:
            st.warning(f"Could not read price file: {e}")

    if rules.empty:
        st.info("No rules to estimate uplift from. Try lowering thresholds.")
    else:
        res = estimate_aov_uplift(baskets, rules, price_lookup=price_lookup, min_confidence=min_confidence)
        if res is None:
            st.warning("No rules passed the chosen min_confidence; cannot estimate uplift.")
        else:
            st.metric("Baseline AOV", f"{res['baseline_aov']:.2f}" + (" items" if price_lookup is None else " (currency)"))
            st.metric("Estimated AOV after cross-sell (sim)", f"{res['new_aov']:.2f}")
            if res['pct_uplift'] is not None:
                st.success(f"Estimated AOV uplift: {res['pct_uplift']:.2f}% (simulation)")

    st.write("---")
    st.markdown("**Notes & next steps:**")
    st.markdown("""
    - This app does automated analysis; consider the results exploratory. For production cross-sell you should:
      1. Validate top rules manually and check business sense.
      2. A/B test recommended cross-sell placements to measure real uplift.
      3. Consider seasonality, promotions, and product availability.
    - You can extend the app with:
      - FP-Growth (faster on huge datasets)
      - Time-windowed rules (recent purchases stronger signal)
      - Customer segmentation & personalized rules
      - Integration with a recommender system (collaborative filtering)
    """)
    st.balloons()
else:
    st.info("Change parameters in sidebar and press *Run analysis* to start.")
