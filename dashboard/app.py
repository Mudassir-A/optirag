import pandas as pd
import streamlit as st

from optirag.config import settings
from optirag.request_log import RequestStore

st.set_page_config(page_title="OptiRAG Monitor", layout="wide")
st.title("OptiRAG Monitor")

df = RequestStore(settings.db_path).read()
if df.empty:
    st.info("No requests logged yet. Call the API or run `make eval`.")
    st.stop()

runs = sorted(df.run_id.unique())
sel = st.sidebar.multiselect("Runs", runs, default=runs)
window = st.sidebar.slider("Rolling window (requests)", 5, 200, 25)
if st.sidebar.button("Refresh"):
    st.rerun()

d = df[df.run_id.isin(sel)].reset_index(drop=True)
if d.empty:
    st.stop()
d["seq"] = d.groupby("run_id").cumcount()
d["cache_hit_f"] = d.cache_hit.astype(float)

k = st.columns(6)
k[0].metric("Requests", len(d))
k[1].metric("Cost / request", f"${d.cost_usd.mean():.5f}")
k[2].metric("Total cost", f"${d.cost_usd.sum():.4f}")
k[3].metric("Cache hit rate", f"{d.cache_hit.mean():.1%}")
k[4].metric("p50 / p95 (ms)", f"{d.latency_ms.quantile(.5):.0f} / {d.latency_ms.quantile(.95):.0f}")
k[5].metric("Avg quality", "n/a" if d.quality.isna().all() else f"{d.quality.mean():.2f}")

st.subheader("By run")
st.dataframe(
    d.groupby("run_id").agg(
        requests=("id", "count"),
        cost_per_request=("cost_usd", "mean"),
        total_cost=("cost_usd", "sum"),
        cache_hit_rate=("cache_hit", "mean"),
        escalation_rate=("escalated", "mean"),
        p50_ms=("latency_ms", lambda s: s.quantile(.5)),
        p95_ms=("latency_ms", lambda s: s.quantile(.95)),
        quality=("quality", "mean"),
    ),
    use_container_width=True,
)


def rolling(col: str) -> pd.DataFrame:
    return d.pivot(index="seq", columns="run_id", values=col).rolling(window, min_periods=1).mean()


c1, c2 = st.columns(2)
c1.subheader("Cost per request (rolling)")
c1.line_chart(rolling("cost_usd"))
c2.subheader("Cache hit rate (rolling)")
c2.line_chart(rolling("cache_hit_f"))

c3, c4 = st.columns(2)
c3.subheader("Route split")
c3.bar_chart(d.groupby(["run_id", "route"]).size().unstack(fill_value=0))
c4.subheader("Latency by route (ms)")
c4.dataframe(
    d.groupby("route").latency_ms.agg(
        count="count", p50=lambda s: s.quantile(.5), p95=lambda s: s.quantile(.95)
    ),
    use_container_width=True,
)

c5, c6 = st.columns(2)
q = d.dropna(subset=["quality"]).reset_index(drop=True)
c5.subheader("Quality by route")
c6.subheader("Quality over time by route (rolling)")
if q.empty:
    c5.info("No quality scores yet.")
else:
    c5.bar_chart(q.groupby("route").quality.mean())
    c6.line_chart(
        q.pivot_table(index=q.index, columns="route", values="quality").rolling(window, min_periods=1).mean()
    )
