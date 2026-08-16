#!/usr/bin/env python3
"""
04_build_dashboard.py
---------------------
Regenerates the self-contained HTML dashboard from articles_v3_enriched.csv.
Same layout as the v2 dashboard (SVG choropleth + micro-state dots, stance
timeline, keyword bars, outlet bars, type donut, full country table), rebuilt
from the corrected v3 data.

Everything is embedded — no CDN, no external libraries, opens offline.

Requires world_paths.json and microstate_dots.json (the map geometry). If they
are not present next to this script, it regenerates them from Natural Earth
(needs geopandas + internet, one time only).

USAGE
    pip install pandas
    python 04_build_dashboard.py
    -> metoo_dashboard_v3.html
"""

# ---------------------------------------------------------------------------
# Repository paths. Added when the project folder was reorganised (Aug 2026).
# Scripts live in scripts/ and resolve every input and output from the repo
# root, so they run from anywhere:  python scripts/17_lda_topics.py
# ---------------------------------------------------------------------------
from pathlib import Path as _Path
ROOT       = _Path(__file__).resolve().parents[1]
CODEBOOK   = ROOT / "codebook"
GEO        = ROOT / "assets" / "geo"
RAW        = ROOT / "data" / "raw"
INTERIM    = ROOT / "data" / "interim"
PROCESSED  = ROOT / "data" / "processed"
FIGURES    = ROOT / "outputs" / "figures"
REPORTS    = ROOT / "outputs" / "reports"
TABLES     = ROOT / "outputs" / "tables"
DASHBOARDS = ROOT / "outputs" / "dashboards"
for _d in (INTERIM, PROCESSED, FIGURES, REPORTS, TABLES, DASHBOARDS, GEO):
    _d.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------------------

import json
import os
import re
import sys
from collections import Counter

import pandas as pd

IN_FILE = INTERIM / "articles_v3_enriched.csv"
OUT_FILE = DASHBOARDS / "metoo_dashboard_v3.html"
PATHS_FILE = GEO / "world_paths.json"
DOTS_FILE = GEO / "microstate_dots.json"

# Natural Earth name -> GDELT source_country name (the ones that differ)
NAME_MAP = {
    "United States of America": "United States", "Czechia": "Czech Republic",
    "Slovakia": "Slovak Republic", "North Macedonia": "Macedonia",
    "S. Sudan": "South Sudan", "Dem. Rep. Congo": "Democratic Republic of the Congo",
    "Central African Rep.": "Central African Republic",
    "Bosnia and Herz.": "Bosnia-Herzegovina", "Côte d'Ivoire": "Ivory Coast",
    "Dominican Rep.": "Dominican Republic", "Solomon Is.": "Solomon Islands",
    "Eq. Guinea": "Equatorial Guinea", "Timor-Leste": "East Timor",
    "Republic of the Congo": "Congo",
}
MICRO = {
    "Singapore": [103.8, 1.35], "Hong Kong": [114.2, 22.3], "Malta": [14.4, 35.9],
    "Mauritius": [57.5, -20.3], "Tuvalu": [179.2, -8.5], "Niue": [-169.9, -19.0],
    "Bahrain": [50.6, 26.0], "Andorra": [1.5, 42.5], "Monaco": [7.4, 43.7],
    "Samoa": [-172.1, -13.8], "Reunion": [55.5, -21.1], "Cape Verde": [-23.6, 15.1],
    "Comoros": [43.3, -11.6], "Seychelles": [55.5, -4.6], "Barbados": [-59.5, 13.2],
    "Grenada": [-61.7, 12.1], "Saint Lucia": [-61.0, 13.9], "Bermuda": [-64.8, 32.3],
    "Guam": [144.8, 13.4], "Macau": [113.5, 22.2], "Gibraltar": [-5.4, 36.1],
    "Jersey": [-2.1, 49.2], "Martinique": [-61.0, 14.6], "Mayotte": [45.2, -12.8],
    "French Polynesia": [-149.4, -17.7], "Cook Islands": [-159.8, -21.2],
    "Anguilla": [-63.1, 18.2], "Antigua and Barbuda": [-61.8, 17.1],
    "Virgin Islands": [-64.8, 18.3], "San Marino": [12.5, 43.9],
    "British Indian Ocean Territory": [72.4, -7.3],
    "Saint Vincent and the Grenadines": [-61.2, 13.2],
}
W, H = 960, 440


def xy(lon, lat):
    return int(round((lon + 180) / 360 * W)), int(round((90 - lat) / 180 * H))


def ensure_geometry():
    """Build world_paths.json + microstate_dots.json once, from Natural Earth."""
    if os.path.exists(PATHS_FILE) and os.path.exists(DOTS_FILE):
        return
    print("map geometry not found; generating from Natural Earth (one time)...")
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("need geopandas for first run: pip install geopandas")
    import urllib.request
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_110m_admin_0_countries.geojson")
    urllib.request.urlretrieve(url, "/tmp/ne.geojson")
    gdf = gpd.read_file("/tmp/ne.geojson").to_crs(epsg=4326)

    def path(geom, min_area=0.05):
        polys = ([geom] if geom.geom_type == "Polygon"
                 else list(geom.geoms) if geom.geom_type == "MultiPolygon" else [])
        out = []
        for p in polys:
            if p.area < min_area:
                continue
            c = list(p.exterior.coords)
            if len(c) > 100:
                c = c[::2] + [c[-1]]
            pts = [xy(a, b) for a, b in c]
            d = [pts[0]]
            for q in pts[1:]:
                if q != d[-1]:
                    d.append(q)
            if len(d) < 3:
                continue
            out.append("M" + " ".join(f"{a},{b}" for a, b in d) + "Z")
        return " ".join(out)

    feats = []
    for _, r in gdf.iterrows():
        nm = NAME_MAP.get(r["NAME"], r["NAME"])
        p = path(r.geometry)
        if p:
            feats.append({"id": nm, "p": p})
    json.dump(feats, open(PATHS_FILE, "w"), ensure_ascii=False, separators=(",", ":"))
    dots = [{"id": k, "x": xy(v[0], v[1])[0], "y": xy(v[0], v[1])[1]}
            for k, v in MICRO.items()]
    json.dump(dots, open(DOTS_FILE, "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"  wrote {len(feats)} country paths, {len(dots)} dots")


def build_data(df):
    df = df.copy()
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    df["month"] = df["published"].dt.strftime("%Y-%m")
    has_country = "source_country" in df.columns
    if not has_country:
        df["source_country"] = "UNKNOWN"

    cg = df.groupby("source_country").agg(
        total=("url", "count"),
        support=("article_stance", lambda s: (s == "Support").sum()),
        backlash=("article_stance", lambda s: (s == "Backlash").sum()),
        mixed=("article_stance", lambda s: (s == "Mixed").sum()),
        opinion=("article_type", lambda s: (s == "Opinion/Editorial").sum()),
        hash_n=("has_hashtag_term", "sum"),
    ).reset_index()
    cg["backlash_pct"] = (cg["backlash"] / cg["total"] * 100).round(2)
    cg["hash_pct"] = (cg["hash_n"] / cg["total"] * 100).round(1)
    by_country = cg[cg["source_country"] != "UNKNOWN"].sort_values("total", ascending=False)

    mt = (df[df["month"].notna()].groupby(["month", "article_stance"])
          .size().unstack(fill_value=0).reset_index().sort_values("month"))

    kwc = Counter()
    for s in df["keywords_matched"].dropna():
        for k in set(x.strip() for x in str(s).split(" | ")):
            kwc[k] += 1
    stance_lookup = {}
    for s in df["keyword_stances"].dropna().head(50000):
        for part in str(s).split(" | "):
            m = re.match(r"^(.*) \[(.+)\]$", part.strip())
            if m and m.group(1) not in stance_lookup:
                stance_lookup[m.group(1)] = m.group(2)

    return {
        "summary": {
            "total": int(len(df)), "outlets": int(df["outlet"].nunique()),
            "date_min": str(df["published"].min())[:10],
            "date_max": str(df["published"].max())[:10],
            "unknown_country": int((df["source_country"] == "UNKNOWN").sum()),
            "hash_n": int(df["has_hashtag_term"].sum()),
            "stance": df["article_stance"].value_counts().to_dict(),
            "type": df["article_type"].value_counts().to_dict(),
        },
        "by_country": by_country.to_dict("records"),
        "monthly": mt.to_dict("records"),
        "top_keywords": [{"kw": k, "n": n, "stance": stance_lookup.get(k, "?")}
                         for k, n in kwc.most_common(25)],
        "top_outlets": [{"outlet": o, "n": int(n)} for o, n in
                        df.groupby("outlet").size().sort_values(ascending=False).head(15).items()],
    }


def main():
    ensure_geometry()
    df = pd.read_csv(IN_FILE)
    d = build_data(df)
    wp = open(PATHS_FILE).read()
    dots = open(DOTS_FILE).read()

    tmpl = open(os.path.join(os.path.dirname(__file__), "dashboard_template.html")).read() \
        if os.path.exists(os.path.join(os.path.dirname(__file__), "dashboard_template.html")) \
        else DASHBOARD_TEMPLATE
    html = (tmpl
            .replace("__WP__", wp)
            .replace("__DOTS__", dots)
            .replace("__DATA__", json.dumps(d, ensure_ascii=False, separators=(",", ":"))))
    open(OUT_FILE, "w").write(html)
    print(f"wrote {OUT_FILE} ({len(html)//1024} KB)")
    print(f"  {d['summary']['total']:,} articles, {len(d['by_country'])} countries")


# The template is stored inline so this script is self-contained. It reads three
# placeholders: __WP__ (country paths), __DOTS__ (micro-states), __DATA__ (stats).
DASHBOARD_TEMPLATE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>#MeToo Global Coverage v3</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#F0F0EC;color:#17171E;font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;line-height:1.45}
.wrap{max-width:1060px;margin:0 auto;padding:16px 16px 60px}
header{border-bottom:3px solid #17171E;padding-bottom:10px}
.eyebrow{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#A2196B;margin-bottom:6px;font-family:monospace}
h1{font-size:22px;font-weight:900;letter-spacing:-.02em;line-height:1.1}h1 span{color:#5A5A66;font-weight:400}
.kpis{display:grid;grid-template-columns:repeat(7,1fr);gap:1px;background:#DCDCD4;border:1px solid #DCDCD4;margin:14px 0}
.kpi{background:#FCFCFA;padding:11px 9px}.kpi b{display:block;font-size:20px;font-weight:900;letter-spacing:-.02em}
.kpi small{font-size:10px;color:#5A5A66;line-height:1.3;display:block}
.kpi.s b{color:#3E6B73}.kpi.bl b{color:#B5432A}.kpi.ac b{color:#A2196B}
.tabs{display:flex;margin-bottom:8px;border:1px solid #17171E;width:fit-content}
.tab{padding:5px 13px;cursor:pointer;font-size:11px;font-family:monospace;background:none;border:0;color:#17171E;border-right:1px solid #17171E}
.tab:last-child{border-right:0}.tab.on{background:#17171E;color:#F0F0EC}
.card{background:#FCFCFA;border:1px solid #DCDCD4;padding:10px}section{margin-top:18px}
h2{font-size:13px;font-weight:700;margin-bottom:2px}.sub{font-size:11px;color:#5A5A66;margin-bottom:8px;max-width:74ch}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
.note{border-left:3px solid #A2196B;padding:10px 14px;background:#FCFCFA;font-size:11px;color:#5A5A66;margin-top:16px;line-height:1.55}
.note b{color:#17171E}svg{display:block;width:100%}canvas{display:block}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:#17171E;color:#F0F0EC;padding:6px 8px;text-align:left;font-weight:600;font-size:10.5px;position:sticky;top:0;z-index:2}
td{padding:4px 8px;border-bottom:1px solid #E6E6E0}td.r{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:#EDEFEA!important}.tscroll{max-height:430px;overflow-y:auto;border:1px solid #DCDCD4}
#tip{position:fixed;background:#17171E;color:#F0F0EC;padding:7px 10px;font-size:11px;pointer-events:none;display:none;z-index:999;border-radius:2px;line-height:1.5}
footer{margin-top:24px;padding-top:12px;border-top:1px solid #DCDCD4;font-family:monospace;font-size:10.5px;color:#8A8A92}
</style></head><body><div class="wrap">
<header><div class="eyebrow">Corrected corpus (v3) · generic + viol + CJK fixes applied · fads-metoo</div>
<h1>#MeToo in the world's press <span>— 27 months, Oct 2017–Dec 2019</span></h1></header>
<div class="kpis" id="kpis"></div>
<section><h2>Where the coverage comes from</h2>
<p class="sub">Country = registered location of the publishing outlet (GDELT domain lookup), not the story's subject. Hover for detail.</p>
<div class="tabs" id="mapTabs">
<button class="tab on" onclick="setMode('total',this)">Article volume</button>
<button class="tab" onclick="setMode('bp',this)">Backlash share %</button>
<button class="tab" onclick="setMode('b',this)">Backlash count</button>
<button class="tab" onclick="setMode('hp',this)">Hashtag share %</button></div>
<div class="card" style="padding:6px"><svg id="mapSvg" viewBox="0 0 960 440" xmlns="http://www.w3.org/2000/svg">
<rect width="960" height="440" fill="#E2E4DF"/><g id="countries"></g><g id="dots"></g></svg>
<div style="font-size:10px;font-family:monospace;color:#5A5A66;padding:4px 2px" id="mapLegend"></div></div></section>
<section><h2>All publishing countries</h2>
<p class="sub">Sorted by volume. Hashtag % = share matching a #hashtag term rather than generic gender-violence vocabulary. Red rows = backlash share &gt;10%.</p>
<div class="tscroll"><table><thead><tr><th>#</th><th>Country</th><th class=r>Total</th><th class=r>Hashtag %</th>
<th class=r>Support</th><th class=r>Backlash</th><th class=r>Mixed</th><th class=r>Backlash %</th><th class=r>Opinion</th></tr></thead>
<tbody id="tbody"></tbody></table></div></section>
<section><h2>Coverage over time — monthly counts by stance</h2>
<p class="sub">Oct 2017 ignition; Oct 2018 Kavanaugh spike.</p>
<div class="card"><canvas id="timeC" width="1036" height="210"></canvas></div></section>
<div class="grid2"><div><h2>Top 25 matched keywords</h2><p class="sub">Distinct per article. Color = codebook stance.</p>
<div class="card"><canvas id="kwC" width="505" height="480"></canvas></div></div>
<div><h2>Top 15 outlets</h2><p class="sub">GDELT SourceCommonName.</p>
<div class="card"><canvas id="outC" width="505" height="300"></canvas></div><div style="height:12px"></div>
<h2>Article type (heuristic)</h2><p class="sub">URL path + title prefix, multilingual. Floor estimate.</p>
<div class="card"><canvas id="typeC" width="505" height="130"></canvas></div></div></div>
<div class="note"><b>Stance caveat.</b> Stance is the codebook label of the <em>keyword matched</em>, not the article's editorial position — a report <em>about</em> #HimToo counts as Backlash. High backlash % in small-n countries reflects a few articles matching specific backlash terms.</div>
<footer id="foot"></footer></div><div id="tip"></div>
<script>
const WP=__WP__, DOTS=__DOTS__, D=__DATA__;
const BC=D.by_country, MONTHLY=D.monthly, KWr=D.top_keywords, OUT=D.top_outlets, S=D.summary;
const KW=[];{const seen=new Set();for(const k of KWr){if(!seen.has(k.kw)){seen.add(k.kw);KW.push(k);}}}
const byName={};BC.forEach(d=>byName[d.source_country]=d);
const st=S.stance,ty=S.type;
document.getElementById('kpis').innerHTML=`
<div class="kpi"><b>${S.total.toLocaleString()}</b><small>articles matched</small></div>
<div class="kpi ac"><b>${BC.length}</b><small>publishing countries</small></div>
<div class="kpi"><b>${S.outlets.toLocaleString()}</b><small>distinct outlets</small></div>
<div class="kpi s"><b>${(st.Support||0).toLocaleString()}</b><small>support-keyword</small></div>
<div class="kpi bl"><b>${(st.Backlash||0).toLocaleString()}</b><small>backlash-keyword</small></div>
<div class="kpi"><b>${((S.hash_n/S.total)*100).toFixed(1)}%</b><small>matched a hashtag</small></div>
<div class="kpi"><b>${(ty['Opinion/Editorial']||0).toLocaleString()}</b><small>opinion/editorial</small></div>`;
document.getElementById('foot').textContent=`FADS · #MeToo v3 corrected corpus · ${S.date_min} to ${S.date_max} · generated ${new Date().toISOString().slice(0,10)}`;
let tb='';BC.forEach((c,i)=>{const hp=c.hash_pct,hb=hp<25?'#B5432A':hp<50?'#C9A227':'#3E6B73',bg=c.backlash_pct>10?'#FFF5F3':'#FCFCFA';
tb+=`<tr style="background:${bg}"><td>${i+1}</td><td><b>${c.source_country}</b></td><td class=r>${c.total.toLocaleString()}</td>
<td class=r style="color:${hb}"><b>${hp}%</b></td><td class=r style="color:#3E6B73">${c.support.toLocaleString()}</td>
<td class=r style="color:#B5432A">${c.backlash.toLocaleString()}</td><td class=r>${c.mixed}</td>
<td class=r><b style="color:${c.backlash_pct>10?'#B5432A':'#17171E'}">${c.backlash_pct}%</b></td><td class=r>${c.opinion.toLocaleString()}</td></tr>`;});
document.getElementById('tbody').innerHTML=tb;
let mode='total';const MAXT=Math.max(...BC.map(d=>d.total),1),MAXB=Math.max(...BC.map(d=>d.backlash),1);
function hx(h){return[parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function ip(a,b,t){const x=hx(a),y=hx(b);return'#'+[0,1,2].map(i=>Math.round(x[i]+(y[i]-x[i])*t).toString(16).padStart(2,'0')).join('');}
function col(d){if(!d)return'#CFD1CB';if(mode==='total')return ip('#CBD3C8','#17171E',Math.log10(d.total+1)/Math.log10(MAXT+1));
if(mode==='bp')return ip('#DEEBDE','#B5432A',Math.min(d.backlash_pct/100,1));
if(mode==='hp')return ip('#B5432A','#3E6B73',Math.min(d.hash_pct/100,1));return ip('#DEEBDE','#8F2F1C',Math.min(d.backlash/MAXB,1));}
const tip=document.getElementById('tip');
function tipHTML(id,d){return d?`<b>${id}</b><br>${d.total.toLocaleString()} articles · ${d.hash_pct}% hashtag<br>backlash ${d.backlash} (${d.backlash_pct}%)`:`<b>${id}</b><br>no articles`;}
const gC=document.getElementById('countries'),gD=document.getElementById('dots'),els={};
function wire(el,id){const d=byName[id];el.addEventListener('mouseenter',()=>{el.setAttribute('stroke','#17171E');el.setAttribute('stroke-width','1.3');tip.innerHTML=tipHTML(id,d);tip.style.display='block';});
el.addEventListener('mousemove',e=>{tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-10)+'px';});
el.addEventListener('mouseleave',()=>{el.setAttribute('stroke','#F0F0EC');el.setAttribute('stroke-width',el.tagName==='circle'?'0.8':'0.4');tip.style.display='none';});}
WP.forEach(f=>{const e=document.createElementNS('http://www.w3.org/2000/svg','path');e.setAttribute('d',f.p);
e.setAttribute('fill',col(byName[f.id]));e.setAttribute('stroke','#F0F0EC');e.setAttribute('stroke-width','0.4');gC.appendChild(e);els[f.id]=e;wire(e,f.id);});
DOTS.forEach(o=>{if(!byName[o.id])return;const e=document.createElementNS('http://www.w3.org/2000/svg','circle');
e.setAttribute('cx',o.x);e.setAttribute('cy',o.y);e.setAttribute('r',3.2);e.setAttribute('fill',col(byName[o.id]));
e.setAttribute('stroke','#F0F0EC');e.setAttribute('stroke-width','0.8');gD.appendChild(e);els[o.id]=e;wire(e,o.id);});
const LEG={total:'Fill = article volume, log scale · dots = micro-states',bp:'Fill = backlash share %',b:'Fill = backlash count',hp:'Fill = hashtag share % (red=generic vocab, teal=hashtag)'};
function setMode(m,btn){mode=m;document.querySelectorAll('#mapTabs .tab').forEach(b=>b.classList.remove('on'));btn.classList.add('on');
Object.keys(els).forEach(id=>els[id].setAttribute('fill',col(byName[id])));document.getElementById('mapLegend').textContent=LEG[m];}
document.getElementById('mapLegend').textContent=LEG.total;
function drawTimeline(){const c=document.getElementById('timeC'),x=c.getContext('2d'),W=c.width,H=c.height,PL=44,PR=10,PT=16,PB=30,cW=W-PL-PR,cH=H-PT-PB;
x.clearRect(0,0,W,H);const mx=Math.max(...MONTHLY.map(m=>(m.Support||0)+(m.Backlash||0)+(m.Mixed||0)),1),N=MONTHLY.length,bw=cW/N;
x.strokeStyle='#DCDCD4';x.lineWidth=1;const step=Math.pow(10,Math.floor(Math.log10(mx)));
for(let v=0;v<=mx;v+=step){const y=PT+cH-v/mx*cH;x.beginPath();x.moveTo(PL,y);x.lineTo(PL+cW,y);x.stroke();
x.fillStyle='#5A5A66';x.font='9px monospace';x.textAlign='right';x.fillText(v>=1000?(v/1000)+'k':v,PL-3,y+3);}
MONTHLY.forEach((m,i)=>{let yb=PT+cH;const px=PL+i*bw+1,w=bw-2;
[[m.Support||0,'rgba(62,107,115,.82)'],[m.Mixed||0,'rgba(201,162,39,.9)'],[m.Backlash||0,'rgba(181,67,42,.9)']].forEach(([v,cc])=>{const h=v/mx*cH;yb-=h;x.fillStyle=cc;x.fillRect(px,yb,w,h);});});
x.fillStyle='#5A5A66';x.font='9px monospace';x.textAlign='center';MONTHLY.forEach((m,i)=>{if(i%3===0)x.fillText(m.month,PL+i*bw+bw/2,H-4);});
let lx=PL;x.textAlign='left';[['Support','rgba(62,107,115,.82)'],['Mixed','rgba(201,162,39,.9)'],['Backlash','rgba(181,67,42,.9)']].forEach(([l,cc])=>{x.fillStyle=cc;x.fillRect(lx,3,10,8);x.fillStyle='#17171E';x.fillText(l,lx+13,11);lx+=58;});}
function hbar(id,items,lab,val,cf,PL2){const c=document.getElementById(id),x=c.getContext('2d'),W=c.width,H=c.height,PR=52,PT=6,PB=6,cW=W-PL2-PR,cH=H-PT-PB;
x.clearRect(0,0,W,H);const N=items.length,bh=Math.floor(cH/N)-2,mx=val(items[0]);
items.forEach((o,i)=>{const y=PT+i*(bh+2),w=val(o)/mx*cW;x.fillStyle=cf(o);x.fillRect(PL2,y,w,bh);
x.fillStyle='#17171E';x.font='10.5px system-ui';x.textAlign='right';x.fillText(lab(o),PL2-4,y+bh-2);
x.fillStyle='#5A5A66';x.font='9px monospace';x.textAlign='left';const v=val(o);x.fillText(v>=1000?(v/1000).toFixed(0)+'k':v,PL2+w+3,y+bh-2);});}
drawTimeline();
hbar('kwC',KW,o=>o.kw,o=>o.n,o=>o.stance==='Backlash'?'rgba(181,67,42,.85)':o.stance==='Mixed'?'rgba(201,162,39,.85)':'rgba(62,107,115,.82)',140);
hbar('outC',OUT,o=>o.outlet,o=>o.n,()=>'rgba(23,23,30,.8)',148);
(function(){const c=document.getElementById('typeC'),x=c.getContext('2d'),W=c.width,H=c.height;x.clearRect(0,0,W,H);
const sl=[['News/Reporting',ty['News/Reporting']||0,'rgba(23,23,30,.75)'],['Opinion/Editorial',ty['Opinion/Editorial']||0,'#A2196B'],['Analysis',ty['Analysis']||0,'#C9A227']];
const t=sl.reduce((a,s)=>a+s[1],0)||1;let a=-Math.PI/2;
sl.forEach(s=>{const sw=s[1]/t*Math.PI*2;x.beginPath();x.moveTo(65,H/2);x.arc(65,H/2,46,a,a+sw);x.closePath();x.fillStyle=s[2];x.fill();a+=sw;});
let ly=22;x.textAlign='left';sl.forEach(s=>{x.fillStyle=s[2];x.fillRect(120,ly-9,10,10);x.fillStyle='#17171E';x.font='11px system-ui';x.fillText(s[0]+': '+(s[1]/t*100).toFixed(1)+'%',134,ly);ly+=22;});})();
</script></body></html>"""


if __name__ == "__main__":
    main()
