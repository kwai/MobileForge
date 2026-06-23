"""
交互式 HTML 仪表板报告生成器

生成自包含的 HTML 文件，内置 Chart.js 图表和客户端筛选逻辑。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """生成交互式 HTML 分析仪表板"""

    def __init__(self, output_dir: str):
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path(output_dir) / f"analysis_{self.ts}"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        tasks: Dict[str, Dict],
        baseline: Dict,
        per_app: Dict,
        filter_sections: List[Dict],
        cli_args: str = "",
    ) -> str:
        """
        生成交互式 HTML 报告。

        Returns:
            生成的 HTML 文件路径
        """
        lite = self._build_lite_data(tasks)
        apps = sorted(set(t.get("app", "Unknown") for t in tasks.values()))

        html = self._render_html(
            lite_data=lite,
            baseline=baseline,
            per_app=per_app,
            filter_sections=filter_sections,
            apps=apps,
            cli_args=cli_args,
        )

        path = self.report_dir / "dashboard.html"
        path.write_text(html, encoding="utf-8")
        logger.info(f"仪表板已保存: {path}")
        return str(path)

    def save_json(self, data: Dict):
        """保存原始分析数据为 JSON"""
        p = self.report_dir / "analysis_data.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_sanitize(data), f, indent=2, ensure_ascii=False)
        logger.info(f"JSON 数据已保存: {p}")

    # ──────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────

    def _build_lite_data(self, tasks: Dict[str, Dict]) -> List[Dict]:
        """将完整任务数据转为轻量格式 (嵌入 HTML 用)"""
        result = []
        for tid, t in tasks.items():
            task_lite = {
                "id": tid,
                "app": t.get("app", "Unknown"),
                "desc": t.get("task_description", "")[:100],
                "sr": round(t.get("avg_sr", 0), 4),
                "attempts": [],
            }
            for aid, a in t["attempts"].items():
                task_lite["attempts"].append({
                    "id": aid,
                    "success": a.get("overall_success", False),
                    "result": a.get("final_result", -1),
                    "feasible": a.get("task_feasible"),
                    "mc": a.get("max_consecutive_same_actions", 0),
                    "hGen": a.get("has_eval_hint", False),
                    "hInp": a.get("has_hints_input", False),
                    "steps": [
                        {
                            "n": s["step_number"],
                            "ok": s.get("step_success", True),
                            "imp": s.get("impact", "unknown"),
                            "reas": s.get("reasonableness", "unknown"),
                        }
                        for s in a.get("steps", [])
                    ],
                })
            result.append(task_lite)
        return result

    def _render_html(self, *, lite_data, baseline, per_app, filter_sections, apps, cli_args):
        ts_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data_json = json.dumps(lite_data, ensure_ascii=False)
        baseline_json = json.dumps(_sanitize(baseline), ensure_ascii=False)
        per_app_json = json.dumps(_sanitize(per_app), ensure_ascii=False)
        filter_json = json.dumps(
            [{"name": f["name"], "desc": f["desc"],
              "metrics": _sanitize(f["metrics"]), "examples": f.get("examples", [])[:8]}
             for f in filter_sections],
            ensure_ascii=False,
        )
        app_options = "\n".join(f'<option value="{a}">{a}</option>' for a in apps)

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MobileForge 数据分析仪表板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>{_CSS}</style>
</head>
<body>

<!-- ═══════ Header ═══════ -->
<header class="header">
  <div>
    <h1>📊 MobileForge 数据分析仪表板</h1>
    <div class="meta">{ts_display} &nbsp;|&nbsp; {cli_args[:120]}</div>
  </div>
  <div style="display:flex;gap:12px;align-items:center">
    <label style="color:#ffffffcc;font-size:13px">App 筛选:</label>
    <select class="app-select" id="app-select" onchange="onAppChange(this.value)">
      <option value="__all__">全部 App</option>
      {app_options}
    </select>
  </div>
</header>

<!-- ═══════ Tabs ═══════ -->
<nav class="tabs">
  <div class="tab active" data-tab="overview">📋 概览</div>
  <div class="tab" data-tab="apps">📱 App 分析</div>
  <div class="tab" data-tab="filters">🔧 筛选实验</div>
  <div class="tab" data-tab="precomputed">📑 预计算分析</div>
  <div class="tab" data-tab="table">📝 数据表</div>
</nav>

<main class="content">

<!-- ═══════ Tab: 概览 ═══════ -->
<section id="overview" class="tab-content active">
  <div class="kpi-row">
    <div class="kpi-card"><div class="label">任务数</div><div class="value" id="k-tasks">-</div></div>
    <div class="kpi-card c2"><div class="label">轨迹数</div><div class="value" id="k-atts">-</div></div>
    <div class="kpi-card c3"><div class="label">步骤数</div><div class="value" id="k-steps">-</div></div>
    <div class="kpi-card c4"><div class="label">平均 SR</div><div class="value" id="k-sr">-</div></div>
    <div class="kpi-card c1"><div class="label">轨迹成功率</div><div class="value" id="k-succ">-</div></div>
    <div class="kpi-card"><div class="label">Pass@1</div><div class="value" id="k-pass1">-</div></div>
  </div>
  <div class="kpi-row" style="margin-bottom:8px">
    <div class="kpi-card sm"><div class="label">Infeasible 任务</div><div class="value" id="k-inf">-</div></div>
    <div class="kpi-card sm"><div class="label">死循环轨迹(k≥3)</div><div class="value" id="k-loop">-</div></div>
    <div class="kpi-card sm"><div class="label">平均 steps/轨迹</div><div class="value" id="k-spa">-</div></div>
    <div class="kpi-card sm"><div class="label">Positive steps%</div><div class="value" id="k-pos">-</div></div>
    <div class="kpi-card sm"><div class="label">生成 Hint</div><div class="value" id="k-hgen">-</div><div class="sub" id="k-hgen-sub"></div></div>
    <div class="kpi-card sm"><div class="label">使用 Hint</div><div class="value" id="k-hinp">-</div><div class="sub" id="k-hinp-sub"></div></div>
  </div>
  <div class="chart-grid">
    <div class="chart-card"><h3>步骤 Impact 分布</h3><canvas id="c-impact"></canvas></div>
    <div class="chart-card"><h3>轨迹成功/失败</h3><canvas id="c-success"></canvas></div>
    <div class="chart-card"><h3>任务 SR 分布</h3><canvas id="c-sr"></canvas></div>
    <div class="chart-card"><h3>步骤合理性分布</h3><canvas id="c-reas"></canvas></div>
  </div>
  <div class="chart-grid">
    <div class="chart-card"><h3>Action 类型 Top-10</h3><canvas id="c-actions"></canvas></div>
    <div class="chart-card"><h3>Infeasible / Feasible / 未知</h3><canvas id="c-feas"></canvas></div>
  </div>
  <div class="chart-grid">
    <div class="chart-card"><h3>Pass@k 曲线 (整体)</h3><canvas id="c-passk"></canvas></div>
  </div>
  <div class="chart-grid">
    <div class="chart-card"><h3>Eval Hint 生成与使用情况</h3><canvas id="c-hint"></canvas></div>
    <div class="chart-card"><h3>各 Attempt 序号的 Hint 覆盖率</h3><canvas id="c-hint-att"></canvas></div>
  </div>
</section>

<!-- ═══════ Tab: App 分析 ═══════ -->
<section id="apps" class="tab-content">
  <div class="chart-grid" style="margin-bottom:20px">
    <div class="chart-card"><h3>各 App 成功率对比</h3><canvas id="c-app-bar"></canvas></div>
    <div class="chart-card"><h3>各 App Pass@k 曲线</h3><canvas id="c-app-passk"></canvas></div>
  </div>
  <div class="card"><h3 style="margin-bottom:12px">各 App 指标明细</h3>
    <table><thead><tr>
      <th>App</th><th>任务</th><th>轨迹</th><th>步骤</th>
      <th>成功率</th><th>SR</th><th>Pass@1</th><th>Pass@K(max)</th><th>Positive%</th><th>Negative%</th><th>死循环</th><th>Hint生成</th><th>Hint使用</th>
    </tr></thead><tbody id="app-tbody"></tbody></table>
  </div>
</section>

<!-- ═══════ Tab: 筛选实验 ═══════ -->
<section id="filters" class="tab-content">
  <div class="filter-panel">
    <div class="filter-controls">
      <h3>筛选控制面板</h3>
      <div class="fg"><label><input type="checkbox" id="f-err"> 删除评估异常 Attempts</label></div>
      <div class="fg">
        <label>Infeasible 剔除阈值 k ≥</label>
        <input type="number" id="f-inf" value="0" min="0" max="10" class="num-input">
      </div>
      <div class="fg">
        <label>SR 范围</label>
        <div class="range-row">
          <input type="range" id="f-sr-lo" min="0" max="1" step="0.05" value="0" oninput="document.getElementById('sr-lo-v').textContent=this.value">
          <span id="sr-lo-v">0</span> &le; SR &le;
          <input type="range" id="f-sr-hi" min="0" max="1" step="0.05" value="1" oninput="document.getElementById('sr-hi-v').textContent=this.value">
          <span id="sr-hi-v">1</span>
        </div>
      </div>
      <div class="fg">
        <label>死循环剔除 k ≥</label>
        <input type="number" id="f-loop" value="0" min="0" max="30" class="num-input">
      </div>
      <div class="fg"><label><input type="checkbox" id="f-succ"> 仅保留成功 Attempts</label></div>
      <div class="fg"><label><input type="checkbox" id="f-best"> 最优轨迹选择</label></div>
      <div class="fg"><label><input type="checkbox" id="f-pos"> 仅 Positive Steps</label></div>
      <div class="fg">
        <label>步骤数范围</label>
        <div class="range-row">
          <input type="number" id="f-smin" value="0" min="0" class="num-input" style="width:55px"> ≤ steps ≤
          <input type="number" id="f-smax" value="100" min="0" class="num-input" style="width:55px">
        </div>
      </div>
      <div class="btn-row">
        <button class="btn pri" onclick="onApply()">▶ 应用筛选</button>
        <button class="btn sec" onclick="onReset()">↺ 重置</button>
      </div>
    </div>
    <div id="filter-out">
      <div class="hint-box">选择筛选条件后点击 <b>应用筛选</b> 查看结果</div>
    </div>
  </div>
</section>

<!-- ═══════ Tab: 预计算分析 ═══════ -->
<section id="precomputed" class="tab-content">
  <p style="margin-bottom:16px;color:#64748b">以下是 Python 端预计算的各筛选策略单独应用后的效果：</p>
  <div id="pre-list"></div>
</section>

<!-- ═══════ Tab: 数据表 ═══════ -->
<section id="table" class="tab-content">
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
      <h3>任务明细表</h3>
      <input class="search-box" placeholder="搜索任务 ID / App / 描述..." oninput="onSearch(this.value)">
    </div>
    <div style="overflow-x:auto">
    <table><thead><tr>
      <th onclick="sortT(0)">任务 ID ⇅</th>
      <th onclick="sortT(1)">App ⇅</th>
      <th onclick="sortT(2)">轨迹 ⇅</th>
      <th onclick="sortT(3)">成功 ⇅</th>
      <th onclick="sortT(4)">SR ⇅</th>
      <th onclick="sortT(5)">步骤 ⇅</th>
      <th>Pos%</th><th>循环</th><th>Hint</th><th>描述</th>
    </tr></thead><tbody id="t-body"></tbody></table>
    </div>
  </div>
</section>

</main>

<script>
// ══════════════════════════════════════════════
// Embedded Data
// ══════════════════════════════════════════════
const TASKS={data_json};
const BASELINE={baseline_json};
const PER_APP={per_app_json};
const FILTERS={filter_json};
{_JS}
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════
_CSS = r"""
:root{--p:#4361ee;--pl:#e8edff;--s:#2ec4b6;--sl:#e6f9f7;
--w:#ff9f1c;--wl:#fff4e5;--d:#e63946;--dl:#fde8ea;
--bg:#f0f2f5;--card:#fff;--tx:#1a1a2e;--tx2:#64748b;--bd:#e2e8f0;
--sh:0 1px 3px rgba(0,0,0,.08);--shl:0 4px 12px rgba(0,0,0,.1);--r:12px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
background:var(--bg);color:var(--tx);line-height:1.6}
.header{background:linear-gradient(135deg,var(--p),#6c63ff);color:#fff;
padding:18px 28px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px}
.header h1{font-size:21px;font-weight:600}
.header .meta{font-size:12px;opacity:.8;max-width:600px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.app-select{padding:5px 10px;border:1px solid rgba(255,255,255,.3);border-radius:6px;
background:rgba(255,255,255,.15);color:#fff;font-size:13px}
.app-select option{color:var(--tx);background:#fff}
.tabs{display:flex;background:var(--card);border-bottom:1px solid var(--bd);
padding:0 20px;gap:2px;box-shadow:var(--sh);overflow-x:auto}
.tab{padding:11px 18px;cursor:pointer;font-size:13.5px;font-weight:500;
color:var(--tx2);border-bottom:3px solid transparent;transition:.2s;white-space:nowrap}
.tab:hover{color:var(--p)}.tab.active{color:var(--p);border-bottom-color:var(--p)}
.content{padding:22px;max-width:1440px;margin:0 auto}
.tab-content{display:none}.tab-content.active{display:block}
.kpi-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:14px;margin-bottom:18px}
.kpi-card{background:var(--card);border-radius:var(--r);padding:16px 18px;
box-shadow:var(--sh);border-left:4px solid var(--p)}
.kpi-card.c1{border-left-color:var(--s)}.kpi-card.c2{border-left-color:#8b5cf6}
.kpi-card.c3{border-left-color:var(--w)}.kpi-card.c4{border-left-color:#ec4899}
.kpi-card.sm{padding:12px 16px}.kpi-card.sm .value{font-size:20px}
.kpi-card .label{font-size:12px;color:var(--tx2);margin-bottom:2px}
.kpi-card .value{font-size:26px;font-weight:700}
.kpi-card .sub{font-size:11px;color:var(--tx2);margin-top:2px}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:18px;margin-bottom:18px}
.chart-card{background:var(--card);border-radius:var(--r);padding:18px;box-shadow:var(--sh)}
.chart-card h3{font-size:14px;margin-bottom:10px;color:var(--tx)}
.card{background:var(--card);border-radius:var(--r);padding:18px;box-shadow:var(--sh);margin-bottom:18px}
table{width:100%;border-collapse:collapse}
th,td{padding:9px 14px;text-align:left;font-size:12.5px;border-bottom:1px solid var(--bd)}
th{background:#f8fafc;font-weight:600;color:var(--tx2);cursor:pointer;user-select:none;white-space:nowrap}
th:hover{background:#eef1f5}tr:hover{background:#f8fafc}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
.bg{background:var(--sl);color:#0d9488}.bd{background:var(--dl);color:var(--d)}
.bw{background:var(--wl);color:#b45309}
.filter-panel{display:grid;grid-template-columns:290px 1fr;gap:18px}
.filter-controls{background:var(--card);border-radius:var(--r);padding:18px;
box-shadow:var(--sh);position:sticky;top:10px;max-height:calc(100vh - 120px);overflow-y:auto}
.filter-controls h3{font-size:14px;margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid var(--bd)}
.fg{margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #f1f5f9}
.fg:last-of-type{border-bottom:none}
.fg label{display:flex;align-items:center;gap:7px;font-size:12.5px;cursor:pointer;margin-bottom:4px}
.num-input{width:58px;padding:4px 7px;border:1px solid var(--bd);border-radius:4px;font-size:12px}
.range-row{display:flex;align-items:center;gap:6px;margin-top:6px;font-size:12px}
.range-row input[type=range]{flex:1;min-width:60px}
.btn{padding:7px 18px;border:none;border-radius:6px;cursor:pointer;font-size:12.5px;font-weight:500;transition:.2s}
.btn.pri{background:var(--p);color:#fff}.btn.pri:hover{background:#3451d4}
.btn.sec{background:var(--bd);color:var(--tx)}.btn.sec:hover{background:#d0d5dd}
.btn-row{display:flex;gap:8px;margin-top:14px}
.hint-box{padding:40px;text-align:center;color:var(--tx2);font-size:13px}
.search-box{padding:7px 12px;border:1px solid var(--bd);border-radius:6px;font-size:12.5px;width:260px}
.cmp-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
.cmp-card{background:var(--card);border-radius:var(--r);padding:14px;box-shadow:var(--sh)}
.cmp-card h4{font-size:13px;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--bd)}
.pre-card{background:var(--card);border-radius:var(--r);padding:16px;box-shadow:var(--sh);margin-bottom:14px}
.pre-card summary{cursor:pointer;font-weight:600;font-size:13.5px}
.pre-card summary:hover{color:var(--p)}
.delta-pos{color:var(--s);font-weight:600}.delta-neg{color:var(--d);font-weight:600}
@media(max-width:900px){.chart-grid{grid-template-columns:1fr}.filter-panel{grid-template-columns:1fr}
.kpi-row{grid-template-columns:repeat(2,1fr)}.cmp-grid{grid-template-columns:1fr}}
"""

# ══════════════════════════════════════════════════════════
# JavaScript
# ══════════════════════════════════════════════════════════
_JS = r"""
// ── State ──
let curApp='__all__';
let charts={};
let fCharts={};  // 筛选实验中的图表实例
let sortCol=-1,sortAsc=true;
let displayTasks=TASKS;

// ── Tabs ──
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById(t.dataset.tab).classList.add('active');
  });
});

// ── Metrics computation (client-side) ──
function calc(tasks){
  const m={tasks:tasks.length,atts:0,steps:0,succ:0,fail:0,err:0,
    sok:0,sfail:0,imp:{positive:0,negative:0,neutral:0,unknown:0},
    reas:{reasonable:0,unreasonable:0,unknown:0},
    sr:[],inf:0,feas:0,loops:{3:0,5:0,7:0},at:{},spa:[],
    hGen:0,hInp:0,hBoth:0,hNone:0,hByAtt:{}};
  for(const t of tasks){
    m.sr.push(t.sr);let iv=0,fv=0;
    for(const a of t.attempts){
      m.atts++;
      if(a.result===1)m.succ++;else if(a.result===0)m.fail++;else m.err++;
      if(a.feasible===true)fv++;else if(a.feasible===false)iv++;
      for(const k of[3,5,7])if(a.mc>=k)m.loops[k]++;
      // hint stats
      const g=!!a.hGen,i=!!a.hInp;
      if(g)m.hGen++;if(i)m.hInp++;if(g&&i)m.hBoth++;if(!g&&!i)m.hNone++;
      // per-attempt-index hint stats
      const aIdx=a.id||'';const aNum=parseInt(aIdx.replace('attempt_',''))||0;
      if(aNum>0){if(!m.hByAtt[aNum])m.hByAtt[aNum]={total:0,gen:0,inp:0};
        m.hByAtt[aNum].total++;if(g)m.hByAtt[aNum].gen++;if(i)m.hByAtt[aNum].inp++;}
      m.spa.push(a.steps.length);
      for(const s of a.steps){
        m.steps++;if(s.ok)m.sok++;else m.sfail++;
        m.imp[s.imp]=(m.imp[s.imp]||0)+1;
        m.reas[s.reas]=(m.reas[s.reas]||0)+1;
      }
    }
    if(iv>fv&&iv>0)m.inf++;else if(fv>0)m.feas++;
  }
  m.avgSr=m.sr.length?m.sr.reduce((a,b)=>a+b,0)/m.sr.length:0;
  m.succPct=m.atts?(m.succ/m.atts*100):0;
  m.posPct=m.steps?(m.imp.positive/m.steps*100):0;
  m.avgSpa=m.spa.length?(m.spa.reduce((a,b)=>a+b,0)/m.spa.length):0;
  // SR dist — 按精确 SR 值统计
  const srCounts={};
  for(const v of m.sr){const label=Math.round(v*100)+'%';srCounts[label]=(srCounts[label]||0)+1;}
  // 按数值排序
  m.srDist=Object.fromEntries(Object.entries(srCounts).sort((a,b)=>parseFloat(a[0])-parseFloat(b[0])));
  // pass@k: 按 attempt 序号排序后，前 k 个 attempt 中至少有 1 个成功的任务比例
  m.passK={};m.maxK=0;
  const taskAttResults=[];
  for(const t of tasks){
    const sorted=[...t.attempts].sort((a,b)=>{
      const na=parseInt((a.id||'').replace('attempt_',''))||0;
      const nb=parseInt((b.id||'').replace('attempt_',''))||0;return na-nb;});
    const succList=sorted.map(a=>a.success);
    taskAttResults.push(succList);
    if(succList.length>m.maxK)m.maxK=succList.length;
  }
  for(let k=1;k<=m.maxK;k++){
    let passed=0;
    for(const r of taskAttResults){if(r.slice(0,k).some(Boolean))passed++;}
    m.passK[k]=m.tasks?+(passed/m.tasks*100).toFixed(2):0;
  }
  return m;
}

// ── KPIs ──
function kpis(m){
  const s=id=>document.getElementById(id);
  s('k-tasks').textContent=m.tasks;
  s('k-atts').textContent=m.atts;
  s('k-steps').textContent=m.steps;
  s('k-sr').textContent=(m.avgSr*100).toFixed(1)+'%';
  s('k-succ').textContent=m.succPct.toFixed(1)+'%';
  s('k-pass1').textContent=(m.passK[1]||0).toFixed(1)+'%';
  s('k-inf').textContent=m.inf;
  s('k-loop').textContent=m.loops[3]||0;
  s('k-spa').textContent=m.avgSpa.toFixed(1);
  s('k-pos').textContent=m.posPct.toFixed(1)+'%';
  s('k-hgen').textContent=m.hGen;
  s('k-hgen-sub').textContent=m.atts?(m.hGen/m.atts*100).toFixed(1)+'% 的轨迹':'';
  s('k-hinp').textContent=m.hInp;
  s('k-hinp-sub').textContent=m.atts?(m.hInp/m.atts*100).toFixed(1)+'% 的轨迹':'';
}

// ── Charts ──
const COLORS={pos:'#2ec4b6',neg:'#e63946',neu:'#ff9f1c',unk:'#94a3b8',
  pri:'#4361ee',succ:'#2ec4b6',fail:'#e63946',err:'#94a3b8'};

function initCharts(m){
  const dopt={responsive:true,plugins:{legend:{position:'bottom',labels:{font:{size:11}}}}};
  charts.impact=new Chart(document.getElementById('c-impact'),{type:'doughnut',
    data:{labels:['Positive','Negative','Neutral','Unknown'],
      datasets:[{data:[m.imp.positive,m.imp.negative,m.imp.neutral,m.imp.unknown],
        backgroundColor:[COLORS.pos,COLORS.neg,COLORS.neu,COLORS.unk]}]},options:dopt});
  charts.success=new Chart(document.getElementById('c-success'),{type:'doughnut',
    data:{labels:['成功','失败','异常'],
      datasets:[{data:[m.succ,m.fail,m.err],
        backgroundColor:[COLORS.succ,COLORS.fail,COLORS.err]}]},options:dopt});
  charts.sr=new Chart(document.getElementById('c-sr'),{type:'bar',
    data:{labels:Object.keys(m.srDist),
      datasets:[{label:'任务数',data:Object.values(m.srDist),backgroundColor:COLORS.pri,borderRadius:5}]},
    options:{responsive:true,scales:{y:{beginAtZero:true}},plugins:{legend:{display:false}}}});
  charts.reas=new Chart(document.getElementById('c-reas'),{type:'doughnut',
    data:{labels:['Reasonable','Unreasonable','Unknown'],
      datasets:[{data:[m.reas.reasonable,m.reas.unreasonable,m.reas.unknown],
        backgroundColor:[COLORS.pos,COLORS.neg,COLORS.unk]}]},options:dopt});
  // Action types — compute from tasks
  const atc={};
  for(const t of displayTasks)for(const a of t.attempts)for(const s of a.steps){
    if(s.action&&Array.isArray(s.action)&&s.action[0])atc[s.action[0]]=(atc[s.action[0]]||0)+1;}
  // Not available in lite data, skip if empty. Use baseline instead
  let atLabels=Object.keys(BASELINE.action_types||{}).slice(0,10);
  let atData=atLabels.map(k=>(BASELINE.action_types||{})[k]||0);
  charts.actions=new Chart(document.getElementById('c-actions'),{type:'bar',
    data:{labels:atLabels,datasets:[{label:'次数',data:atData,backgroundColor:'#8b5cf6',borderRadius:5}]},
    options:{indexAxis:'y',responsive:true,scales:{x:{beginAtZero:true}},plugins:{legend:{display:false}}}});
  charts.feas=new Chart(document.getElementById('c-feas'),{type:'doughnut',
    data:{labels:['Feasible','Infeasible','未评估'],
      datasets:[{data:[m.feas,m.inf,m.tasks-m.feas-m.inf],
        backgroundColor:[COLORS.pos,COLORS.neg,COLORS.unk]}]},options:dopt});
  // Pass@k 折线图
  const pkKeys=Object.keys(m.passK).sort((a,b)=>a-b);
  charts.passk=new Chart(document.getElementById('c-passk'),{type:'line',
    data:{labels:pkKeys.map(k=>'@'+k),
      datasets:[{label:'Pass@k %',data:pkKeys.map(k=>m.passK[k]),
        borderColor:COLORS.pri,backgroundColor:COLORS.pri+'33',fill:true,tension:.3,pointRadius:5,pointHoverRadius:7}]},
    options:{responsive:true,scales:{y:{beginAtZero:true,max:100,title:{display:true,text:'%'}},
      x:{title:{display:true,text:'k (尝试次数)'}}},
      plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>ctx.parsed.y.toFixed(1)+'%'}}}}});
  // Hint 分布饼图
  charts.hint=new Chart(document.getElementById('c-hint'),{type:'doughnut',
    data:{labels:['仅生成 Hint','仅使用 Hint','生成+使用','无 Hint'],
      datasets:[{data:[m.hGen-m.hBoth,m.hInp-m.hBoth,m.hBoth,m.hNone],
        backgroundColor:['#8b5cf6','#06b6d4','#2ec4b6','#94a3b8']}]},options:dopt});
  // 各 Attempt 序号 Hint 覆盖率
  const hAttKeys=Object.keys(m.hByAtt).sort((a,b)=>a-b);
  charts.hintAtt=new Chart(document.getElementById('c-hint-att'),{type:'bar',
    data:{labels:hAttKeys.map(k=>'Attempt '+k),
      datasets:[
        {label:'生成 Hint %',data:hAttKeys.map(k=>m.hByAtt[k].total?(m.hByAtt[k].gen/m.hByAtt[k].total*100).toFixed(1):0),backgroundColor:'#8b5cf6',borderRadius:5},
        {label:'使用 Hint %',data:hAttKeys.map(k=>m.hByAtt[k].total?(m.hByAtt[k].inp/m.hByAtt[k].total*100).toFixed(1):0),backgroundColor:'#06b6d4',borderRadius:5}
      ]},
    options:{responsive:true,scales:{y:{beginAtZero:true,max:100}},plugins:{legend:{position:'bottom',labels:{font:{size:11}}}}}});
}

function updateCharts(m){
  charts.impact.data.datasets[0].data=[m.imp.positive,m.imp.negative,m.imp.neutral,m.imp.unknown];
  charts.impact.update();
  charts.success.data.datasets[0].data=[m.succ,m.fail,m.err];charts.success.update();
  charts.sr.data.labels=Object.keys(m.srDist);
  charts.sr.data.datasets[0].data=Object.values(m.srDist);charts.sr.update();
  charts.reas.data.datasets[0].data=[m.reas.reasonable,m.reas.unreasonable,m.reas.unknown];
  charts.reas.update();
  charts.feas.data.datasets[0].data=[m.feas,m.inf,m.tasks-m.feas-m.inf];charts.feas.update();
  // Pass@k
  const pkKeys2=Object.keys(m.passK).sort((a,b)=>a-b);
  charts.passk.data.labels=pkKeys2.map(k=>'@'+k);
  charts.passk.data.datasets[0].data=pkKeys2.map(k=>m.passK[k]);charts.passk.update();
  // Hint charts
  charts.hint.data.datasets[0].data=[m.hGen-m.hBoth,m.hInp-m.hBoth,m.hBoth,m.hNone];charts.hint.update();
  const hAttKeys=Object.keys(m.hByAtt).sort((a,b)=>a-b);
  charts.hintAtt.data.labels=hAttKeys.map(k=>'Attempt '+k);
  charts.hintAtt.data.datasets[0].data=hAttKeys.map(k=>m.hByAtt[k].total?(m.hByAtt[k].gen/m.hByAtt[k].total*100).toFixed(1):0);
  charts.hintAtt.data.datasets[1].data=hAttKeys.map(k=>m.hByAtt[k].total?(m.hByAtt[k].inp/m.hByAtt[k].total*100).toFixed(1):0);
  charts.hintAtt.update();
}

// ── App Analysis ──
function renderApps(){
  const tbody=document.getElementById('app-tbody');
  const apps=Object.entries(PER_APP).sort((a,b)=>(b[1].task_count||0)-(a[1].task_count||0));
  let html='';
  const barLabels=[],barData=[];
  for(const [app,m] of apps){
    if(m.empty)continue;
    barLabels.push(app);barData.push(m.success_att_pct||0);
    const pk=m.pass_at_k||{};const maxK=m.max_k||0;
    const p1=(pk['1']||pk[1]||0);const pMax=maxK>0?(pk[String(maxK)]||pk[maxK]||0):0;
    html+=`<tr><td><b>${app}</b></td><td>${m.task_count}</td><td>${m.attempt_count}</td>
      <td>${m.step_count}</td><td>${m.success_att_pct}%</td><td>${(m.avg_sr||0).toFixed(2)}</td>
      <td>${p1}%</td><td>${pMax}% (@${maxK})</td>
      <td>${(m.impact_pct||{}).positive||0}%</td><td>${(m.impact_pct||{}).negative||0}%</td>
      <td>${(m.loop_counts||{})[3]||0}</td>
      <td>${m.hint_generated||0} (${m.hint_generated_pct||0}%)</td>
      <td>${m.hint_consumed||0} (${m.hint_consumed_pct||0}%)</td></tr>`;
  }
  tbody.innerHTML=html;
  if(!charts.appBar){
    charts.appBar=new Chart(document.getElementById('c-app-bar'),{type:'bar',
      data:{labels:barLabels,datasets:[{label:'成功率 %',data:barData,backgroundColor:COLORS.pri,borderRadius:5}]},
      options:{indexAxis:'y',responsive:true,scales:{x:{beginAtZero:true,max:100}},
        plugins:{legend:{display:false}}}});
  }
  // Per-app pass@k 折线图
  const appColors=['#4361ee','#e63946','#2ec4b6','#ff9f1c','#8b5cf6','#ec4899','#06b6d4','#f97316','#84cc16','#6366f1'];
  const appPkDatasets=[];let ci=0;
  let globalMaxK=0;
  for(const [app,m] of apps){
    if(m.empty)continue;
    const pk=m.pass_at_k||{};const ks=Object.keys(pk).map(Number).sort((a,b)=>a-b);
    if(ks.length&&ks[ks.length-1]>globalMaxK)globalMaxK=ks[ks.length-1];
    appPkDatasets.push({app,pk,ks,color:appColors[ci%appColors.length]});ci++;
  }
  const pkLabels=Array.from({length:globalMaxK},(_, i)=>'@'+(i+1));
  const pkDatasets=appPkDatasets.map(d=>({label:d.app,
    data:pkLabels.map((_,i)=>{const k=i+1;return d.pk[String(k)]||d.pk[k]||null;}),
    borderColor:d.color,backgroundColor:d.color+'33',tension:.3,pointRadius:3,spanGaps:true}));
  if(!charts.appPassk){
    charts.appPassk=new Chart(document.getElementById('c-app-passk'),{type:'line',
      data:{labels:pkLabels,datasets:pkDatasets},
      options:{responsive:true,scales:{y:{beginAtZero:true,max:100,title:{display:true,text:'%'}},
        x:{title:{display:true,text:'k'}}},
        plugins:{legend:{position:'bottom',labels:{font:{size:11}}}}}});
  }
}

// ── Pre-computed filter results ──
function renderPrecomputed(){
  const el=document.getElementById('pre-list');
  let html='';
  for(const f of FILTERS){
    const m=f.metrics;if(!m||m.empty)continue;
    const b=BASELINE;
    html+=`<details class="pre-card"><summary>${f.name}</summary>
      <p style="color:var(--tx2);font-size:12px;margin:8px 0">${f.desc}</p>
      <table><tr><th>指标</th><th>筛选前</th><th>筛选后</th><th>变化</th></tr>
      ${cmpRow('任务数',b.task_count,m.task_count)}
      ${cmpRow('轨迹数',b.attempt_count,m.attempt_count)}
      ${cmpRow('步骤数',b.step_count,m.step_count)}
      ${cmpRowPct('平均SR',b.avg_sr*100,m.avg_sr*100)}
      ${cmpRowPct('成功率',b.success_att_pct,m.success_att_pct)}
      ${cmpRowPct('Pass@1',(b.pass_at_k||{})[1]||0,(m.pass_at_k||{})[1]||0)}
      ${(()=>{const bk=b.max_k||0,mk=m.max_k||0;
        return cmpRowPct('Pass@max('+Math.max(bk,mk)+')',(b.pass_at_k||{})[bk]||0,(m.pass_at_k||{})[mk]||0);})()}
      ${cmpRowPct('Positive%',(b.impact_pct||{}).positive||0,(m.impact_pct||{}).positive||0)}
      ${cmpRowPct('Negative%',(b.impact_pct||{}).negative||0,(m.impact_pct||{}).negative||0)}
      ${cmpRow('Hint生成',b.hint_generated||0,m.hint_generated||0)}
      ${cmpRow('Hint使用',b.hint_consumed||0,m.hint_consumed||0)}
      </table>`;
    // examples
    if(f.examples&&f.examples.length){
      html+='<div style="margin-top:8px;font-size:12px;color:var(--tx2)"><b>示例:</b><ul>';
      for(const ex of f.examples){
        if(ex._summary){html+=`<li>删除 ${ex.removed||0} 步, 保留 ${ex.kept||0} 步</li>`;continue;}
        if(ex.task_id)html+=`<li><code>${ex.task_id}</code>${ex.attempt?' / '+ex.attempt:''} — ${JSON.stringify(ex).slice(0,120)}</li>`;
        else html+=`<li>${JSON.stringify(ex).slice(0,150)}</li>`;
      }
      html+='</ul></div>';
    }
    html+='</details>';
  }
  el.innerHTML=html;
}
function cmpRow(label,bv,av){
  const d=av-bv;const cls=d>0?'delta-pos':d<0?'delta-neg':'';
  return `<tr><td>${label}</td><td>${bv}</td><td>${av}</td><td class="${cls}">${d>0?'+':''}${d}</td></tr>`;
}
function cmpRowPct(label,bv,av){
  const d=(av-bv).toFixed(2);const cls=d>0?'delta-pos':d<0?'delta-neg':'';
  return `<tr><td>${label}</td><td>${bv.toFixed(2)}%</td><td>${av.toFixed(2)}%</td><td class="${cls}">${d>0?'+':''}${d}%</td></tr>`;
}

// ── Filter Lab ──
function onApply(){
  let base=curApp==='__all__'?TASKS:TASKS.filter(t=>t.app===curApp);
  const f={
    err:document.getElementById('f-err').checked,
    inf:parseInt(document.getElementById('f-inf').value)||0,
    srLo:parseFloat(document.getElementById('f-sr-lo').value)||0,
    srHi:parseFloat(document.getElementById('f-sr-hi').value),
    loop:parseInt(document.getElementById('f-loop').value)||0,
    succ:document.getElementById('f-succ').checked,
    best:document.getElementById('f-best').checked,
    pos:document.getElementById('f-pos').checked,
    smin:parseInt(document.getElementById('f-smin').value)||0,
    smax:parseInt(document.getElementById('f-smax').value)||9999,
  };
  if(isNaN(f.srHi))f.srHi=1;
  let r=JSON.parse(JSON.stringify(base));
  const ex=[];
  // apply filters in order
  if(f.err)r=r.map(t=>({...t,attempts:t.attempts.filter(a=>a.result===0||a.result===1)})).filter(t=>t.attempts.length);
  if(f.inf>0)r=r.filter(t=>{const c=t.attempts.filter(a=>a.feasible===false).length;
    if(c>=f.inf){ex.push({t:'inf',id:t.id,v:c});return false;}return true;});
  r=r.filter(t=>{if(t.sr<f.srLo||t.sr>f.srHi){ex.push({t:'sr',id:t.id,v:t.sr});return false;}return true;});
  if(f.loop>0)r=r.map(t=>({...t,attempts:t.attempts.filter(a=>{
    if(a.mc>=f.loop){ex.push({t:'loop',id:t.id,a:a.id,v:a.mc});return false;}return true;})})).filter(t=>t.attempts.length);
  if(f.succ)r=r.map(t=>({...t,attempts:t.attempts.filter(a=>a.success)})).filter(t=>t.attempts.length);
  r=r.map(t=>({...t,attempts:t.attempts.filter(a=>a.steps.length>=f.smin&&a.steps.length<=f.smax)})).filter(t=>t.attempts.length);
  if(f.best)r=r.map(t=>{if(t.attempts.length<=1)return t;
    const sc=t.attempts.map(a=>{const pr=a.steps.length?a.steps.filter(s=>s.imp==='positive').length/a.steps.length:0;
      return{...a,sc:(a.success?1e3:0)+pr};}).sort((a,b)=>b.sc-a.sc);
    return{...t,attempts:[sc[0]]};});
  if(f.pos)r=r.map(t=>({...t,attempts:t.attempts.map(a=>({...a,steps:a.steps.filter(s=>s.imp==='positive')}))}));
  // recalc sr
  r=r.map(t=>{const ok=t.attempts.filter(a=>a.success).length;return{...t,sr:t.attempts.length?ok/t.attempts.length:0};});

  const before=calc(base),after=calc(r);
  const el=document.getElementById('filter-out');
  const d=(b,a)=>{const v=a-b;return v>0?`<span class="delta-pos">+${v}</span>`:v<0?`<span class="delta-neg">${v}</span>`:'0';};
  const dp=(b,a)=>{const v=(a-b).toFixed(2);return v>0?`<span class="delta-pos">+${v}%</span>`:v<0?`<span class="delta-neg">${v}%</span>`:'0%';};
  const negPctB=before.steps?before.imp.negative/before.steps*100:0;
  const negPctA=after.steps?after.imp.negative/after.steps*100:0;
  el.innerHTML=`
    <!-- KPI 变化汇总 -->
    <div class="kpi-row">
      <div class="kpi-card sm"><div class="label">任务</div><div class="value">${before.tasks} → ${after.tasks}</div></div>
      <div class="kpi-card sm c1"><div class="label">轨迹</div><div class="value">${before.atts} → ${after.atts}</div></div>
      <div class="kpi-card sm c2"><div class="label">步骤</div><div class="value">${before.steps} → ${after.steps}</div></div>
      <div class="kpi-card sm c4"><div class="label">SR</div><div class="value">${(before.avgSr*100).toFixed(1)}→${(after.avgSr*100).toFixed(1)}%</div></div>
      <div class="kpi-card sm c3"><div class="label">成功率</div><div class="value">${before.succPct.toFixed(1)}→${after.succPct.toFixed(1)}%</div></div>
      <div class="kpi-card sm"><div class="label">Pass@1</div><div class="value">${(before.passK[1]||0).toFixed(1)}→${(after.passK[1]||0).toFixed(1)}%</div></div>
    </div>
    <div class="kpi-row" style="margin-bottom:8px">
      <div class="kpi-card sm"><div class="label">Infeasible</div><div class="value">${before.inf} → ${after.inf}</div></div>
      <div class="kpi-card sm"><div class="label">死循环(k≥3)</div><div class="value">${before.loops[3]} → ${after.loops[3]}</div></div>
      <div class="kpi-card sm"><div class="label">平均steps/轨迹</div><div class="value">${before.avgSpa.toFixed(1)} → ${after.avgSpa.toFixed(1)}</div></div>
      <div class="kpi-card sm"><div class="label">Positive%</div><div class="value">${before.posPct.toFixed(1)}→${after.posPct.toFixed(1)}%</div></div>
      <div class="kpi-card sm"><div class="label">Hint生成</div><div class="value">${before.hGen} → ${after.hGen}</div></div>
      <div class="kpi-card sm"><div class="label">Hint使用</div><div class="value">${before.hInp} → ${after.hInp}</div></div>
    </div>
    <!-- 筛选前后对比表 -->
    <div class="card"><table>
      <tr><th>指标</th><th>筛选前</th><th>筛选后</th><th>变化</th></tr>
      <tr><td>任务数</td><td>${before.tasks}</td><td>${after.tasks}</td><td>${d(before.tasks,after.tasks)}</td></tr>
      <tr><td>轨迹数</td><td>${before.atts}</td><td>${after.atts}</td><td>${d(before.atts,after.atts)}</td></tr>
      <tr><td>步骤数</td><td>${before.steps}</td><td>${after.steps}</td><td>${d(before.steps,after.steps)}</td></tr>
      <tr><td>平均SR</td><td>${(before.avgSr*100).toFixed(2)}%</td><td>${(after.avgSr*100).toFixed(2)}%</td><td>${dp(before.avgSr*100,after.avgSr*100)}</td></tr>
      <tr><td>轨迹成功率</td><td>${before.succPct.toFixed(2)}%</td><td>${after.succPct.toFixed(2)}%</td><td>${dp(before.succPct,after.succPct)}</td></tr>
      <tr><td>Pass@1</td><td>${(before.passK[1]||0).toFixed(2)}%</td><td>${(after.passK[1]||0).toFixed(2)}%</td><td>${dp(before.passK[1]||0,after.passK[1]||0)}</td></tr>
      <tr><td>Positive%</td><td>${before.posPct.toFixed(2)}%</td><td>${after.posPct.toFixed(2)}%</td><td>${dp(before.posPct,after.posPct)}</td></tr>
      <tr><td>Negative%</td><td>${negPctB.toFixed(2)}%</td><td>${negPctA.toFixed(2)}%</td><td>${dp(negPctB,negPctA)}</td></tr>
      <tr><td>Infeasible</td><td>${before.inf}</td><td>${after.inf}</td><td>${d(before.inf,after.inf)}</td></tr>
      <tr><td>死循环(k≥3)</td><td>${before.loops[3]}</td><td>${after.loops[3]}</td><td>${d(before.loops[3],after.loops[3])}</td></tr>
      <tr><td>Hint生成</td><td>${before.hGen}</td><td>${after.hGen}</td><td>${d(before.hGen,after.hGen)}</td></tr>
      <tr><td>Hint使用</td><td>${before.hInp}</td><td>${after.hInp}</td><td>${d(before.hInp,after.hInp)}</td></tr>
    </table></div>
    <!-- ═══ 筛选后数据分布图表 (与概览一致) ═══ -->
    <h3 style="margin:18px 0 12px;font-size:15px">📊 筛选后数据分布</h3>
    <div class="chart-grid">
      <div class="chart-card"><h3>步骤 Impact 分布</h3><canvas id="fc-impact"></canvas></div>
      <div class="chart-card"><h3>轨迹成功/失败</h3><canvas id="fc-success"></canvas></div>
      <div class="chart-card"><h3>任务 SR 分布</h3><canvas id="fc-sr"></canvas></div>
      <div class="chart-card"><h3>步骤合理性分布</h3><canvas id="fc-reas"></canvas></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><h3>Pass@k 曲线</h3><canvas id="fc-passk"></canvas></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><h3>Infeasible / Feasible / 未知</h3><canvas id="fc-feas"></canvas></div>
      <div class="chart-card"><h3>Eval Hint 生成与使用</h3><canvas id="fc-hint"></canvas></div>
    </div>
    <div class="chart-grid">
      <div class="chart-card"><h3>各 Attempt 序号 Hint 覆盖率</h3><canvas id="fc-hint-att"></canvas></div>
    </div>
    ${ex.length?'<div class="card"><h4 style="margin-bottom:8px">筛选示例 ('+ex.length+')</h4><ul style="font-size:12px;color:var(--tx2)">'+
      ex.slice(0,12).map(e=>'<li><code>'+e.id+'</code>'+(e.a?'/'+e.a:'')+' — '+e.t+'='+JSON.stringify(e.v)+'</li>').join('')+'</ul></div>':''}`;
  // 渲染筛选后的图表
  renderFilterCharts(after);
}
function renderFilterCharts(m){
  // 销毁旧图表
  Object.values(fCharts).forEach(c=>{try{c.destroy();}catch(e){}});
  fCharts={};
  const dopt={responsive:true,plugins:{legend:{position:'bottom',labels:{font:{size:11}}}}};
  // Impact 饼图
  fCharts.impact=new Chart(document.getElementById('fc-impact'),{type:'doughnut',
    data:{labels:['Positive','Negative','Neutral','Unknown'],
      datasets:[{data:[m.imp.positive,m.imp.negative,m.imp.neutral,m.imp.unknown],
        backgroundColor:[COLORS.pos,COLORS.neg,COLORS.neu,COLORS.unk]}]},options:dopt});
  // 成功/失败 饼图
  fCharts.success=new Chart(document.getElementById('fc-success'),{type:'doughnut',
    data:{labels:['成功','失败','异常'],
      datasets:[{data:[m.succ,m.fail,m.err],
        backgroundColor:[COLORS.succ,COLORS.fail,COLORS.err]}]},options:dopt});
  // SR 分布
  fCharts.sr=new Chart(document.getElementById('fc-sr'),{type:'bar',
    data:{labels:Object.keys(m.srDist),
      datasets:[{label:'任务数',data:Object.values(m.srDist),backgroundColor:COLORS.pri,borderRadius:5}]},
    options:{responsive:true,scales:{y:{beginAtZero:true}},plugins:{legend:{display:false}}}});
  // 合理性
  fCharts.reas=new Chart(document.getElementById('fc-reas'),{type:'doughnut',
    data:{labels:['Reasonable','Unreasonable','Unknown'],
      datasets:[{data:[m.reas.reasonable,m.reas.unreasonable,m.reas.unknown],
        backgroundColor:[COLORS.pos,COLORS.neg,COLORS.unk]}]},options:dopt});
  // Pass@k
  const pkKeys=Object.keys(m.passK).sort((a,b)=>a-b);
  fCharts.passk=new Chart(document.getElementById('fc-passk'),{type:'line',
    data:{labels:pkKeys.map(k=>'@'+k),
      datasets:[{label:'Pass@k %',data:pkKeys.map(k=>m.passK[k]),
        borderColor:COLORS.pri,backgroundColor:COLORS.pri+'33',fill:true,tension:.3,pointRadius:5,pointHoverRadius:7}]},
    options:{responsive:true,scales:{y:{beginAtZero:true,max:100,title:{display:true,text:'%'}},
      x:{title:{display:true,text:'k (尝试次数)'}}},
      plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>ctx.parsed.y.toFixed(1)+'%'}}}}});
  // Feasibility
  fCharts.feas=new Chart(document.getElementById('fc-feas'),{type:'doughnut',
    data:{labels:['Feasible','Infeasible','未评估'],
      datasets:[{data:[m.feas,m.inf,m.tasks-m.feas-m.inf],
        backgroundColor:[COLORS.pos,COLORS.neg,COLORS.unk]}]},options:dopt});
  // Hint 分布
  fCharts.hint=new Chart(document.getElementById('fc-hint'),{type:'doughnut',
    data:{labels:['仅生成 Hint','仅使用 Hint','生成+使用','无 Hint'],
      datasets:[{data:[m.hGen-m.hBoth,m.hInp-m.hBoth,m.hBoth,m.hNone],
        backgroundColor:['#8b5cf6','#06b6d4','#2ec4b6','#94a3b8']}]},options:dopt});
  // 各 Attempt 序号 Hint 覆盖率
  const hAttKeys=Object.keys(m.hByAtt).sort((a,b)=>a-b);
  fCharts.hintAtt=new Chart(document.getElementById('fc-hint-att'),{type:'bar',
    data:{labels:hAttKeys.map(k=>'Attempt '+k),
      datasets:[
        {label:'生成 Hint %',data:hAttKeys.map(k=>m.hByAtt[k].total?(m.hByAtt[k].gen/m.hByAtt[k].total*100).toFixed(1):0),backgroundColor:'#8b5cf6',borderRadius:5},
        {label:'使用 Hint %',data:hAttKeys.map(k=>m.hByAtt[k].total?(m.hByAtt[k].inp/m.hByAtt[k].total*100).toFixed(1):0),backgroundColor:'#06b6d4',borderRadius:5}
      ]},
    options:{responsive:true,scales:{y:{beginAtZero:true,max:100}},plugins:{legend:{position:'bottom',labels:{font:{size:11}}}}}});
}

function onReset(){
  document.querySelectorAll('.filter-controls input[type=checkbox]').forEach(c=>c.checked=false);
  document.getElementById('f-inf').value=0;document.getElementById('f-loop').value=0;
  document.getElementById('f-sr-lo').value=0;document.getElementById('f-sr-hi').value=1;
  document.getElementById('sr-lo-v').textContent='0';document.getElementById('sr-hi-v').textContent='1';
  document.getElementById('f-smin').value=0;document.getElementById('f-smax').value=100;
  document.getElementById('filter-out').innerHTML='<div class="hint-box">选择筛选条件后点击 <b>应用筛选</b> 查看结果</div>';
}

// ── Data Table ──
function renderTable(tasks){
  const tbody=document.getElementById('t-body');
  let html='';
  for(const t of tasks){
    const ac=t.attempts.length,sc=t.attempts.filter(a=>a.success).length;
    const ts=t.attempts.reduce((s,a)=>s+a.steps.length,0);
    const ps=t.attempts.reduce((s,a)=>s+a.steps.filter(x=>x.imp==='positive').length,0);
    const lp=t.attempts.filter(a=>a.mc>=3).length;
    const hg=t.attempts.filter(a=>a.hGen).length;
    const hi=t.attempts.filter(a=>a.hInp).length;
    const hInfo=hg||hi?`<span class="badge bg" title="生成:${hg} 使用:${hi}">G${hg}/U${hi}</span>`:'<span style="color:#94a3b8">-</span>';
    html+=`<tr>
      <td style="font-size:11px;font-family:monospace">${t.id}</td>
      <td>${t.app}</td><td>${ac}</td>
      <td><span class="badge ${sc?'bg':'bd'}">${sc}/${ac}</span></td>
      <td>${(t.sr*100).toFixed(0)}%</td><td>${ts}</td>
      <td>${ts?(ps/ts*100).toFixed(0):0}%</td>
      <td>${lp?'<span class="badge bw">'+lp+'</span>':'-'}</td>
      <td>${hInfo}</td>
      <td style="max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${(t.desc||'').replace(/"/g,'&quot;')}">${t.desc||''}</td></tr>`;
  }
  tbody.innerHTML=html;
}
function sortT(col){
  if(sortCol===col)sortAsc=!sortAsc;else{sortCol=col;sortAsc=true;}
  const fn=[
    (a,b)=>a.id.localeCompare(b.id),(a,b)=>a.app.localeCompare(b.app),
    (a,b)=>a.attempts.length-b.attempts.length,
    (a,b)=>a.attempts.filter(x=>x.success).length-b.attempts.filter(x=>x.success).length,
    (a,b)=>a.sr-b.sr,
    (a,b)=>a.attempts.reduce((s,x)=>s+x.steps.length,0)-b.attempts.reduce((s,x)=>s+x.steps.length,0),
  ][col]||((a,b)=>0);
  const sorted=[...displayTasks].sort((a,b)=>sortAsc?fn(a,b):fn(b,a));
  renderTable(sorted);
}
function onSearch(q){
  q=q.toLowerCase();
  const filtered=displayTasks.filter(t=>t.id.toLowerCase().includes(q)||t.app.toLowerCase().includes(q)||(t.desc||'').toLowerCase().includes(q));
  renderTable(filtered);
}

// ── App Change ──
function onAppChange(app){
  curApp=app;
  displayTasks=app==='__all__'?TASKS:TASKS.filter(t=>t.app===app);
  const m=calc(displayTasks);
  kpis(m);updateCharts(m);renderTable(displayTasks);
}

// ── Init ──
document.addEventListener('DOMContentLoaded',()=>{
  const m=calc(TASKS);
  kpis(m);initCharts(m);renderTable(TASKS);
  renderApps();renderPrecomputed();
});
"""


def _sanitize(obj):
    """递归清理对象使其可 JSON 序列化"""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)
