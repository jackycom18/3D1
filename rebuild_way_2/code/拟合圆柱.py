# -*- coding: utf-8 -*-
"""拟合圆柱：对 pointcloud.ply 点云做圆柱最小二乘拟合（轴方向+轴线位置+半径，共 5 参数），
输出拟合参数与精度指标，并生成自包含 HTML 可视化报告（内联 SVG，无外部依赖）。

流程：
1. 读入 pointcloud.ply 的 (N,3) 点云
2. PCA 估计初始轴方向（最小方差主方向），中位数估计初始半径
3. 高斯-牛顿/LM 迭代精修 5 参数（θ,φ 轴方向 + α,β 轴线位置 + 半径 r）
4. 计算精度：径向 RMS/MAE/最大偏差/标准差、半径 1σ 不确定度、椭圆度、锥度
5. 保存 cylinder_fit.npz + 生成 HTML 报告

运行：python 拟合圆柱.py
"""
import os
import numpy as np

PTS = r"C:\python\3D_project_3\rebuild_way_2\data\pointcloud\pointcloud_steger.ply"   # 输入点云
OUT_NPZ = r"C:\python\3D_project_3\cylinder_fit.npz"             # 拟合结果
OUT_HTML = r"C:\python\3D_project_3\圆柱拟合报告.html"           # 可视化报告


def _perp_dist(p, a, C):
    """功能：计算各点到轴线 (C, a) 的垂直距离。输入 p：(N,3)；a：单位轴方向；C：轴线上一点。输出 (N,)"""
    d = p - C
    return np.linalg.norm(d - (d @ a)[:, None] * a, axis=1)


def fit_cylinder(pts):
    """功能：最小二乘拟合圆柱。输入 pts：(N,3) 点云。输出 dict：轴方向 axis、轴线上一点 center、半径 radius、
    半径1σ sigma_r、径向残差 res、rms/mae/max/std、夹角 angle_deg。"""
    N = len(pts)
    c0 = pts.mean(0)                                    # 质心，作轴线上一点初值

    # 初始轴方向 = PCA 最小方差主方向（圆柱侧面点云沿轴向方差最小）
    _, _, vt = np.linalg.svd(pts - c0, full_matrices=False)
    a0 = vt[2].copy()
    if a0[2] < 0:
        a0 = -a0
    a0 /= np.linalg.norm(a0)
    r0 = float(np.median(_perp_dist(pts, a0, c0)))      # 中位数抗噪

    # 参数化：a=[sinθcosφ, sinθsinφ, cosθ]；C=c0+αe1+βe2；r
    theta0 = float(np.arccos(np.clip(a0[2], -1, 1)))
    phi0 = float(np.arctan2(a0[1], a0[0]))
    params = np.array([theta0, phi0, 0.0, 0.0, r0])

    def _basis(a):
        t = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = t - (t @ a) * a
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        return e1, e2

    def residuals(p):
        th, ph, al, be, r = p
        a = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        e1, e2 = _basis(a)
        C = c0 + al * e1 + be * e2
        return _perp_dist(pts, a, C) - r

    # LM 迭代（数值雅可比）
    steps = np.array([1e-6, 1e-6, 1e-3, 1e-3, 1e-3])
    lam = 1e-3
    for _ in range(200):
        R = residuals(params)
        cost = float(R @ R)
        J = np.zeros((N, 5))
        for j in range(5):
            pp = params.copy(); pp[j] += steps[j]
            J[:, j] = (residuals(pp) - R) / steps[j]
        A = J.T @ J
        g = J.T @ R
        try:
            delta = np.linalg.solve(A + lam * np.diag(np.diag(A)), -g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        newR = residuals(params + delta)
        if float(newR @ newR) < cost:
            params = params + delta
            lam = max(lam / 3, 1e-12)
            if np.linalg.norm(delta) < 1e-9:
                break
        else:
            lam *= 10

    th, ph, al, be, r = params
    a = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    e1, e2 = _basis(a)
    C = c0 + al * e1 + be * e2
    R = residuals(params)

    rms = float(np.sqrt(np.mean(R ** 2)))
    mae = float(np.mean(np.abs(R)))
    mx = float(np.max(np.abs(R)))
    sd = float(np.std(R))
    angle_deg = float(np.degrees(np.arccos(np.clip(abs(a[2]), 0, 1))))

    # 半径 1σ 不确定度
    J = np.zeros((N, 5))
    for j in range(5):
        pp = params.copy(); pp[j] += steps[j]
        J[:, j] = (residuals(pp) - R) / steps[j]
    try:
        sigma_r = float(np.sqrt(np.linalg.inv(J.T @ J)[4, 4] * (float(R @ R) / (N - 5))))
    except np.linalg.LinAlgError:
        sigma_r = float('nan')

    return dict(axis=a, center=C, radius=r, sigma_r=sigma_r, res=R,
                rms=rms, mae=mae, maxdev=mx, std=sd, angle_deg=angle_deg,
                n=N, e1=e1, e2=e2)


def report_console(pts, res):
    """功能：在控制台打印拟合参数与精度报告。输入 pts 点云；res 拟合结果 dict。输出 无"""
    a, C, r = res['axis'], res['center'], res['radius']
    print("=" * 64)
    print("圆柱拟合结果")
    print("=" * 64)
    print(f"点数           : {res['n']}")
    print(f"轴方向 a       : [{a[0]:+.6f}, {a[1]:+.6f}, {a[2]:+.6f}]")
    print(f"轴线与 z 轴夹角: {res['angle_deg']:.4f} deg")
    print(f"轴线上一点 C   : [{C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f}] mm")
    print(f"半径 r         : {r:.4f} mm  (±{res['sigma_r']:.4f} mm, 1σ 统计不确定度)")
    print("-" * 64)
    print("拟合精度（径向点到表面距离 = 垂直距离 - 半径）:")
    print(f"  RMS (均方根) : {res['rms']:.4f} mm")
    print(f"  MAE (平均绝对): {res['mae']:.4f} mm")
    print(f"  标准差 σ     : {res['std']:.4f} mm")
    print(f"  最大偏差     : {res['maxdev']:.4f} mm")
    print(f"  相对半径误差 : {res['rms']/r*100:.3f} %")
    print("=" * 64)

    # 锥度
    zs = pts[:, 2]
    zmin, zmax = zs.min(), zs.max()
    print("\n沿 z 分层半径（锥度检查）:")
    for k in range(0, int(zmax - zmin) + 1, 3):
        lo, hi = zmin + k, zmin + k + 3
        m = (zs >= lo) & (zs < hi)
        if m.sum() < 10:
            continue
        rr = _perp_dist(pts[m], a, C)
        print(f"  z[{lo:5.1f},{hi:5.1f}]: n={m.sum():5d}, r_mean={rr.mean():.3f}, 偏差={rr.mean()-r:+.3f} mm")


# ---------------- HTML 报告（内联 SVG） ----------------
def _svg_xy(center, scale, pts2d):
    """功能：把 2D 点映射到 SVG 画布坐标。输出字符串拼接的 <circle> 元素"""
    out = []
    cx, cy = center
    for x, y in pts2d:
        sx = cx + x * scale
        sy = cy - y * scale
        out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="0.9"/>')
    return "".join(out)


def gen_html(pts, res, out_path):
    """功能：生成自包含 HTML 可视化报告。输入 pts 点云；res 拟合结果；out_path 输出路径。输出 无"""
    a, C, r = res['axis'], res['center'], res['radius']
    e1, e2 = res['e1'], res['e2']

    # --- 截面散点（抽样 3500 点，投影到垂直平面） ---
    d = pts - C
    perp = d - (d @ a)[:, None] * a
    u = perp @ e1
    v = perp @ e2
    idx = np.random.default_rng(0).choice(len(pts), 3500, replace=False)
    pts2d = np.stack([u[idx], v[idx]], 1)
    scale = 280 / (r + 8)                     # 画布 640，半径约 47.5
    svg_pts = _svg_xy((320, 320), scale, pts2d)

    # --- 残差直方图（30 bin） ---
    R = res['res']
    bins = np.linspace(R.min(), R.max(), 31)
    hist, edges = np.histogram(R, bins=bins)
    hmax = hist.max()
    hw, hh, hbase = 560, 180, 140
    bars = []
    for k in range(30):
        x = 40 + k * (hw / 30)
        bh = hist[k] / hmax * (hh - 10)
        bars.append(f'<rect x="{x:.1f}" y="{hbase-bh:.1f}" width="{hw/30-1:.1f}" height="{bh:.1f}"/>')

    # --- 椭圆度：绕轴每 30° 桶半径 ---
    ang = np.arctan2(v, u)
    binedges = np.linspace(-np.pi, np.pi, 13)
    r_bins = []
    for k in range(12):
        m = (ang >= binedges[k]) & (ang < binedges[k + 1])
        r_bins.append(_perp_dist(pts[m], a, C).mean() if m.sum() else np.nan)
    r_bins = np.array(r_bins)
    # 极坐标柱状
    ecx, ecy, erad = 320, 200, 120
    ell_bars = []
    for k in range(12):
        angk = np.pi / 2 - (k * 30 + 15) * np.pi / 180
        rr = r_bins[k] * (erad / (r + 6))
        x2 = ecx + rr * np.cos(angk)
        y2 = ecy - rr * np.sin(angk)
        ell_bars.append(f'<line x1="{ecx:.1f}" y1="{ecy:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')

    # --- 锥度：沿 z 分层半径偏差 ---
    zs = pts[:, 2]
    zmin, zmax = zs.min(), zs.max()
    zlayers, zdev = [], []
    for k in range(0, int(zmax - zmin) + 1, 3):
        lo, hi = zmin + k, zmin + k + 3
        m = (zs >= lo) & (zs < hi)
        if m.sum() >= 10:
            zlayers.append((lo + hi) / 2)
            zdev.append(_perp_dist(pts[m], a, C).mean() - r)
    zdev = np.array(zdev)
    zdmax = max(abs(zdev).max(), 0.05)
    zh = 160
    zbase = 160
    zx = lambda zz: 40 + (zz - zlayers[0]) / max(zlayers[-1] - zlayers[0], 1e-6) * 540
    zy = lambda dv: zbase - dv / zdmax * (zh / 2)
    zcandles = []
    for zz, dv in zip(zlayers, zdev):
        zcandles.append(f'<rect x="{zx(zz)-6:.1f}" y="{min(zy(dv),zy(0)):.1f}" width="12" height="{abs(zy(dv)-zy(0)):.1f}"/>')

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>圆柱拟合报告</title>
<style>
:root{{--bg:#0f1420;--card:#1a2233;--line:#2a3550;--txt:#e6ebf5;--mut:#8b98b0;--acc:#4da3ff;--ok:#3ddc84;--warn:#ffb454;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,-apple-system,sans-serif;line-height:1.6}}
.wrap{{max-width:1180px;margin:0 auto;padding:32px 20px 60px}}
h1{{font-size:24px;margin:0 0 4px}}
.sub{{color:var(--mut);margin:0 0 28px;font-size:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}}
.card .k{{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.5px}}
.card .v{{font-size:22px;font-weight:600;margin-top:4px}}
.card .v small{{font-size:13px;color:var(--mut);font-weight:400}}
.sec{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:20px}}
.sec h2{{margin:0 0 12px;font-size:16px}}
.sec p.note{{color:var(--mut);font-size:13px;margin:10px 0 0}}
svg{{max-width:100%;height:auto}}
g.points circle{{fill:#5b8dd6;opacity:.55}}
.fit{{fill:none;stroke:var(--acc);stroke-width:2}}
.center{{fill:none;stroke:var(--ok);stroke-width:1.5}}
.cross{{stroke:var(--ok);stroke-width:2}}
.hbar{{fill:var(--acc);opacity:.7}}
.ellbar{{stroke:var(--acc);stroke-width:2;stroke-linecap:round}}
.candle{{fill:var(--ok);opacity:.8}}
.bar-candle{{fill:var(--warn)}}
.axis{{stroke:var(--line);stroke-width:1}}
.axis text{{fill:var(--mut);font-size:11px}}
.ok{{color:var(--ok)}}.warn{{color:var(--warn)}}.mut{{color:var(--mut)}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line)}}
th{{color:var(--mut);font-weight:500}}
</style></head><body><div class="wrap">
<h1>圆柱拟合报告</h1>
<p class="sub">数据源：pointcloud.ply · {res['n']} 个点 · 高斯-牛顿/LM 最小二乘拟合（5 参数）</p>

<div class="grid">
<div class="card"><div class="k">拟合半径 r</div><div class="v">{r:.4f} <small>mm</small></div></div>
<div class="card"><div class="k">半径 1σ 不确定度</div><div class="v">±{res['sigma_r']:.4f} <small>mm</small></div></div>
<div class="card"><div class="k">轴线与 z 轴夹角</div><div class="v">{res['angle_deg']:.3f}<small>°</small></div></div>
<div class="card"><div class="k">直径 D = 2r</div><div class="v">{2*r:.4f} <small>mm</small></div></div>
</div>

<div class="grid">
<div class="card"><div class="k">径向 RMS（均方根）</div><div class="v" style="color:var(--ok)">{res['rms']:.4f} <small>mm</small></div></div>
<div class="card"><div class="k">径向 MAE（平均绝对）</div><div class="v">{res['mae']:.4f} <small>mm</small></div></div>
<div class="card"><div class="k">最大偏差</div><div class="v" style="color:var(--warn)">{res['maxdev']:.4f} <small>mm</small></div></div>
<div class="card"><div class="k">相对半径误差</div><div class="v">{res['rms']/r*100:.3f}<small>%</small></div></div>
</div>

<div class="sec"><h2>拟合轴线</h2>
<p style="font-family:monospace;font-size:14px">轴方向 a = [{a[0]:+.6f}, {a[1]:+.6f}, {a[2]:+.6f}]<br>轴线上一点 C = [{C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f}] mm</p>
</div>

<div class="sec"><h2>① 径向截面散点 + 拟合圆（投影到垂直轴平面）</h2>
<svg viewBox="0 0 640 640" width="640" height="640">
<line x1="320" y1="0" x2="320" y2="640" class="axis"/><line x1="0" y1="320" x2="640" y2="320" class="axis"/>
<circle cx="320" cy="320" r="{r*scale:.1f}" class="fit"/>
<g class="points">{svg_pts}</g>
<line x1="312" y1="320" x2="328" y2="320" class="cross"/><line x1="320" y1="312" x2="320" y2="328" class="cross"/>
<text x="330" y="316" class="axis">圆心(轴线)</text>
</svg>
<p class="note">蓝点为 3500 个抽样点的径向投影，蓝色圆为拟合圆（半径 {r:.2f} mm）。点越贴合圆环说明圆柱度越好。</p>
</div>

<div class="sec"><h2>② 径向残差分布（点到表面距离 = 距离 - 半径）</h2>
<svg viewBox="0 0 640 200" width="640" height="200">
<line x1="40" y1="140" x2="600" y2="140" class="axis"/>
{''.join(bars)}
<text x="40" y="160" class="axis">{R.min():.2f}</text>
<text x="300" y="160" class="axis">0 mm</text>
<text x="560" y="160" class="axis">{R.max():.2f}</text>
<text x="280" y="20" class="axis">RMS={res['rms']:.3f} mm · 标准差={res['std']:.3f} mm</text>
</svg>
<p class="note">残差越集中、越接近 0，说明圆柱面越规则。</p>
</div>

<div class="sec"><h2>③ 椭圆度：绕轴每 30° 平均半径（极坐标）</h2>
<svg viewBox="0 0 640 400" width="640" height="400">
<circle cx="320" cy="200" r="{erad:.1f}" class="center"/>
{''.join(ell_bars)}
<text x="320" y="380" class="axis">最大-最小 = {r_bins.max()-r_bins.min():.3f} mm（值小→截面接近正圆）</text>
</svg>
</div>

<div class="sec"><h2>④ 锥度：沿 z 各层平均半径偏差</h2>
<svg viewBox="0 0 640 200" width="640" height="200">
<line x1="40" y1="160" x2="600" y2="160" class="axis"/>
{''.join(zcandles)}
<text x="560" y="170" class="axis">z = {zmax:.1f} mm</text>
<text x="40" y="170" class="axis">z = {zmin:.1f} mm</text>
</svg>
<p class="note">各层半径相对拟合半径的偏差（mm）。偏差接近 0 且平直 → 无锥度。</p>
</div>

<div class="sec"><h2>精度结论</h2>
<table>
<tr><th>指标</th><th>数值</th><th>含义</th></tr>
<tr><td>径向 RMS</td><td class="ok">{res['rms']:.4f} mm</td><td>表面点到拟合圆柱的平均偏差（核心精度指标）</td></tr>
<tr><td>最大偏差</td><td>{res['maxdev']:.4f} mm</td><td>最差单点偏差（约 {res['maxdev']/r*100:.1f}% 半径）</td></tr>
<tr><td>半径 1σ</td><td>±{res['sigma_r']:.4f} mm</td><td>半径估计的统计不确定度（点数多所以很小，不代表表面质量）</td></tr>
<tr><td>椭圆度</td><td>{r_bins.max()-r_bins.min():.3f} mm</td><td>截面各方向半径差异（圆度）</td></tr>
<tr><td>锥度</td><td>{2*abs(zdev).max():.3f} mm</td><td>两端直径差（沿轴）</td></tr>
</table>
</div>

</div></body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    """功能：脚本入口，读点云→拟合→打印报告→存 npz→生成 HTML。输入 无 输出 无"""
    pts = np.loadtxt(PTS, skiprows=7)
    res = fit_cylinder(pts)
    report_console(pts, res)

    a, C, r = res['axis'], res['center'], res['radius']
    np.savez_compressed(OUT_NPZ, axis=a, center=C, radius=r, sigma_r=res['sigma_r'],
                        rms=res['rms'], mae=res['mae'], maxdev=res['maxdev'], std=res['std'],
                        res=res['res'])
    print(f"\n已保存拟合结果 -> {OUT_NPZ}")
    gen_html(pts, res, OUT_HTML)
    print(f"已生成可视化报告 -> {OUT_HTML}")


if __name__ == "__main__":
    main()
