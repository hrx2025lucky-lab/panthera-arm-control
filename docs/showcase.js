"use strict";
(() => {
  const data = window.PANTHERA_SHOWCASE;
  const $ = (id) => document.getElementById(id);
  const blue = "#205ce5", amber = "#a96520", teal = "#107267";
  const report = "https://github.com/hrx2025lucky-lab/panthera-arm-control/blob/main/docs/project_summary.md";
  const videos = {
    tall: {file: "panthera_tall_zh.mp4", poster: "panthera_tall_poster.png", title: "七障碍 · 完整抓放", duration: "42.98 s",
      caption: "固定七障碍布局，两个预设规划种子均完成任务；本片展示 seed 30。全臂、夹持特写与仿真腕部相机来自同一时刻的记录。"},
    vision: {file: "panthera_visual_execution_zh.mp4", poster: "panthera_visual_execution_poster.png", title: "腕部视觉 · 完整抓放", duration: "41.10 s",
      caption: "仿真腕部 RGB-D 定位后的成功抓放案例；声明批次为 1/2，四项故障输入均在运动前拒绝。相机画面与机械臂状态使用同一曝光时刻。"}
  };
  let currentVideo = "tall";
  function selectVideo(key, scroll) {
    const v = videos[key];
    if (!v) return;
    const player = $("main-video");
    if (currentVideo !== key) {
      player.pause();
      player.poster = "media/" + v.poster;
      player.querySelector("source").src = "media/" + v.file;
      player.load();
      currentVideo = key;
    }
    player.setAttribute("aria-label", v.title + "视频");
    $("video-label").textContent = v.title;
    $("video-duration").textContent = v.duration + " · 1080p / 50 FPS";
    $("video-caption").textContent = v.caption;
    $("video-download").href = "media/" + v.file;
    $("video-error").hidden = true;
    document.querySelectorAll("[data-video]").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.video === key)));
    if (scroll) $("main").scrollIntoView({block: "start"});
  }
  document.querySelectorAll("[data-video]").forEach(b => b.addEventListener("click", () => selectVideo(b.dataset.video, false)));
  document.querySelectorAll("[data-select-video]").forEach(b => b.addEventListener("click", () => selectVideo(b.dataset.selectVideo, true)));
  $("main-video").addEventListener("error", () => { $("video-error").hidden = false; });
  $("main-video").querySelector("source").addEventListener("error", () => { $("video-error").hidden = false; });

  const plot = $("control-plot");
  const ns = "http://www.w3.org/2000/svg";
  let experiment = "tracking", plotState = null;
  const compactPlot = () => window.matchMedia("(max-width: 480px)").matches;
  const chartWidth = () => compactPlot() ? 360 : 840;
  function svg(tag, attrs, text) {
    const e = document.createElementNS(ns, tag);
    Object.entries(attrs || {}).forEach(([key, value]) => e.setAttribute(key, value));
    if (text !== undefined) e.textContent = text;
    plot.appendChild(e);
    return e;
  }
  function clearPlot(description) {
    plot.replaceChildren();
    plot.setAttribute("viewBox", "0 0 " + chartWidth() + " 330");
    svg("desc", {id: "plot-description"}, description);
  }
  function legend(items) {
    $("legend").replaceChildren();
    items.forEach(item => {
      const span = document.createElement("span");
      span.style.setProperty("--series", item.color);
      span.textContent = item.name;
      $("legend").appendChild(span);
    });
  }
  function metrics(items) {
    $("metrics").replaceChildren();
    items.forEach(([name, value, unit]) => {
      const item = document.createElement("div"); item.className = "metric";
      const label = document.createElement("span"); label.className = "metric-label"; label.textContent = name;
      const number = document.createElement("span"); number.className = "metric-value"; number.textContent = value;
      const suffix = document.createElement("small"); suffix.textContent = unit;
      number.appendChild(suffix); item.append(label, number); $("metrics").appendChild(item);
    });
  }
  function linePlot(times, series, unit, band) {
    clearPlot("保存的仿真采样曲线，横轴为秒，纵轴为 " + unit + "。" + series.map(s => s.name).join("，"));
    const left = compactPlot() ? 48 : 70, top = 25, width = chartWidth() - left - 28, height = 250;
    const values = series.flatMap(s => s.values);
    let low = Math.min(0, ...values), high = Math.max(0, ...values);
    const pad = Math.max((high - low) * .12, .01); low -= pad; high += pad;
    const end = times[times.length - 1];
    const x = t => left + t / end * width;
    const y = v => top + (high - v) / (high - low) * height;
    if (band) {
      svg("rect", {x: x(band[0]), y: top, width: x(band[1]) - x(band[0]), height, fill: "#eef3f8"});
      svg("text", {x: x(band[0]) + 8, y: top + 15, class: "axis-title"}, band[2]);
    }
    for (let i = 0; i <= 4; i++) {
      const value = low + (high - low) * i / 4, py = y(value);
      svg("line", {x1: left, x2: left + width, y1: py, y2: py, stroke: "#dfe7ee"});
      svg("text", {x: left - 12, y: py + 4, "text-anchor": "end"}, value.toFixed(1));
      const time = end * i / 4, px = x(time);
      svg("text", {x: px, y: top + height + 24, "text-anchor": "middle"}, time.toFixed(1));
    }
    svg("line", {x1: left, x2: left + width, y1: y(0), y2: y(0), stroke: "#8fa8bc", "stroke-dasharray": "4 4"});
    svg("text", {x: 16, y: 13, class: "axis-title"}, unit);
    svg("text", {x: left + width, y: 322, "text-anchor": "end", class: "axis-title"}, "时间 / s");
    series.forEach(s => {
      const points = times.map((t, i) => x(t).toFixed(2) + "," + y(s.values[i]).toFixed(2)).join(" ");
      svg("polyline", {points, fill: "none", stroke: s.color, "stroke-width": "2.5", "stroke-linejoin": "round", "vector-effect": "non-scaling-stroke"});
    });
    plotState = {times, series, unit, x, y, top, height};
    $("time-slider").max = String(times.length - 1);
    $("time-slider").value = String(Math.min(Number($("time-slider").value), times.length - 1));
    updateTime();
  }
  function updateTime() {
    if (!plotState || experiment === "adaptive") return;
    const s = plotState, i = Number($("time-slider").value);
    plot.querySelectorAll(".cursor").forEach(e => e.remove());
    svg("line", {class: "cursor", x1: s.x(s.times[i]), x2: s.x(s.times[i]), y1: s.top, y2: s.top + s.height, stroke: "#71899b", "stroke-dasharray": "3 4"});
    $("time-label").textContent = s.times[i].toFixed(3) + " s";
    $("time-slider").setAttribute("aria-valuetext", $("time-label").textContent);
    $("sample-values").replaceChildren();
    s.series.forEach(line => {
      const value = line.values[i];
      svg("circle", {class: "cursor", cx: s.x(s.times[i]), cy: s.y(value), r: 4, fill: "white", stroke: line.color, "stroke-width": 2});
      const label = document.createElement("span");
      label.textContent = line.name + " " + value.toFixed(3) + " " + s.unit;
      label.style.color = line.color; $("sample-values").appendChild(label);
    });
  }
  function renderTracking() {
    const j = Number($("joint-select").value), c = data.control;
    $("plot-title").textContent = "J" + (j + 1) + " · 关节位置误差";
    $("plot-subtitle").textContent = "目标角度 − 实际角度 · mrad";
    const series = [{name: "PD+g", color: amber, values: data.tracking.pd.map(row => row[j])},
      {name: "CTC", color: blue, values: data.tracking.ctc.map(row => row[j])}];
    legend(series); linePlot(data.tracking.time_s, series, "mrad", [0, 1, "起步段"]);
    $("experiment-tag").textContent = "全六轴指标 · 三次重复";
    metrics([["PD + 重力补偿 · 关节 RMSE", c.joint_rmse_mrad.pd_gravity.toFixed(3), "mrad"],
      ["计算力矩 CTC · 关节 RMSE", c.joint_rmse_mrad.computed_torque.toFixed(3), "mrad"]]);
    $("formula").textContent = "PD+g：g(q) + Kp·e + Kd·ė\nCTC：M(q)·aᵣ + 速度项 + g(q)";
    $("experiment-copy").textContent = "相同模型、参考轨迹和力矩限制；使用各自声明的增益。幅值逐轴设置，参考频率 0.5 Hz，统计窗口去除前 1 秒。";
    $("plot-note").textContent = "曲线来自第 1 次运行，每 20 ms 抽取原采样点；未平滑曲线。右侧 RMSE 使用原报告 3 次重复的完整评价窗口。";
    $("experiment-image").href = "media/control_comparison.png";
  }
  function renderImpedance() {
    $("plot-title").textContent = "外力作用下的 TCP 位移";
    $("plot-subtitle").textContent = "基座 Z 方向 · mm";
    const series = [{name: "MuJoCo 回读", color: blue, values: data.impedance.offset_mm},
      {name: "F/K 静态平衡位置", color: teal, values: data.impedance.force_n.map(v => v / 500 * 1000)}];
    legend(series); linePlot(data.impedance.time_s, series, "mm", [.5, 4.5, "施加 −10 N"]);
    $("experiment-tag").textContent = "固定姿态 · 单方向外力";
    metrics([["F/K 静态相对误差", data.control.impedance_f_over_k_relative_error_pct.toFixed(3), "%"], ["理论静态位移", "−20.0", "mm"]]);
    $("formula").textContent = "Δz = Fz / Kz\n−10 N ÷ 500 N/m = −0.020 m";
    $("experiment-copy").textContent = "在 0.5–4.5 秒施加外力，以 3.5–4.5 秒稳态窗口评价。F/K 给出静态平衡位置，动态过程还受阻尼、惯量影响。";
    $("plot-note").textContent = "显示第 1 次运行的原始 TCP 回读，每 20 ms 取样；静态指标来自原报告。移除外力后，位移回到零附近。";
    $("experiment-image").href = "media/control_comparison.png";
  }
  function renderAdaptive() {
    const c = data.adaptive.comparisons[Number($("case-select").value)];
    const labels = ["冻结为零负载", "在线质量自适应", "已知质量参考"];
    const values = [c.rmse_mrad.frozen_zero, c.rmse_mrad.adaptive, c.rmse_mrad.known_mass_oracle];
    const colors = [amber, blue, teal];
    $("plot-title").textContent = (c.plant_mass_kg * 1000).toFixed(0) + " g 负载 · 关节 RMSE";
    $("plot-subtitle").textContent = "相同反馈与参考 · " + c.frequency_hz.toFixed(2) + " Hz";
    clearPlot("该条件下三种负载补偿的关节 RMSE：" + values.map(v => v.toFixed(4)).join("、") + " mrad。");
    legend(labels.map((name, i) => ({name, color: colors[i]})));
    const max = Math.max(...values) * 1.23;
    const left = compactPlot() ? 119 : 190, width = chartWidth() - left - (compactPlot() ? 75 : 115);
    for (let i = 0; i <= 4; i++) {
      const x = left + width * i / 4;
      svg("line", {x1: x, x2: x, y1: 23, y2: 278, stroke: "#e1e8ee"});
      svg("text", {x, y: 302, "text-anchor": "middle"}, (max * i / 4).toFixed(max < 1 ? 3 : 1));
    }
    values.forEach((value, i) => {
      const y = 42 + i * 81, barWidth = Math.max(2, value / max * width);
      svg("text", {x: left - 12, y: y + 24, "text-anchor": "end", class: "axis-title"}, labels[i]);
      svg("rect", {x: left, y, width: barWidth, height: 38, rx: 3, fill: colors[i]});
      svg("text", {x: left + 9 + barWidth, y: y + 24}, value.toFixed(4));
    });
    svg("text", {x: chartWidth() - 18, y: 322, "text-anchor": "end", class: "axis-title"}, "RMSE / mrad");
    $("experiment-tag").textContent = "单质量参数 · 六组条件";
    metrics([["在线自适应 · 关节 RMSE", c.rmse_mrad.adaptive.toFixed(4), "mrad"],
      [c.adaptive_reduction_pct >= 0 ? "相对冻结估计降低" : "相对冻结估计增加", Math.abs(c.adaptive_reduction_pct).toFixed(2), "%"]]);
    $("formula").textContent = "在线更新负载质量估计\n反馈与参考保持一致";
    $("experiment-copy").textContent = "36 次对照，共 216,000 个物理步。质心和单位质量惯量已知；这里只适应负载质量，不等同于完整惯性参数辨识。";
    $("plot-note").textContent = "切换菜单查看全部六组预定条件。图使用原始报告的组内汇总值，包含零负载对照；每组横轴按实际数值缩放。";
    $("experiment-image").href = "media/payload_adaptive_comparison.png";
  }
  function render() {
    $("joint-control").hidden = experiment !== "tracking";
    $("time-control").hidden = experiment === "adaptive";
    $("mass-control").hidden = experiment !== "adaptive";
    document.querySelectorAll("[data-experiment]").forEach(a => a.setAttribute("aria-current", String(a.dataset.experiment === experiment)));
    if (experiment === "tracking") renderTracking();
    else if (experiment === "impedance") renderImpedance();
    else renderAdaptive();
  }
  function fromHash(scroll) {
    const selected = location.hash.replace("#control-", "");
    if (["tracking", "impedance", "adaptive"].includes(selected)) {
      experiment = selected; render();
      if (scroll) $("control").scrollIntoView({block: "start"});
    }
  }
  try {
    if (!data || !data.tracking.time_s.length) throw new Error("missing_showcase_data");
    data.adaptive.comparisons.forEach((c, i) => {
      const option = document.createElement("option"); option.value = String(i);
      option.textContent = (c.plant_mass_kg * 1000).toFixed(0) + " g / " + c.frequency_hz.toFixed(2) + " Hz";
      $("case-select").appendChild(option);
    });
    $("case-select").value = "2";
    $("joint-select").addEventListener("change", render);
    $("case-select").addEventListener("change", render);
    $("time-slider").addEventListener("input", updateTime);
    window.addEventListener("resize", render);
    window.addEventListener("hashchange", () => fromHash(false));
    render(); fromHash(false);
    document.documentElement.dataset.showcaseReady = "true";
  } catch (error) {
    $("plot-title").textContent = "曲线数据未能加载";
    $("plot-subtitle").textContent = "可通过右侧链接查看完整实验报告。";
    $("time-control").hidden = true;
    $("experiment-image").textContent = "打开实验报告 ↗";
    $("experiment-image").href = report;
    document.documentElement.dataset.showcaseReady = "false";
  }
})();
