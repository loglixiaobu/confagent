#!/usr/bin/env python
"""Build CPU-only presentation artifacts from the published numeric snapshot.

Run from any directory. The snapshot contains the frozen values used in the
report; unpublished collection logs are not required. No model, client,
GPU library, network request or experiment launcher is used.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text
import numpy as np

OUT = Path(__file__).resolve().parent
DATE = "2026-10-20"
C = dict(blue="#3868c9", teal="#008c82", gold="#c18a2c", red="#d55e68",
         gray="#8592a5", ink="#243448", light="#eef3f8", purple="#7866c6")
NAMES = dict(logprob="Token logprob", self_report="Self-report", consistency="Consistency",
             raw_consistency="Raw consistency", p_correct="Frozen calibration", training_prior="Training prior",
             oof_calibrated="Task-grouped OOF", training_fold_prior="Fold training prior")
ERROR_NAMES = {
    "fabricated_travel_inputs": "Invented travel inputs", "fabricated_meeting_location": "Invented meeting location",
    "unsupported_salary_period_conversion": "Unsupported salary\nperiod conversion", "fabricated_checkout_date": "Invented checkout date",
    "unverified_coordinate_sign_change": "Unverified coordinate\nsign change", "kilometers_not_converted_to_meters": "km-to-m conversion\nnot applied",
    "calculation_before_all_counts_observed": "Calculation before\nall counts were retrieved", "humidity_unit_not_resolved": "Unresolved humidity unit",
    "search_no_result": "Search returned no result", "retrieval_capability_mismatch": "Wrong capability retrieved",
}
FIGURES = [
    ("01_labels_and_task_outcomes", "Define correctness before measuring performance", "85 call labels changed. Evaluate local calls and complete tasks separately."),
    ("02_error_types", "Unsupported inputs and missing dependencies drive errors", "24/40 incorrect calls invent missing inputs. Counts include retries within a task."),
    ("03_signal_discrimination", "Consistency is the strongest signal on these development tasks", "The paired AUROC difference supports consistency over logprob;\nthe self-report advantage remains uncertain."),
    ("04_verification_budget", "Rank steps for review; correction gains remain untested", "Use the same strict budget of 39/198 steps.\nSeparate random-tie expectations from fixed-seed selections."),
    ("05_development_calibration", "Internal calibration improves probability error, not ranking", "Task-grouped five-fold OOF reduces Brier error; the prior still has lower ECE."),
    ("06_new_task_validation", "New tasks expose probability shift and final-output failures", "Only one incorrect call was observed. Calibrated Brier does not beat the prior."),
    ("07_stopped_experiments", "Stopped experiments clarified the next research steps", "Reference prefixes lacked errors; thinking probes were slow;\nsynthetic stress data had no independent holdout evaluation."),
    ("08_inference_cost", "Review budgets and scoring costs are different", "Every new candidate incurs three probes. Probe tokens equal 102.8% of primary tokens."),
    ("09_method_and_scope", "A callable prototype is ready; the research loop is incomplete", "Pre-call probabilities and review flags work.\nStopping confidence, actual correction, and experience updates remain untested."),
]
FIGURE_TEXTS = {}
CJK_TEXT = re.compile(r"[\u2e80-\u303f\u3040-\u30ff\u3100-\u31bf\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\U00020000-\U0002fa1f]")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def budget(rows, signal, n=39, seed=20261019):
    # Mirror calibrate_consistency.budget_selection exactly: the frozen input
    # row order and NumPy generator are both part of the tie policy.
    values = np.asarray([r[signal] for r in rows], dtype=float)
    permutation = np.random.default_rng(seed).permutation(len(rows))
    tie_rank = np.empty(len(rows), dtype=int)
    tie_rank[permutation] = np.arange(len(rows))
    ordered = [rows[int(i)] for i in np.lexsort((tie_rank, values))]
    actual = sum(1 - r["y"] for r in ordered[:n])
    remaining, expected = n, 0.
    for value in sorted({r[signal] for r in ordered}):
        group = [r for r in ordered if r[signal] == value]
        take = min(remaining, len(group))
        expected += take / len(group) * sum(1 - r["y"] for r in group)
        remaining -= take
        if not remaining:
            break
    errors = sum(1-r["y"] for r in ordered)
    return dict(n=len(rows), budget=n, selected_fraction=n/len(rows), errors=errors,
                captured_fixed_seed=actual, capture_fixed_seed=actual/errors,
                captured_tie_expectation=expected, capture_tie_expectation=expected/errors, seed=seed,
                tie_method="NumPy default_rng permutation; frozen joined_scores row order")


def style():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(font)
        bold=font.with_name("NotoSansCJK-Bold.ttc")
        if bold.exists(): font_manager.fontManager.addfont(bold)
        family=font_manager.FontProperties(fname=font).get_name()
    else:
        family="sans-serif"
    plt.rcParams.update({"font.family":family, "font.size":12, "axes.titlesize":14,
        "axes.labelsize":12, "axes.spines.top":False, "axes.spines.right":False,
        "axes.labelcolor":C["ink"], "text.color":C["ink"], "xtick.color":C["ink"],
        "ytick.color":C["ink"], "axes.unicode_minus":False, "figure.facecolor":"white",
        "savefig.facecolor":"white", "pdf.fonttype":3, "ps.fonttype":3, "svg.fonttype":"path"})


def canvas(title, subtitle, n=2, ratios=None):
    fig, axes=plt.subplots(1,n,figsize=(15.8,7.8),gridspec_kw={"width_ratios":ratios or [1]*n}, squeeze=False)
    fig.subplots_adjust(left=.08,right=.96,bottom=.21,top=.78,wspace=.36)
    fig.text(.055,.945,title,fontsize=22,weight="bold",va="top")
    fig.text(.055,.875,subtitle,fontsize=12,color=C["gray"],va="top")
    return fig,axes[0]


def save(fig, stem, note, source):
    fig.text(.055,.10,note,fontsize=11,va="top")
    fig.text(.055,.035,"Sources: "+source+"  |  As of "+DATE,fontsize=8.5,color=C["gray"],va="bottom")
    save_english_figure(fig, stem)


def save_english_figure(fig, stem):
    # This helper is only for chart PNG/SVG assets, not Chinese PDF pages.
    fig.canvas.draw()
    texts = [artist.get_text() for artist in fig.findobj(match=Text) if artist.get_text().strip()]
    invalid = [text for text in texts if CJK_TEXT.search(text)]
    if invalid:
        raise ValueError(f"Non-English chart text in {stem}: {invalid!r}")
    FIGURE_TEXTS[stem] = texts
    write_json(OUT/"data/figure_texts.json", FIGURE_TEXTS)
    for suffix in ["png","svg"]:
        fig.savefig(OUT/"figures"/(stem+"."+suffix),dpi=210)
    plt.close(fig)


def figure_labels(d):
    fig,ax=canvas("01  Relabeling changes measurement, not model ability", "Same 50 tasks and 198 calls; 18 saved no-tool outputs retain separate ending labels.")
    fig.subplots_adjust(left=.12)
    matrix=np.array([[80,7],[78,33]])
    ax[0].imshow(matrix,cmap="Blues",vmin=0,vmax=80)
    for (i,j),value in np.ndenumerate(matrix):ax[0].text(j,i,str(value),ha="center",va="center",fontsize=25,color="white" if value>50 else C["ink"])
    ax[0].set(xticks=[0,1],xticklabels=["Revised: correct","Revised: incorrect"],yticks=[0,1],yticklabels=["Old correct: 87","Old incorrect: 111"],title="78 negatives revised; 7 positives revised")
    vals=[158/198,14/50,9/18];names=["Call correctness\n158 / 198","Task success\n14 / 50","Valid saved endings\n9 / 18"]
    bars=ax[1].bar(range(3),vals,color=[C["teal"],C["blue"],C["gold"]],width=.6)
    ax[1].set(xticks=range(3),xticklabels=names,ylim=(0,1.1),ylabel="Fraction",title="Keep the three populations separate")
    for b,v in zip(bars,vals):ax[1].text(b.get_x()+b.get_width()/2,v+.035,f"{v:.1%}",ha="center",fontsize=18,weight="bold")
    save(fig,FIGURES[0][0],"85/198 call labels changed. Logprob AUROC 0.5401 → 0.6351 reflects relabeling; the model was not updated.\nProtocol-based Codex review; no independent human double annotation or annotator-agreement study.","legacy_binary_v2/label_summary.json; logprob_metrics.json")


def figure_errors(d):
    reasons=sorted(d["binary"]["error_reasons"],key=d["binary"]["error_reasons"].get,reverse=True)
    fig,ax=canvas("02  Unsupported inputs dominate the reviewed errors", "24 of 40 incorrect calls invent missing travel inputs, meeting locations, or checkout dates.",ratios=[1.25,1])
    fig.subplots_adjust(left=.20, wspace=.42)
    y=np.arange(len(reasons));counts=[d["binary"]["error_reasons"][r] for r in reasons]
    ax[0].barh(y,counts,color=[C["red"] if r.startswith("fabricated") else C["blue"] for r in reasons],height=.65)
    ax[0].set(yticks=y,yticklabels=[ERROR_NAMES[r] for r in reasons],xlabel="Incorrect calls",xlim=(0,12),title="Error calls by cause\n(including retries)");ax[0].invert_yaxis()
    for i,v in enumerate(counts):ax[0].text(v+.15,i,str(v),va="center")
    groups={r["group"]:r for r in d["error_type_diagnostics"]}
    for signal,color in [("self_report",C["gold"]),("consistency",C["teal"])]:
        values=[groups[r]["signals"][signal]["mean"] for r in reasons]
        ax[1].scatter(values,y,color=color,s=50,label=NAMES[signal],zorder=3)
    ax[1].set(yticks=y,yticklabels=[],xlim=(-.05,1.05),xlabel="Mean raw score\n(not a calibrated probability)",title="Incorrect calls can still\nreceive high scores");ax[1].invert_yaxis();ax[1].grid(axis="y",alpha=.2);ax[1].legend(loc="lower right",fontsize=10)
    save(fig,FIGURES[1][0],"Five invented-checkout errors: mean self-report 0.92 and consistency 0.80. Repeated errors can remain consistent.\nThe 40 incorrect steps come from 16 tasks; they are not 40 independent error tasks.","legacy_binary_v2/label_summary.json; new_signals_fast_v1/new_signal_metrics.json")


def figure_signals(d):
    fig,ax=canvas("03  Consistency outperforms logprob on these legacy tasks", "Same binary labels for all 198 steps; 1,000 task-cluster bootstrap replicates preserve within-task dependence.")
    common=d["signals"]["all_calls"];order=["logprob","self_report","consistency"]
    colors=[C["blue"],C["gold"],C["teal"]]
    for y,(s,color) in enumerate(zip(order,colors)):
        v=common["signals"][s];p=v["AUROC"];lo,hi=v["AUROC_bootstrap"]["ci95"]
        ax[0].errorbar(p,y,xerr=[[p-lo],[hi-p]],fmt="o",capsize=5,color=color,markersize=10,lw=3)
        ax[0].text(p,y+.20,f"{p:.3f} [{lo:.3f}, {hi:.3f}]",ha="center",fontsize=11)
    ax[0].axvline(.5,color=C["gray"],ls="--");ax[0].set(yticks=range(3),yticklabels=[NAMES[s] for s in order],xlim=(.43,1),ylim=(-.5,2.6),xlabel="AUROC (higher is better)",title="Primary analysis: 158 correct, 40 incorrect")
    for y,s in enumerate(order[1:]):
        v=common["paired_AUROC_difference_vs_logprob"][s];p=v["point"];lo,hi=v["ci95"];color=colors[y+1]
        ax[1].errorbar(p,y,xerr=[[p-lo],[hi-p]],fmt="o",capsize=5,color=color,markersize=10,lw=3)
        ax[1].text(p,y+.20,f"Δ {p:+.3f} [{lo:+.3f}, {hi:+.3f}]",ha="center",fontsize=11)
    ax[1].axvline(0,color=C["gray"],ls="--");ax[1].set(yticks=[0,1],yticklabels=[NAMES[s] for s in order[1:]],xlim=(-.1,.43),ylim=(-.5,1.6),xlabel="Paired AUROC difference vs. logprob",title="Self-report CI crosses zero;\nconsistency CI does not")
    save(fig,FIGURES[2][0],"After semantic deduplication (130 steps), consistency AUROC is 0.8793. The pre-mock-error paired CI crosses zero.\nNew scores use reconstructed contexts and no-thinking probes; legacy logprob uses thinking, so mechanisms are confounded.","new_signals_fast_v1/new_signal_metrics.json / cohorts")


def capture_curve(rows,signal):
    xs=[0.];ys=[0.];used=caught=0.;errors=sum(1-r["y"] for r in rows)
    for value in sorted({r[signal] for r in rows}):
        group=[r for r in rows if r[signal]==value];used+=len(group);caught+=sum(1-r["y"] for r in group)
        xs.append(used/len(rows));ys.append(caught/errors)
    return xs,ys


def figure_budget(d):
    fig,ax=canvas("04  Low scores prioritize review, but scoring has a cost", "Same strict budget: floor(0.2 × 198) = 39 steps (19.7%). Show random-tie expectations and fixed-seed selections separately.")
    for s,color in [("logprob",C["blue"]),("self_report",C["gold"]),("consistency",C["teal"])]:
        x,y=capture_curve(d["rows"],s);ax[0].plot(x,y,label=NAMES[s],color=color,lw=2.5)
    ax[0].plot([0,1],[0,1],"--",color=C["gray"],label="Random review")
    ax[0].axvline(39/198,color=C["gray"],ls=":")
    ax[0].scatter([65/198],[35/40],s=85,color=C["teal"],zorder=5)
    ax[0].annotate("raw < 1\nReview 65 steps;\nflag 35 errors",(65/198,35/40),xytext=(.52,.66),arrowprops=dict(arrowstyle="->",color=C["teal"]),fontsize=11)
    ax[0].set(xlim=(0,1),ylim=(0,1.05),xlabel="Fraction of steps selected for review",ylabel="Fraction of errors flagged",title="Interpolate tied scores by random-order expectation");ax[0].legend(loc="lower right",fontsize=10)
    order=["logprob","self_report","consistency"];x=np.arange(3)
    expected=[d["strict_budget"][s]["capture_tie_expectation"] for s in order]
    actual=[d["strict_budget"][s]["capture_fixed_seed"] for s in order]
    ax[1].bar(x,expected,color=[C["blue"],C["gold"],C["teal"]],width=.55,label="Random-tie expectation")
    ax[1].scatter(x,actual,color=C["ink"],marker="D",s=55,label="Fixed-seed selection",zorder=4)
    for i,(e,a) in enumerate(zip(expected,actual)):ax[1].text(i,max(e,a)+.045,f"Expected {e:.1%}\nSelected {a:.1%}",ha="center",fontsize=11)
    ax[1].axhline(39/198,color=C["gray"],ls="--",label="Random-budget expectation: 19.7%")
    ax[1].set(xticks=x,xticklabels=[NAMES[s] for s in order],ylim=(0,1),ylabel="Fraction of 40 errors flagged",title="Strict budget: 39 steps");ax[1].legend(loc="upper left",fontsize=9)
    save(fig,FIGURES[3][0],"Consistency: the fixed seed selects 39 steps and flags 26/40 errors (65.0%); the random-tie expectation is 67.65%.\nThe earlier 68.69% used an expected budget of 39.6 steps. These are allocation simulations; no verification or correction ran.","joined_scores.csv (frozen); data/strict_verification_budget.csv; calibration_metrics.json")


def figure_calibration(d):
    fig,ax=canvas("05  Internal calibration improves probabilities, not ranking", "Fixed five-fold StratifiedGroupKFold on 47 tasks with calls; one-variable sigmoid, C=1, no hyperparameter search.",3,[1.6,1,1])
    order=["raw_consistency","oof_calibrated","training_fold_prior"];colors=[C["purple"],C["teal"],C["gray"]]
    for s,color in zip(order,colors):
        bins=d["calibration"]["metrics"][s]["bins"]
        ax[0].plot([b["confidence"] for b in bins],[b["accuracy"] for b in bins],"o-",color=color,label=NAMES[s],lw=2)
    ax[0].plot([0,1],[0,1],"--",color=C["gray"]);ax[0].set(xlim=(0,1),ylim=(0,1.05),xlabel="Predicted probability of correctness",ylabel="Observed call correctness",title="OOF reliability\n(5 equal-width bins)");ax[0].legend(fontsize=9,loc="lower right")
    for a,metric in zip(ax[1:],["Brier","ECE"]):
        values=[d["calibration"]["metrics"][s][metric] for s in order]
        a.bar(range(3),values,color=colors,width=.6)
        for i,v in enumerate(values):a.text(i,v+.006,f"{v:.4f}",ha="center",fontsize=11)
        a.set(xticks=range(3),xticklabels=["Raw","OOF\ncalibrated","Fold\nprior"],ylim=(0,max(values)*1.3),title=metric+": lower is better")
    save(fig,FIGURES[4][0],"Brier: 0.1235 → 0.0917; ECE: 0.1347 → 0.0674, but prior ECE is lower (0.0411). ECE alone is insufficient.\nOOF AUROC: 0.8812 → 0.8479; fold-specific maps change pooled ranks. A deployed positive-slope sigmoid preserves ranking.","confidence_minimal_v1/calibration_metrics.json; internal evaluation, not an independent external test")


def figure_new(d):
    fig,ax=canvas("06  New-task integration works; probability transfer is unproven", "30 new tasks: 26 with calls and 4 without. With only one incorrect call, AUROC 0.9286 cannot establish stable generalization.",3,[1.05,1.25,1.1])
    vals=[49/50,22/30]
    for y,(v,label) in enumerate(zip(vals,["Correct calls: 49/50","Valid task deliveries: 22/30"])):
        ax[0].barh(y,v,color=C["teal"]);ax[0].barh(y,1-v,left=v,color=C["red"])
        ax[0].text(.5,y,f"{v:.1%}",ha="center",va="center",color="white",weight="bold",fontsize=18)
    ax[0].set(yticks=[0,1],yticklabels=["Calls","Tasks"],xlim=(0,1),ylim=(-.7,1.7),xlabel="Fraction within each population",title="49 correct calls ≠\nonly one task failure")
    ax[0].text(0,1.4,"22 valid deliveries\n6 goals not completed\n2 invalid executions / final answers",fontsize=10.5,va="bottom")
    values=d["new"]["cohorts"]["matched_all_signals"]["signals"]
    order=["raw_consistency","p_correct","training_prior","logprob"];v=[values[s]["Brier"] for s in order]
    ax[1].bar(range(4),v,color=[C["purple"],C["teal"],C["gray"],C["blue"]])
    for i,a in enumerate(v):ax[1].text(i,a+.008,f"{a:.4f}",ha="center",fontsize=11)
    ax[1].set(xticks=range(4),xticklabels=["Raw\nconsistency","Frozen\ncalibration","Constant\nprior","Token\nlogprob"],ylim=(0,.205),title="New-task Brier:\ncalibration does not beat the prior")
    old_zero=[r for r in d["rows"] if r["consistency"]==0];a=sum(r["y"] for r in old_zero)/len(old_zero)
    b=next(x for x in values["p_correct"]["reliability_bins"] if x["low"]==.4)
    ax[2].bar([0,1],[a,b["accuracy"]],color=[C["gray"],C["teal"]],width=.55)
    ax[2].axhline(1/(1+math.exp(-d["deploy"]["intercept"])),color=C["red"],ls="--",label="Frozen model: p≈0.406")
    for i,(y,text) in enumerate([(a,"15/49 correct"),(b["accuracy"],"7/8 correct")]):ax[2].text(i,y+.045,text,ha="center",fontsize=12)
    ax[2].set(xticks=[0,1],xticklabels=["Legacy: raw=0","New tasks: raw=0"],ylim=(0,1.1),title="Correctness shifts\nat the same raw score");ax[2].legend(loc="upper left",fontsize=9)
    save(fig,FIGURES[5][0],"The rule flags 11/50 calls: the sole error plus 10 correct calls. Flagging 1/1 does not establish stable 100% detection.\nAt 98% call correctness, constant high scores can have low Brier. Success rates across different tasks do not prove capability gains.","confidence_minimal_v1/validation_metrics.json; validation_task_outcomes.jsonl; frozen legacy step scores")


def figure_stops(d):
    fig,ax=canvas("07  Stopped experiments helped narrow the research scope", "These diagnostics or incomplete pilots are not complete natural-task tests and must not be added to the main 198 steps.",3,[1.4,1,1])
    for i,part in enumerate(["dev","test"]):
        r=d["prefix"][part];ax[0].bar(i,r["correct"],color=C["teal"]);ax[0].bar(i,r["ungraded"],bottom=r["correct"],color=C["gray"])
        ax[0].text(i,r["candidate_count"]+2,f"{r['correct']} correct / 0 errors\n{r['ungraded']} ungraded",ha="center",fontsize=12)
    ax[0].set(xticks=[0,1],xticklabels=["Reference-prefix\ndevelopment","Incomplete reference-\nprefix test"],ylim=(0,90),ylabel="Candidates",title="Correct prefixes were too easy\nto train a discriminator")
    pilot=d["thinking_pilot"]
    ax[1].barh([0],[198],height=.22,color=C["light"]);ax[1].barh([0],[pilot["scored_steps"]],height=.22,color=C["gold"])
    ax[1].set(xlim=(0,198),ylim=(-.4,1.7),yticks=[],xlabel="Target legacy steps",title="Slow thinking probes:\nstopped early")
    ax[1].text(0,.9,f"Only {pilot['scored_steps']}/198 steps scored\n{pilot['request_attempts']} requests saved\nExcluded from the fast run",fontsize=12,va="center")
    stress=d["stress"];ax[2].bar([0,1],[stress["pairs_prepared"],stress["pairs_scored"]],color=[C["gray"],C["purple"]])
    ax[2].set(xticks=[0,1],xticklabels=["Constructed\nperturbation pairs","Scored\npairs"],ylim=(0,24),title="Synthetic perturbations\nare not natural errors")
    for i,v in enumerate([stress["pairs_prepared"],stress["pairs_scored"]]):ax[2].text(i,v+.7,str(v),ha="center",fontsize=16)
    save(fig,FIGURES[6][0],"Reference-prefix tests covered 9/20 planned tasks and 20 candidates, with no negative examples for discrimination.\n17 of 18 synthetic pairs were scored and fitted, with no independent stress holdout. Training fit does not establish natural-task generalization.","confidence_v1/prefix_diagnostic_summary.json; stress_frozen_spec.json; new_signals_v1/")


def figure_cost(d):
    fig,ax=canvas("08  Three probes cost extra: review budget is not inference cost", "Show actual requests and tokens for each experiment. Different tasks and functions prevent direct efficiency comparisons.")
    old=d["signal_cost"];new=d["new"]["run_diagnostics"]["model_cost"]["by_kind"]
    values=[[0,new["primary"]["requests"]],[198,0],[594,new["confidence_probe"]["requests"]]]
    totals=[sum(x[i] for x in values) for i in range(2)]
    tokens=[[0,new["primary"]["total_tokens"]],[157098,0],[487913,new["confidence_probe"]["total_tokens"]]]
    labels=["Legacy 198 steps:\nnew-signal screening","30 new tasks:\nfull-trajectory integration"]
    for a,data,scale,ylab in [(ax[0],values,1,"Model requests"),(ax[1],tokens,1000,"Actual tokens (thousands)")]:
        bottom=np.zeros(2)
        for v,color,label in zip(data,[C["blue"],C["gold"],C["teal"]],["Primary trajectory","Self-report probes","Consistency probes"]):
            a.bar([0,1],np.array(v)/scale,bottom=bottom,color=color,label=label,width=.6);bottom+=np.array(v)/scale
        for i,v in enumerate(bottom):a.text(i,v+max(bottom)*.025,f"{v:,.1f}" if scale>1 else f"{int(v):,}",ha="center",fontsize=16)
        a.set(xticks=[0,1],xticklabels=labels,ylim=(0,max(bottom)*1.22),ylabel=ylab)
    ax[0].legend(loc="upper right",fontsize=10)
    save(fig,FIGURES[7][0],"New trajectories: 80 primary requests / 89,869 tokens + 150 probes / 92,383 tokens = 230 requests / 182,252 tokens.\nAll 50 candidates incur three probes before selection. Verification and correction were not run; their extra costs are not included.","new_signals_fast_v1/new_signal_metrics.json / collection_diagnostics.cost; validation_metrics.json / model_cost")


def figure_method(d):
    fig=plt.figure(figsize=(15.8,7.8));a=fig.add_axes([0,0,1,1]);a.set(xlim=(0,1),ylim=(0,1));a.axis("off")
    a.text(.055,.95,"09  A minimal prototype: pre-call score → probability → review flag",fontsize=22,weight="bold",va="top")
    a.text(.055,.88,"Qwen3-8B: thinking generates the candidate; three probes disable thinking. Current tool results and labels are excluded.",fontsize=12,color=C["gray"])
    boxes=[("Full pre-call state\nRequest + prior results",.05), ("Candidate tool call\nNot yet executed",.235), ("3 independent draws\nCompare locally",.42), ("Frozen sigmoid\nEstimate correctness",.605), ("Verification flag\nraw < 1 or missing",.79)]
    for i,(text,x) in enumerate(boxes):
        a.add_patch(patches.FancyBboxPatch((x,.62),.16,.15,boxstyle="round,pad=0.012",facecolor=C["light"],edgecolor=C["teal"],lw=1.7))
        a.text(x+.08,.695,text,ha="center",va="center",fontsize=13)
        if i<4:a.annotate("",(x+.182,.695),(x+.162,.695),arrowprops=dict(arrowstyle="->",lw=2,color=C["teal"]))
    a.text(.06,.51,"Mapping",fontsize=16,weight="bold")
    for x,raw in zip([.24,.42,.60,.78],[0,1/3,2/3,1]):
        p=1/(1+math.exp(-(d["deploy"]["intercept"]+d["deploy"]["coefficient"]*raw)))
        a.text(x,.51,f"{round(raw*3)}/3 agree → {p:.1%}",ha="center",fontsize=14,color=C["teal"])
    for y,title,text,color in [(.35,"Implemented","Full input logs; separate binary labels; task-grouped OOF; frozen model; new-task integration",C["teal"]),(.23,"To validate","Confidence for no-tool endings; new errors from the same distribution; equal-cost correction gains",C["gold"]),(.11,"Next phase","Validate reusable experience; update memory or policy; test whether recursive changes improve capability",C["gray"])]:
        a.text(.06,y,title,fontsize=16,weight="bold",color=color);a.text(.20,y,text,fontsize=12)
    a.text(.055,.025,"Local call reliability is not task-success probability. Agreement can still be wrong. Sources: confidence_runtime.py; calibrator.json  |  "+DATE,fontsize=9,color=C["gray"])
    save_english_figure(fig, FIGURES[8][0])


def make_slides(d):
    with PdfPages(OUT/"中期汇报.pdf",metadata={"Title":"ConfAgent 阶段性实验汇报", "Author":"ConfAgent", "Subject":"截至 "+DATE+" 的实验结果、失败诊断和研究状态"}) as pdf:
        fig=plt.figure(figsize=(16,9));a=fig.add_axes([0,0,1,1]);a.axis("off")
        a.text(.06,.90,"ConfAgent",fontsize=44,weight="bold",color=C["teal"])
        a.text(.06,.80,"置信度驱动的智能体递归自我改进",fontsize=28,weight="bold")
        a.text(.06,.72,"阶段性实验汇报  ·  "+DATE,fontsize=18,color=C["gray"])
        for x,big,small in [(.06,"50 / 198","旧主任务 / 调用，反复重评不重复计数"),(.37,"30 / 50","新增任务 / 调用，冻结校准器验证"),(.68,"阶段一原型","概率输出已接通，纠错收益尚未验证")]:
            a.add_patch(patches.FancyBboxPatch((x,.43),.26,.19,boxstyle="round,pad=.015",facecolor=C["light"],edgecolor="none"))
            a.text(x+.02,.55,big,fontsize=29,weight="bold");a.text(x+.02,.47,small,fontsize=11)
        a.text(.06,.29,"核心发现：标签口径会改变结论；一致性有开发集价值；内部校准改善概率误差。",fontsize=18)
        a.text(.06,.20,"核心限制：新域错误过少，校准未胜常数基线；尚无实际纠错与递归能力提升证据。",fontsize=18,color=C["red"])
        a.text(.06,.07,"全部图表由冻结结果离线生成；未新增模型请求。完整说明、数据和代码在同目录。",fontsize=12,color=C["gray"])
        pdf.savefig(fig);plt.close(fig)
        for i,(stem,title,takeaway) in enumerate(FIGURES,2):
            fig=plt.figure(figsize=(16,9));a=fig.add_axes([.015,.10,.97,.88]);a.axis("off")
            a.imshow(plt.imread(OUT/"figures"/(stem+".png")))
            takeaway="\n".join(textwrap.fill(line,width=108) for line in takeaway.splitlines())
            fig.text(.055,.055,takeaway,fontsize=13,color=C["ink"],va="center")
            fig.text(.95,.025,f"{i}/11",fontsize=10,color=C["gray"],ha="right")
            pdf.savefig(fig);plt.close(fig)
        fig=plt.figure(figsize=(16,9));a=fig.add_axes([0,0,1,1]);a.axis("off")
        a.text(.06,.88,"下一步：先验证概率与检查的价值，再推进递归改进",fontsize=28,weight="bold")
        lines=[("1  数据与标签","保留自然连续轨迹，补足同分布错误；复核结束事件，避免 mock 覆盖限制充当错误真值。"),
               ("2  置信度验证","冻结评分协议和校准器，在新的任务组检查 Brier、排序、覆盖率及错误类型。"),
               ("3  预算实验","把评分与验证成本一起计入，与随机检查、全量检查比较实际纠错和成功率。"),
               ("4  递归改进","只把独立核验有效的经验纳入更新，并保留未参与更新的任务检验能力提升。")]
        for y,(title,text) in zip([.68,.50,.32,.14],lines):
            a.text(.06,y,title,fontsize=22,color=C["teal"],weight="bold");a.text(.06,y-.075,text,fontsize=17)
        pdf.savefig(fig);plt.close(fig)


def inline(text):
    text=html.escape(text)
    text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text)
    return text


def render_markdown(text):
    output=[];paragraph=[];in_list=False;in_table=False;in_code=False;code=[]
    def flush():
        if paragraph:output.append("<p>"+inline(" ".join(paragraph))+"</p>");paragraph.clear()
    def close():
        nonlocal in_list,in_table
        if in_list:output.append("</ul>");in_list=False
        if in_table:output.append("</tbody></table></div>");in_table=False
    for line in text.splitlines():
        if line.startswith("```"):
            flush();close()
            if in_code:output.append("<pre><code>"+html.escape("\n".join(code))+"</code></pre>");code=[]
            in_code=not in_code;continue
        if in_code:code.append(line);continue
        if not line.strip():flush();close();continue
        image=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line.strip())
        if image:
            flush();close();path=(OUT/image[2]).resolve()
            if path.is_file() and OUT in path.parents:
                encoded=base64.b64encode(path.read_bytes()).decode()
                output.append('<figure><img alt="'+html.escape(image[1])+'" src="data:image/png;base64,'+encoded+'"><figcaption>'+inline(image[1])+"</figcaption></figure>")
            continue
        if line.startswith("#"):
            flush();close();n=min(len(line)-len(line.lstrip("#")),6);output.append(f"<h{n}>"+inline(line[n:].strip())+f"</h{n}>");continue
        if line.strip().startswith("|"):
            flush()
            cells=[x.strip() for x in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r':?-{2,}:?',x) for x in cells):continue
            if not in_table:close();output.append('<div class="table"><table><tbody>');in_table=True
            output.append("<tr>"+"".join("<td>"+inline(x)+"</td>" for x in cells)+"</tr>");continue
        if line.startswith("- ") or re.match(r'^\d+\. ',line):
            flush()
            if not in_list:close();output.append("<ul>");in_list=True
            output.append("<li>"+inline(re.sub(r'^(?:- |\d+\. )','',line))+"</li>");continue
        if line.strip()=="---":flush();close();output.append("<hr>");continue
        close();paragraph.append(line.lstrip("> "))
    flush();close();return "\n".join(output)


def make_html():
    report=OUT/"中期研究报告.md"
    if not report.exists():return False
    text=report.read_text(encoding="utf-8")
    missing=[(stem,title) for stem,title,_ in FIGURES if stem+".png" not in text]
    if missing:text+="\n\n## 补充图表\n\n"+"\n\n".join(f"![{title}](figures/{stem}.png)" for stem,title in missing)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ConfAgent 中期研究报告</title><style>
body{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;color:#243448;background:#f2f5f9;margin:0;line-height:1.8}main{max-width:1120px;margin:32px auto;padding:44px 54px;background:white;box-shadow:0 4px 30px #17304d0c}h1{font-size:30px;border-bottom:4px solid #008c82;padding-bottom:20px}h2{margin-top:44px;color:#006c66}h3{margin-top:25px}p{margin:14px 0}a{color:#2262a5}strong{color:#004f4a}code{font-family:monospace;background:#eef3f8;padding:2px 5px;border-radius:4px}pre{background:#eef3f8;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere}pre code{padding:0}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td{border-bottom:1px solid #dfe7ef;padding:8px 12px;vertical-align:top}tr:first-child{background:#eef3f8;font-weight:600}figure{margin:28px -12px;break-inside:avoid}img{width:100%;height:auto}figcaption{font-size:13px;text-align:center;color:#5e6b7b}li{margin:6px 0}hr{border:0;border-top:1px solid #dfe7ef;margin:30px 0}@media print{body{background:white}main{box-shadow:none;margin:0;padding:0;max-width:none}h2{break-after:avoid}figure{margin:15px 0}a{color:inherit}@page{size:A4;margin:18mm}}@media(max-width:760px){main{margin:0;padding:22px}h1{font-size:25px}}
</style><main>'''+render_markdown(text)+"</main></html>"
    (OUT/"中期研究报告.html").write_text(page,encoding="utf-8");return True


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    (OUT/"figures").mkdir(exist_ok=True);(OUT/"data").mkdir(exist_ok=True)
    snapshot=OUT/"data/presentation_data.json"
    data=json.loads(snapshot.read_text(encoding="utf-8"))
    style()
    for function in [figure_labels,figure_errors,figure_signals,figure_budget,figure_calibration,figure_new,figure_stops,figure_cost,figure_method]:function(data)
    write_json(OUT/"data/figure_manifest.json",[dict(stem=s,title=t,takeaway=k,png="figures/"+s+".png",svg="figures/"+s+".svg") for s,t,k in FIGURES])
    make_slides(data);html_ready=make_html()
    print(json.dumps({"figures":9,"formats":["PNG","SVG"],"presentation_pages":11,"report_html":html_ready,"new_model_requests":0,"gpu_used":False},ensure_ascii=False))


if __name__=="__main__":main()
