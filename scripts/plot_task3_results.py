"""Generate Task 3 GRPO figures and metric tables from saved run records."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/task3_grpo"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE = "#2463a6", "#e07a24", "#16856b", "#8755a1"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 220, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
def j(name): return json.loads((RESULTS / name).read_text(encoding="utf-8"))
def jl(name): return [json.loads(x) for x in (RESULTS / name).read_text(encoding="utf-8").splitlines() if x.strip()]
def save(fig, name):
    fig.tight_layout(); fig.savefig(FIGURES / name, bbox_inches="tight"); plt.close(fig)
def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (RESULTS / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
def main():
    trajectory = jl("standard_trajectory.jsonl"); updates = [x["update"] for x in trajectory]
    panels = [("mean_reward","Mean reward","Reward"),("sampled_kl","Sampled reference KL","KL / token"),("policy_loss","Policy loss","Loss"),("gradient_norm","Policy gradient norm","Norm"),("sample_entropy","Sample entropy","Entropy"),("clip_fraction","Clip fraction","Fraction"),("response_length","Response length","Tokens"),("group_reward_std","Within group reward SD","Reward SD")]
    fig, axes = plt.subplots(4,2,figsize=(10,11))
    for ax,(key,title,ylabel) in zip(axes.flat,panels):
        ax.plot(updates,[x[key] for x in trajectory],"o-",color=BLUE,ms=3,lw=1); ax.set(title=title,xlabel="Update",ylabel=ylabel); ax.grid(axis="y",alpha=.2)
    summary=j("standard_summary.json")
    fig.suptitle(f"Task 3: standard GRPO, {summary['updates']} updates ({summary['wall_clock_seconds']:.0f}s; peak VRAM {summary['peak_vram_bytes']/1e9:.2f} GB)",y=1.01)
    save(fig,"grpo_standard_trajectory.png")
    group_data=j("group_size.json"); groups=group_data["conditions"]; ks=sorted(groups,key=int)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    measures=[("informative_group_rate","Informative group rate","Fraction of groups"),("mean_group_reward_std","Mean within group reward SD","Reward SD"),("relative_signal_variance","Relative signal variance","Fraction")]
    for ax,(metric,title,ylabel) in zip(axes,measures):
        for bucket,color,marker in [("overall",BLUE,"o"),("hard",ORANGE,"s"),("easy",GREEN,"^")]:
            vals=[]
            for k in ks:
                d=groups[k]["overall"] if bucket=="overall" else groups[k]["difficulty_bins"][bucket]
                vals.append(d[metric])
            ax.plot([int(k) for k in ks],vals,marker=marker,color=color,label=bucket.title())
        ax.set(title=title,xlabel="Completions per prompt (K)",ylabel=ylabel,xticks=[int(k) for k in ks]); ax.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False); fig.suptitle("Task 3: equal generation budget across GRPO group sizes",y=1.04); save(fig,"grpo_group_size_analysis.png")
    comparison=j("normalization_comparison.json"); conditions=[("grpo","GRPO",BLUE),("dr_grpo","Dr. GRPO",ORANGE)]
    fig,axes=plt.subplots(2,2,figsize=(9.5,6.5))
    measures=[("mean_reward","Held-out reward","Reward"),("mean_sampled_kl","Held-out sampled reference KL","KL / token"),("mean_response_length","Held-out response length","Mean tokens"),("truncated_fraction","Held-out truncation rate","Fraction")]
    for ax,(key,title,ylabel) in zip(axes.flat,measures):
        for name,label,color in conditions: ax.bar(label,comparison[name]["heldout"][key],color=color,width=.58)
        ax.set(title=title,ylabel=ylabel); ax.grid(axis="y",alpha=.2)
    fig.suptitle("Task 3: held-out normalization comparison (200 prompts per condition)",y=1.02); save(fig,"grpo_normalization_heldout.png")
    fig,ax=plt.subplots(figsize=(7.5,4.5)); labels=[x[1] for x in conditions]; short=[comparison[x[0]]["mean_short_policy_coefficient_per_token"] for x in conditions]; long=[comparison[x[0]]["mean_long_policy_coefficient_per_token"] for x in conditions]; xs=range(len(labels)); width=.34
    ax.bar([i-width/2 for i in xs],short,width,label="Short completions",color=BLUE); ax.bar([i+width/2 for i in xs],long,width,label="Long completions",color=ORANGE)
    ax.set(title="Length conditioned policy coefficient",ylabel="Mean absolute coefficient per token",xticks=list(xs),xticklabels=labels); ax.grid(axis="y",alpha=.2); ax.legend(frameon=False); save(fig,"grpo_length_conditioned_coefficients.png")
    fig,axes=plt.subplots(2,2,figsize=(9.5,6.5)); panels=[("mean_reward","Training reward","Reward"),("sampled_kl","Sampled reference KL","KL / token"),("sample_entropy","Sample entropy","Entropy"),("response_length","Response length","Tokens")]
    for ax,(key,title,ylabel) in zip(axes.flat,panels):
        for name,label,color in conditions:
            suffix="canonical" if name=="grpo" else "dr_grpo"; rows=jl(f"normalization_{suffix}_trajectory.jsonl")
            ax.plot([r["update"] for r in rows],[r[key] for r in rows],"o-",ms=3,lw=1,label=label,color=color)
        ax.set(title=title,xlabel="Update",ylabel=ylabel); ax.grid(axis="y",alpha=.2)
    axes[0,0].legend(frameon=False); fig.suptitle("Task 3: matched 8 update normalization trajectories",y=1.02); save(fig,"grpo_normalization_trajectories.png")
    rows=[]
    for name,label,_ in conditions:
        tr,ev=comparison[name]["train"],comparison[name]["heldout"]
        rows.append({"condition":label,"updates":tr["updates"],"group_size":tr["group_size"],"generated_tokens":tr["generated_tokens"],"wall_clock_seconds":tr["wall_clock_seconds"],"peak_vram_bytes":tr["peak_vram_bytes"],"heldout_examples":ev["examples"],"heldout_mean_reward":ev["mean_reward"],"heldout_mean_sampled_kl":ev["mean_sampled_kl"],"heldout_mean_response_length":ev["mean_response_length"],"heldout_truncated_fraction":ev["truncated_fraction"],"mean_short_policy_coefficient_per_token":comparison[name]["mean_short_policy_coefficient_per_token"],"mean_long_policy_coefficient_per_token":comparison[name]["mean_long_policy_coefficient_per_token"]})
    write_csv("normalization_metrics.csv",rows)
    rows=[]
    for k in ks:
        row={"group_size":int(k),"generation_count":groups[k]["generation_count"]}
        for bucket in ("overall","hard","easy"):
            d=groups[k]["overall"] if bucket=="overall" else groups[k]["difficulty_bins"][bucket]
            for metric in ("groups","informative_group_rate","mean_group_reward_std","relative_signal_variance"): row[f"{bucket}_{metric}"]=d[metric]
        rows.append(row)
    write_csv("group_size_metrics.csv",rows)
    print(f"Generated {len(list(FIGURES.glob('*.png')))} Task 3 figures and two metric CSV files.")
if __name__=="__main__": main()
