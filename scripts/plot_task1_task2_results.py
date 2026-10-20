"""Generate Task 1/2 report figures and CSV tables from saved run records."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
T1, T2 = ROOT / "results/task1_dpo", ROOT / "results/task2_ppo"
F1, F2 = T1 / "figures", T2 / "figures"
F1.mkdir(parents=True, exist_ok=True); F2.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE = "#2463a6", "#e07a24", "#16856b", "#8755a1"
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 220, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

def j(path):
    with path.open(encoding="utf-8") as f: return json.load(f)
def jl(path):
    with path.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def save(fig, path):
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
def csvout(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def task1():
    study, std = j(T1/"beta_study.json"), j(T1/"standard_summary.json")
    std_train = j(T1/"standard_train_summary.json")
    short = sorted(study["conditions"], key=lambda x:x["beta"]); rows=[]
    for c in short:
        t,e=c["training"],c["evaluation"]
        rows.append({"condition":e["name"],"budget":"short","beta":c["beta"],"examples":t["examples"],"optimizer_steps":t["optimizer_steps"],"train_loss":t["mean_training_dpo_loss"],"train_preference_accuracy":t["mean_training_preference_accuracy"],"heldout_dpo_loss":e["heldout_dpo_loss"],"heldout_preference_accuracy":e["heldout_preference_accuracy"],"sampled_kl_per_response_token":e["sampled_kl_per_response_token"],"mean_reward_score":e["mean_reward_score"],"mean_reference_reward_score":e["mean_reference_reward_score"],"response_length_mean_tokens":e["response_length_tokens"]["mean"],"response_length_std_tokens":e["response_length_tokens"]["std"]})
    rows.append({"condition":"standard_1_epoch","budget":"one_epoch","beta":std["beta_for_loss"],"examples":std_train["examples"],"optimizer_steps":std_train["optimizer_steps"],"train_loss":std_train["mean_training_dpo_loss"],"train_preference_accuracy":std_train["mean_training_preference_accuracy"],"heldout_dpo_loss":std["heldout_dpo_loss"],"heldout_preference_accuracy":std["heldout_preference_accuracy"],"sampled_kl_per_response_token":std["sampled_kl_per_response_token"],"mean_reward_score":std["mean_reward_score"],"mean_reference_reward_score":std["mean_reference_reward_score"],"response_length_mean_tokens":std["response_length_tokens"]["mean"],"response_length_std_tokens":std["response_length_tokens"]["std"]})
    csvout(T1/"beta_metrics.csv", rows)
    fig,axs=plt.subplots(2,3,figsize=(12.3,6.3)); xs=[r["beta"] for r in rows[:3]]
    fields=[("heldout_preference_accuracy","Held-out preference accuracy","Accuracy"),("heldout_dpo_loss","Held-out DPO loss","Loss"),("sampled_kl_per_response_token","Sampled reference KL","KL / response token"),("mean_reward_score","Reward-model score","Mean score"),("mean_reference_reward_score","Reference reward-model score","Mean score"),("response_length_mean_tokens","Generated response length","Tokens (mean ± SD)")]
    full=rows[-1]
    for ax,(key,title,ylabel) in zip(axs.flat,fields):
        ys=[r[key] for r in rows[:3]]; ax.plot(xs,ys,"o-",color=BLUE,label="Short β fork")
        ax.scatter([full["beta"]],[full[key]],marker="D",facecolors="none",edgecolors=ORANGE,s=48,label="Standard, 1 epoch")
        if key=="response_length_mean_tokens":
            ax.errorbar(xs,ys,yerr=[r["response_length_std_tokens"] for r in rows[:3]],fmt="none",ecolor=BLUE,alpha=.35,capsize=3)
            ax.errorbar([full["beta"]],[full[key]],yerr=[full["response_length_std_tokens"]],fmt="none",ecolor=ORANGE,capsize=3)
        ax.set(title=title,xlabel="β",ylabel=ylabel); ax.set_xticks([.03,.10,.30],["0.03","0.10","0.30"]); ax.grid(axis="y",alpha=.2)
    axs[0,0].legend(frameon=False); fig.suptitle("Task 1: DPO regularization (300 held-out pairs)",y=1.02); save(fig,F1/"dpo_beta_tradeoffs.png")
    fig,axs=plt.subplots(1,2,figsize=(9.4,3.8)); traces=[("standard","Standard, 1 epoch",BLUE),("beta_0.03","β=0.03, short",GREEN),("beta_0.10","β=0.10, short","#d99b22"),("beta_0.30","β=0.30, short","#bd5e55")]
    for name,label,color in traces:
        data=jl(T1/f"{name}_train.jsonl"); steps=[r["optimizer_step"] for r in data]
        axs[0].plot(steps,[r["batch_loss"] for r in data],color=color,lw=1,label=label); axs[1].plot(steps,[r["batch_preference_accuracy"] for r in data],color=color,lw=1,label=label)
    axs[0].set(title="Training DPO loss",xlabel="Optimizer step",ylabel="Batch loss"); axs[1].set(title="Training preference accuracy",xlabel="Optimizer step",ylabel="Batch accuracy")
    for ax in axs: ax.grid(axis="y",alpha=.2)
    axs[1].legend(frameon=False,ncol=2); fig.suptitle("Task 1: training traces",y=1.03); save(fig,F1/"dpo_training_traces.png")
    length=j(T1/"length_study.json"); strata=["preferred_longer","length_matched","rejected_longer"]; labels=["Preferred longer","Length matched","Rejected longer"]
    fig,axs=plt.subplots(1,3,figsize=(10.8,3.8)); xx=range(3); width=.34
    for idx,(key,name,color) in enumerate([("standard_stratified","Standard DPO",BLUE),("balanced_stratified","Length-balanced DPO",ORANGE)]):
        axs[0].bar([x+(idx-.5)*width for x in xx],[length[key]["strata"][s]["heldout_preference_accuracy"] for s in strata],width,label=name,color=color)
    axs[0].set(xticks=list(xx),xticklabels=labels,ylim=(0,1),ylabel="Preference accuracy",title="Length strata (82 pairs each)"); axs[0].tick_params(axis="x",rotation=15); axs[0].legend(frameon=False)
    wlim=[length["standard_word_limits"],length["balanced_word_limits"]]; names=["Standard","Balanced"]
    axs[1].bar(names,[r["response_length_tokens"]["mean"] for r in wlim],yerr=[r["response_length_tokens"]["std"] for r in wlim],capsize=4,color=[BLUE,ORANGE]); axs[1].set(ylabel="Generated tokens (mean ± SD)",title="Word-limit prompts (n=10)")
    axs[2].bar(names,[r["word_limit_compliance"] for r in wlim],color=[BLUE,ORANGE]); axs[2].set(ylim=(0,1),ylabel="Compliance rate",title="Explicit word-limit compliance")
    for ax in axs: ax.grid(axis="y",alpha=.2)
    fig.suptitle("Task 1: length confounding diagnostics",y=1.04); save(fig,F1/"dpo_length_bias.png")
    out=[]
    for key,name,w in [("standard_stratified","standard",wlim[0]),("balanced_stratified","balanced",wlim[1])]:
        item=length[key]; r={"condition":name,"heldout_preference_accuracy":item["heldout_preference_accuracy"],"heldout_dpo_loss":item["heldout_dpo_loss"],"stratified_pair_count":item["pair_count"],"word_limit_compliance":w["word_limit_compliance"],"word_limit_mean_response_tokens":w["response_length_tokens"]["mean"],"word_limit_std_response_tokens":w["response_length_tokens"]["std"]}
        for s in strata: r[s+"_accuracy"]=item["strata"][s]["heldout_preference_accuracy"]; r[s+"_dpo_loss"]=item["strata"][s]["heldout_dpo_loss"]
        out.append(r)
    csvout(T1/"length_metrics.csv",out)

def task2():
    data=jl(T2/"standard/updates.jsonl"); steps=[r["update"] for r in data]; summary=j(T2/"standard/summary.json")
    fig,axs=plt.subplots(4,2,figsize=(10,10)); panels=[("reward","Mean learned reward","Reward"),("kl_per_token","Reference KL","KL / token"),("policy_loss","Policy loss","Loss"),("value_loss","Value loss","Loss"),("entropy","Policy entropy","Entropy"),("clip_fraction","PPO clip fraction","Token fraction"),("response_length","Response length","Tokens")]
    for ax,(key,title,ylabel) in zip(axs.flat,panels): ax.plot(steps,[r[key] for r in data],"o-",color=BLUE,ms=3,lw=1); ax.set(title=title,xlabel="Update",ylabel=ylabel,xticks=steps[::2]); ax.grid(axis="y",alpha=.2)
    ax=axs.flat[7]; ax.plot(steps,[r["policy_gradient_norm"] for r in data],"o-",ms=3,label="Policy",color=BLUE); ax.plot(steps,[r["value_gradient_norm"] for r in data],"o-",ms=3,label="Value",color=ORANGE); ax.set(title="Gradient norms",xlabel="Update",ylabel="Norm",xticks=steps[::2]); ax.legend(frameon=False); ax.grid(axis="y",alpha=.2)
    fig.suptitle(f"Task 2: standard PPO, 20 updates ({summary['wall_clock_seconds']:.0f}s; peak VRAM {summary['peak_vram_bytes']/1e9:.2f} GB)",y=1.01); save(fig,F2/"ppo_standard_trajectory.png")
    clip=sorted(j(T2/"clipping_comparison.json"),key=lambda r:r["epsilon"]); cached=sorted(j(T2/"cached_clipping.json"),key=lambda r:r["epsilon"]); eps=[r["epsilon"] for r in clip]
    fig,axs=plt.subplots(3,2,figsize=(9.5,9)); axs[0,0].plot(eps,[r["clip_fraction"] for r in cached],"o-",label="Outside range",color=BLUE); axs[0,0].plot(eps,[r["affected_token_fraction"] for r in cached],"s--",label="Surrogate affected",color=ORANGE); axs[0,0].set(title="Fixed cached rollout geometry",xlabel="Clip ε",ylabel="Token fraction"); axs[0,0].legend(frameon=False)
    axs[0,1].plot(eps,[r["clipped_surrogate"] for r in cached],"o-",color=GREEN); axs[0,1].set(title="Cached clipped surrogate",xlabel="Clip ε",ylabel="Objective")
    axs[1,0].plot(eps,[r["heldout"]["reward_mean"] for r in clip],"o-",color=BLUE); axs[1,0].set(title="Short-fork held-out reward",xlabel="Clip ε",ylabel="Mean reward")
    axs[1,1].plot(eps,[r["heldout"]["kl_per_token_mean"] for r in clip],"o-",color=PURPLE); axs[1,1].set(title="Short-fork held-out reference KL",xlabel="Clip ε",ylabel="KL / response token")
    axs[2,0].plot(eps,[r["heldout"]["response_length_mean"] for r in clip],"o-",color=GREEN); axs[2,0].set(title="Short-fork held-out response length",xlabel="Clip ε",ylabel="Mean tokens")
    axs[2,1].plot(eps,[r["stability"]["policy_loss_std_across_updates"] for r in clip],"o-",label="Policy loss SD",color=BLUE); axs[2,1].plot(eps,[r["stability"]["gradient_norm_std_across_updates"] for r in clip],"s--",label="Gradient norm SD",color=ORANGE); axs[2,1].set(title="Across-update stability",xlabel="Clip ε",ylabel="Standard deviation"); axs[2,1].legend(frameon=False)
    for ax in axs.flat: ax.grid(axis="y",alpha=.2)
    fig.suptitle("Task 2: clipping study (cached batch and 8-update forks)",y=1.02); save(fig,F2/"ppo_clipping_study.png")
    kl=sorted(j(T2/"kl_comparison.json"),key=lambda r:r["beta_kl"]); betas=[r["beta_kl"] for r in kl]
    fig,axs=plt.subplots(2,2,figsize=(9.5,6.5)); metrics=[("reward_mean","Held-out reward","Mean reward"),("kl_per_token_mean","Held-out reference KL","KL / token"),("entropy_mean","Held-out entropy","Entropy"),("response_length_mean","Held-out response length","Tokens (mean)")]
    for ax,(key,title,ylabel) in zip(axs.flat,metrics): ax.plot(betas,[r["heldout"][key] for r in kl],"o-",color=PURPLE); ax.set(title=title,xlabel="KL penalty βₖₗ",ylabel=ylabel); ax.grid(axis="y",alpha=.2)
    fig.suptitle("Task 2: KL-pressure held-out comparison",y=1.02); save(fig,F2/"ppo_kl_pressure_heldout.png")
    fig,axs=plt.subplots(2,2,figsize=(9.5,6.5)); traj=[("reward","Training reward","Reward"),("kl_per_token","Training KL","KL / token"),("entropy","Training entropy","Entropy"),("response_length","Training response length","Tokens")]
    for ax,(key,title,ylabel) in zip(axs.flat,traj):
        for row,color in zip(kl,[BLUE,GREEN,"#bd5e55"]): ax.plot([x["update"] for x in row["trajectory"]],[x[key] for x in row["trajectory"]],"o-",ms=3,lw=1,label=f"βₖₗ={row['beta_kl']:.2f}",color=color)
        ax.set(title=title,xlabel="Update",ylabel=ylabel); ax.grid(axis="y",alpha=.2)
    axs[0,0].legend(frameon=False); fig.suptitle("Task 2: KL-pressure 8-update trajectories",y=1.02); save(fig,F2/"ppo_kl_pressure_trajectories.png")
    csvout(T2/"clipping_metrics.csv",[{"epsilon":x["epsilon"],"cached_clipped_surrogate":c["clipped_surrogate"],"cached_clip_fraction":c["clip_fraction"],"cached_affected_token_fraction":c["affected_token_fraction"],"cached_valid_tokens":c["valid_tokens"],"updates":x["training"]["updates"],"heldout_n":x["heldout"]["n"],"heldout_reward_mean":x["heldout"]["reward_mean"],"heldout_kl_per_token_mean":x["heldout"]["kl_per_token_mean"],"heldout_entropy_mean":x["heldout"]["entropy_mean"],"heldout_response_length_mean":x["heldout"]["response_length_mean"],"heldout_eos_rate":x["heldout"]["eos_rate"],**x["stability"],"wall_clock_seconds":x["training"]["wall_clock_seconds"],"peak_vram_bytes":x["training"]["peak_vram_bytes"]} for x,c in zip(clip,cached)])
    csvout(T2/"kl_metrics.csv",[{"beta_kl":x["beta_kl"],"updates":x["training"]["updates"],"heldout_n":x["heldout"]["n"],"heldout_reward_mean":x["heldout"]["reward_mean"],"heldout_kl_per_token_mean":x["heldout"]["kl_per_token_mean"],"heldout_entropy_mean":x["heldout"]["entropy_mean"],"heldout_response_length_mean":x["heldout"]["response_length_mean"],"heldout_eos_rate":x["heldout"]["eos_rate"],"wall_clock_seconds":x["training"]["wall_clock_seconds"],"peak_vram_bytes":x["training"]["peak_vram_bytes"]} for x in kl])

if __name__=="__main__":
    task1(); task2(); print("Generated Task 1 and Task 2 figures and metric tables.")
