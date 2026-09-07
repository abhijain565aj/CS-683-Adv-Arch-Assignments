#!/usr/bin/env python3
import argparse, csv, math, sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: python3 -m pip install matplotlib",file=sys.stderr)
    raise

NUMERIC={
    "width_bits":int,"size":int,"time_ms":float,"gflops":float,
    "speedup_vs_naive":float,"instructions_per_kernel":float,
    "instruction_ratio_vs_naive":float,"instruction_reduction_pct":float,
    "cycles_per_kernel":float,"ipc":float
}

def load(path):
    rows=[]
    with open(path,newline="") as f:
        for r in csv.DictReader(f):
            for k,c in NUMERIC.items():
                if r.get(k,"")!="":
                    try:r[k]=c(r[k])
                    except ValueError:pass
            rows.append(r)
    return rows

def line(series,xlabel,ylabel,title,path,logy=False):
    plt.figure()
    for label,x,y in series: plt.plot(x,y,marker="o",label=label)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    if logy: plt.yscale("log")
    if len(series)>1: plt.legend()
    plt.grid(True,alpha=.25); plt.tight_layout()
    plt.savefig(path,dpi=180); plt.close()

def available(rows,k):
    return any(r.get(k,"") not in ("",None) for r in rows)

def gmean(v):
    v=[float(x) for x in v if float(x)>0]
    return math.exp(sum(math.log(x) for x in v)/len(v)) if v else -1

def label(w): return "No SIMD" if w==0 else f"{w}-bit"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("csv")
    SCRIPT_DIR = Path(__file__).resolve().parent
    TASK2_ROOT = SCRIPT_DIR.parent
    ap.add_argument(
        "--outdir",
        type=Path,
        default=TASK2_ROOT / "results" / "task2b_plots"
    )
    ap.add_argument("--outdir",type=Path,default=Path("task2b_plots"))
    a=ap.parse_args()
    rows=load(a.csv); out=a.outdir; out.mkdir(parents=True,exist_ok=True)
    widths=sorted({r["width_bits"] for r in rows if r["width_bits"]>0})
    sizes=sorted({r["size"] for r in rows})

    s=[]
    for w in widths:
        rs=sorted([r for r in rows if r["width_bits"]==w],key=lambda r:r["size"])
        s.append((label(w),[r["size"] for r in rs],[r["speedup_vs_naive"] for r in rs]))
    if s: line(s,"Matrix size N (M=N=K)","Speedup over no SIMD",
               "Task 2B: Speedup vs Matrix Size",out/"task2b_speedup_vs_size.png")

    s=[]
    for n in sizes:
        rs=sorted([r for r in rows if r["size"]==n and r["width_bits"]>0],
                  key=lambda r:r["width_bits"])
        if rs:s.append((f"N={n}",[r["width_bits"] for r in rs],
                        [r["speedup_vs_naive"] for r in rs]))
    if s: line(s,"SIMD width (bits)","Speedup over no SIMD",
               "Task 2B: Speedup vs SIMD Width",out/"task2b_speedup_vs_width.png")

    order=[0]+widths
    if available(rows,"instructions_per_kernel"):
        s=[]
        for w in order:
            rs=sorted([r for r in rows if r["width_bits"]==w and
                       r.get("instructions_per_kernel","")!=""],key=lambda r:r["size"])
            if rs:s.append((label(w),[r["size"] for r in rs],
                            [r["instructions_per_kernel"] for r in rs]))
        line(s,"Matrix size N (M=N=K)","Instructions per matrix multiplication",
             "Task 2B: Instruction Count vs Matrix Size",
             out/"task2b_instructions_vs_size.png",True)

    if available(rows,"instruction_ratio_vs_naive"):
        s=[]
        for w in widths:
            rs=sorted([r for r in rows if r["width_bits"]==w and
                       r.get("instruction_ratio_vs_naive","")!=""],key=lambda r:r["size"])
            if rs:s.append((label(w),[r["size"] for r in rs],
                            [r["instruction_ratio_vs_naive"] for r in rs]))
        if s:line(s,"Matrix size N (M=N=K)","Instructions / no-SIMD instructions",
                  "Task 2B: Normalized Instruction Count",
                  out/"task2b_instruction_ratio_vs_size.png")

    s=[]
    for w in order:
        rs=sorted([r for r in rows if r["width_bits"]==w],key=lambda r:r["size"])
        if rs:s.append((label(w),[r["size"] for r in rs],[r["time_ms"] for r in rs]))
    line(s,"Matrix size N (M=N=K)","Execution time (ms)",
         "Task 2B: Execution Time vs Matrix Size",
         out/"task2b_execution_time_vs_size.png",True)

    s=[]
    for w in order:
        rs=sorted([r for r in rows if r["width_bits"]==w],key=lambda r:r["size"])
        if rs:s.append((label(w),[r["size"] for r in rs],[r["gflops"] for r in rs]))
    line(s,"Matrix size N (M=N=K)","GFLOP/s","Task 2B: Throughput vs Matrix Size",
         out/"task2b_gflops_vs_size.png")

    if available(rows,"ipc"):
        s=[]
        for w in order:
            rs=sorted([r for r in rows if r["width_bits"]==w and r.get("ipc","")!=""],
                      key=lambda r:r["size"])
            if rs:s.append((label(w),[r["size"] for r in rs],[r["ipc"] for r in rs]))
        if s:line(s,"Matrix size N (M=N=K)","Instructions per cycle",
                  "Task 2B: IPC vs Matrix Size",out/"task2b_ipc_vs_size.png")

    summary=[f"Input: {a.csv}",f"SIMD widths present: {','.join(map(str,widths))}",""]
    if widths:
        by=defaultdict(list)
        for r in rows:
            if r["width_bits"]>0:by[r["width_bits"]].append(r["speedup_vs_naive"])
        best=max(by,key=lambda w:gmean(by[w]))
        summary.append(f"Best width by geometric-mean speedup: {best}-bit ({gmean(by[best]):.4f}x)")
    summary.append("")
    for n in sizes:
        summary.append(f"N={n}")
        for r in [x for x in rows if x["size"]==n]:
            ins=r.get("instructions_per_kernel","")
            ins="n/a" if ins=="" else f"{ins:.0f}"
            summary.append(f"  {label(r['width_bits'])}: {r['time_ms']:.4f} ms, "
                           f"{r['speedup_vs_naive']:.4f}x, {r['gflops']:.3f} GFLOP/s, "
                           f"instructions={ins}")
    (out/"task2b_summary.txt").write_text("\n".join(summary)+"\n")
    print(f"Wrote plots to {out}")

if __name__=="__main__":main()
