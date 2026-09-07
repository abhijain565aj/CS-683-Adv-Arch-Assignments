#!/usr/bin/env python3

from __future__ import annotations
import argparse, csv, json, math, re, shlex, socket, subprocess, sys
from pathlib import Path

COMMON_FLAGS = ["-std=c++17", "-O2", "-fno-tree-vectorize", "-Wall"]

WIDTH_CONFIG = {
    128: dict(function="matmul_simd_128", helper="task2b_simd_128.cpp",
              flags=["-msse4.2"], required="sse4_2", label="SIMD 128"),
    256: dict(function="matmul_simd", helper=None,
              flags=["-mavx2", "-mfma"], required="avx2", label="SIMD 256"),
    512: dict(function="matmul_simd_512", helper="task2b_simd_512.cpp",
              flags=["-mavx512f", "-mfma"], required="avx512f", label="SIMD 512"),
}

EVENT_CANDIDATES = {
    "instructions": ["instructions", "cpu_core/instructions/"],
    "cycles": ["cycles", "cpu_core/cycles/"],
    "branches": ["branches", "cpu_core/branches/"],
    "branch_misses": ["branch-misses", "cpu_core/branch-misses/"],
    "cache_references": ["cache-references"],
    "cache_misses": ["cache-misses"],
    "l1d_misses": ["L1-dcache-load-misses", "l1-dcache-load-misses",
                    "cpu_core/L1D.REPLACEMENT/", "cpu_core/l1d.replacement/"],
    "llc_misses": ["LLC-load-misses", "llc-load-misses",
                    "cpu_core/MEM_LOAD_RETIRED.L3_MISS/",
                    "cpu_core/mem_load_retired.l3_miss/"],
}

CSV_FIELDS = [
    "host","cpu","stage","width_bits","size","correct","time_ms","gflops",
    "speedup_vs_naive","perf_reps","instructions_per_kernel",
    "instruction_ratio_vs_naive","instruction_reduction_pct",
    "cycles_per_kernel","ipc","branches_per_kernel","branch_misses_per_kernel",
    "branch_miss_rate_pct","cache_references_per_kernel","cache_misses_per_kernel",
    "cache_miss_rate_pct","l1d_misses_per_kernel","llc_misses_per_kernel",
    *[f"perf_event_{k}" for k in EVENT_CANDIDATES],
    "checksum",
]

def parse_int_list(s):
    vals=[int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals: raise argparse.ArgumentTypeError("empty list")
    return vals

def run(cmd):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)

def find_source(root, candidates):
    for rel in candidates:
        p=root/rel
        if p.is_file(): return p
    raise FileNotFoundError("Could not find: " + ", ".join(candidates))

def find_include(root, src):
    for d in [root/"include", src.parent.parent/"include", src.parent, root]:
        if (d/"matmul.h").is_file(): return d
    raise FileNotFoundError("Could not find matmul.h")

def cpu_flags():
    try: text=Path("/proc/cpuinfo").read_text()
    except OSError: return set()
    for line in text.splitlines():
        if line.lower().startswith("flags"):
            return set(line.split(":",1)[1].split())
    return set()

def pin(cpu, cmd):
    return cmd if cpu is None else ["taskset","-c",str(cpu),*cmd]

def compile_bin(cxx, driver, kernel, naive, out, fn, inc, flags):
    cmd=[cxx,*COMMON_FLAGS,*flags,f"-I{inc}",f"-DBENCH_FN={fn}",
         str(driver),str(kernel)]
    if kernel.resolve()!=naive.resolve(): cmd.append(str(naive))
    cmd += ["-o",str(out)]
    p=run(cmd)
    if p.returncode:
        print(p.stderr,file=sys.stderr)
        raise RuntimeError("compile failed: "+" ".join(map(shlex.quote,cmd)))

def parse_json(stdout):
    lines=[x for x in stdout.splitlines() if x.strip()]
    if not lines: raise RuntimeError("no JSON output")
    return json.loads(lines[-1])

def run_json(binary,args,cpu, allow_fail=False):
    p=run(pin(cpu,[str(binary),*args]))
    if p.returncode and not allow_fail:
        print(p.stderr,file=sys.stderr)
        raise RuntimeError("benchmark failed")
    return parse_json(p.stdout), p.returncode

def perf_supported(event,cpu):
    p=run(["perf","stat","-x",";","-e",event,"--",*pin(cpu,["true"])])
    t=(p.stdout+"\n"+p.stderr).lower()
    bad=["not supported","event syntax error","no permission",
         "permission denied","access to performance monitoring"]
    return p.returncode==0 and not any(x in t for x in bad)

def resolve_events(cpu, overrides, no_perf):
    if no_perf: return {k:None for k in EVENT_CANDIDATES}
    try:
        if run(["perf","--version"]).returncode: raise FileNotFoundError
    except FileNotFoundError:
        print("warning: perf unavailable; timing only",file=sys.stderr)
        return {k:None for k in EVENT_CANDIDATES}
    ans={}
    for logical,cands in EVENT_CANDIDATES.items():
        choices=[overrides[logical]] if logical in overrides else cands
        ans[logical]=next((e for e in choices if perf_supported(e,cpu)),None)
    return ans

def norm(s): return re.sub(r"[^a-z0-9]+","",s.lower())

def number(s):
    s=s.strip().replace(",","")
    if not s or s.startswith("<"): return None
    try: return float(s)
    except ValueError: return None

def parse_perf(stderr,events):
    ans={k:None for k in events}
    expected={k:norm(v) if v else "" for k,v in events.items()}
    for line in stderr.splitlines():
        parts=line.split(";")
        if len(parts)<3: continue
        val,seen=number(parts[0]),norm(parts[2])
        if val is None or not seen: continue
        for k,want in expected.items():
            if want and (seen==want or seen in want or want in seen):
                ans[k]=val
                break
    return ans

def perf_run(binary,args,cpu,events):
    selected=[e for e in events.values() if e]
    prog=pin(cpu,[str(binary),*args])
    if not selected:
        p=run(prog)
        return parse_json(p.stdout), {k:None for k in events}
    cmd=["perf","stat","-x",";"]
    for e in selected: cmd += ["-e",e]
    cmd += ["--",*prog]
    p=run(cmd)
    if p.returncode:
        print(p.stderr,file=sys.stderr)
        raise RuntimeError("perf run failed")
    return parse_json(p.stdout), parse_perf(p.stderr,events)

def corrected(measured,setup,reps):
    out={}
    for k,a in measured.items():
        b=setup.get(k)
        if a is None: out[k]=None
        elif b is None: out[k]=a/reps
        else: out[k]=max(0.0,a-b)/reps
    return out

def div(a,b):
    return None if a is None or b is None or b==0 else a/b

def percent(a,b):
    x=div(a,b)
    return None if x is None else 100*x

def adaptive_reps(ms,target,minr,maxr):
    if ms<=0: return minr
    return max(minr,min(maxr,int(math.ceil(target/ms))))

def main():
    ap=argparse.ArgumentParser()
    here=Path(__file__).resolve().parent
    SCRIPT_DIR = Path(__file__).resolve().parent
    TASK2_ROOT = SCRIPT_DIR.parent
    ap.add_argument(
        "--project-root",
        type=Path,
        default=TASK2_ROOT
    )
    ap.add_argument("--driver",type=Path,default=here/"task2b_driver.cpp")
    ap.add_argument("--simd128",type=Path,default=here/"task2b_simd_128.cpp")
    ap.add_argument("--simd512",type=Path,default=here/"task2b_simd_512.cpp")
    ap.add_argument("--sizes",type=parse_int_list,
                    default=[128,256,512,1024,1536,2048])
    ap.add_argument("--widths",type=parse_int_list,default=[128,256,512])
    ap.add_argument("--reps",type=int,default=5)
    ap.add_argument("--warmups",type=int,default=1)
    ap.add_argument("--seed",type=int,default=1234)
    ap.add_argument("--cpu",type=int,default=None)
    ap.add_argument("--perf-target-ms",type=float,default=750.0)
    ap.add_argument("--perf-min-reps",type=int,default=1)
    ap.add_argument("--perf-max-reps",type=int,default=1000)
    ap.add_argument(
        "--output",
        type=Path,
        default=TASK2_ROOT / "results" / "task2b_results.csv"
    )   
    ap.add_argument("--capabilities-output",type=Path,
                        default=Path("task2b_capabilities.txt"))
    ap.add_argument("--build-dir",type=Path,default=Path(".task2b_build"))
    ap.add_argument("--cxx",default="g++")
    ap.add_argument("--no-perf",action="store_true")
    ap.add_argument("--event",action="append",default=[])
    a=ap.parse_args()

    if any(w not in WIDTH_CONFIG for w in a.widths):
        ap.error("--widths supports only 128,256,512")

    root=a.project_root.resolve()
    naive=find_source(root,["src/matmul_naive.cpp","matmul_naive.cpp"])
    simd256=find_source(root,["src/matmul_simd.cpp","matmul_simd.cpp"])
    inc=find_include(root,simd256)
    build=(root/a.build_dir).resolve() if not a.build_dir.is_absolute() else a.build_dir
    build.mkdir(parents=True,exist_ok=True)
    output=(root/a.output).resolve() if not a.output.is_absolute() else a.output
    cap=(root/a.capabilities_output).resolve() if not a.capabilities_output.is_absolute() else a.capabilities_output

    flags=cpu_flags()
    widths=[]
    skipped=[]
    for w in dict.fromkeys(a.widths):
        req=WIDTH_CONFIG[w]["required"]
        if not flags or req in flags: widths.append(w)
        else: skipped.append((w,req))

    overrides={}
    for item in a.event:
        if "=" not in item: ap.error("--event must be LOGICAL=PERF_EVENT")
        k,v=item.split("=",1)
        if k not in EVENT_CANDIDATES: ap.error("bad logical perf event "+k)
        overrides[k]=v

    events=resolve_events(a.cpu,overrides,a.no_perf)

    print("CPU SIMD support:")
    for w in a.widths:
        print(f"  {w}-bit: {'YES' if w in widths else 'NO - skipped'}")
    print("Perf events:")
    for k,v in events.items(): print(f"  {k:18s}: {v or 'UNAVAILABLE'}")

    cap.write_text("\n".join([
        f"host={socket.gethostname()}",
        f"cpu={'' if a.cpu is None else a.cpu}",
        "supported_widths="+",".join(map(str,widths)),
        "skipped="+",".join(f"{w}:{req}" for w,req in skipped),
        "cpu_flags="+" ".join(sorted(flags)),
        *[f"{k}={v or 'UNAVAILABLE'}" for k,v in events.items()],
        ""
    ]))

    naive_bin=build/"task2b_naive"
    compile_bin(a.cxx,a.driver.resolve(),naive,naive,naive_bin,
                "matmul_naive",inc,["-mavx2","-mfma"])

    bins={}
    for w in widths:
        cfg=WIDTH_CONFIG[w]
        kernel = (a.simd128.resolve() if w==128 else
                  simd256 if w==256 else a.simd512.resolve())
        b=build/f"task2b_simd_{w}"
        compile_bin(a.cxx,a.driver.resolve(),kernel,naive,b,cfg["function"],
                    inc,cfg["flags"])
        bins[w]=b

    print("\nCorrectness checks (N=129):")
    for w,b in bins.items():
        r,code=run_json(b,["check","129",str(a.seed)],a.cpu,allow_fail=True)
        print(f"  {w}-bit: {'correct' if r.get('correct') else 'INCORRECT'}")
        if code or not r.get("correct"): raise RuntimeError(f"{w}-bit failed correctness")

    def collect(stage,width,binary,size):
        timing,_=run_json(binary,["time",str(size),str(a.reps),
                                  str(a.warmups),str(a.seed)],a.cpu)
        ms=float(timing["time_ms"])
        pr=adaptive_reps(ms,a.perf_target_ms,a.perf_min_reps,a.perf_max_reps)
        if any(events.values()):
            _,m=perf_run(binary,["perf",str(size),str(pr),str(a.seed)],a.cpu,events)
            _,s=perf_run(binary,["setup",str(size),str(a.seed)],a.cpu,events)
            c=corrected(m,s,pr)
        else: c={k:None for k in events}
        ins,cyc=c.get("instructions"),c.get("cycles")
        br,bm=c.get("branches"),c.get("branch_misses")
        cr,cm=c.get("cache_references"),c.get("cache_misses")
        row=dict(host=socket.gethostname(),cpu="" if a.cpu is None else a.cpu,
                 stage=stage,width_bits=width,size=size,correct="yes",
                 time_ms=ms,gflops=float(timing["gflops"]),speedup_vs_naive="",
                 perf_reps=pr,
                 instructions_per_kernel="" if ins is None else ins,
                 instruction_ratio_vs_naive="",instruction_reduction_pct="",
                 cycles_per_kernel="" if cyc is None else cyc,
                 ipc="" if div(ins,cyc) is None else div(ins,cyc),
                 branches_per_kernel="" if br is None else br,
                 branch_misses_per_kernel="" if bm is None else bm,
                 branch_miss_rate_pct="" if percent(bm,br) is None else percent(bm,br),
                 cache_references_per_kernel="" if cr is None else cr,
                 cache_misses_per_kernel="" if cm is None else cm,
                 cache_miss_rate_pct="" if percent(cm,cr) is None else percent(cm,cr),
                 l1d_misses_per_kernel="" if c.get("l1d_misses") is None else c["l1d_misses"],
                 llc_misses_per_kernel="" if c.get("llc_misses") is None else c["llc_misses"],
                 checksum=float(timing["checksum"]))
        for k,v in events.items(): row[f"perf_event_{k}"]=v or ""
        return row

    rows=[]
    print("\nCollecting:")
    for size in a.sizes:
        n=collect("no SIMD",0,naive_bin,size)
        n["speedup_vs_naive"]=1.0
        n["instruction_ratio_vs_naive"]=1.0
        n["instruction_reduction_pct"]=0.0
        rows.append(n)
        print(f"N={size}: no SIMD {n['time_ms']:.3f} ms")

        ni=float(n["instructions_per_kernel"]) if n["instructions_per_kernel"]!="" else None
        for w in widths:
            r=collect(WIDTH_CONFIG[w]["label"],w,bins[w],size)
            r["speedup_vs_naive"]=float(n["time_ms"])/float(r["time_ms"])
            if ni is not None and r["instructions_per_kernel"]!="":
                ratio=float(r["instructions_per_kernel"])/ni
                r["instruction_ratio_vs_naive"]=ratio
                r["instruction_reduction_pct"]=100*(1-ratio)
            rows.append(r)
            print(f"         {w}-bit {r['time_ms']:.3f} ms  {r['speedup_vs_naive']:.3f}x")

    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=CSV_FIELDS)
        wr.writeheader()
        for r in rows: wr.writerow({k:r.get(k,"") for k in CSV_FIELDS})

    print(f"\nWrote {output}")
    print(f"Wrote {cap}")
    print("No assignment source file was modified.")

if __name__=="__main__":
    main()
