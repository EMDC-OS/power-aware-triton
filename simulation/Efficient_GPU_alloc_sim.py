#!/usr/bin/python3
from ortools.linear_solver import pywraplp
import multiprocessing
import multiprocessing.pool
import random
import time
import math
# from concurrent import futures
import parmap
import os
import datetime

max_dynamic = 150
static_power = 235
j_to_kwh = 3600000

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

#Load config file
def read_config():
    f = open("./config.txt","r")
    
    model = f.readline().split()
    gpu = f.readline().split()
    priority = list(map(int,f.readline().split()))

    # print("%d models : "%len(model),model)
    # print("%d GPUs : "%len(gpu),gpu)

    p_ij = []
    r_ij = []
    
    server = list(map(int,f.readline().split()))

    if len(server) != len(gpu):
        print("Server config mismatch!")
        exit()

    for i in range(len(gpu)):
        pwr = list(map(float,f.readline().split()))
        p_ij += [pwr]
    for i in range(len(gpu)):
        rate = list(map(float,f.readline().split()))
        r_ij += [rate]
    
    # for i in p_ij:
    #     print(i)
    # for j in r_ij:
    #     print(j)

    n = int(f.readline())
    service = f.readline().split()
    print("Service list: ",",".join(service))

    t_name = f.readline().split()[0]
    print("Trace file name :",t_name)

    f.close()

    return model, gpu, server, p_ij, r_ij, service, t_name, priority

# Cluster assingment
def CA_opt(server, m, g_type, p, r, q):
    # print("a4 ",server)

    # Create the solver instance
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Create the variable    
    x={}
    for i in range(m):
        for j in range(g_type):
            x[(i,j)] = solver.IntVar(0,server[j],"x_%d_%d"%(i,j))
    
    # Def. constraint
    for i in range(m):
        solver.Add(sum(x[i,j]*r[i][j] for j in range(g_type))>=q[i])
    
    for j in range(g_type):
        solver.Add(sum(x[i,j] for i in range(m))<=server[j])

    # Objective function
    objective = solver.Objective()
    for i in range(m):
        for j in range(g_type):
            objective.SetCoefficient(x[i,j], p[i][j]+ max_dynamic)
    objective.SetMinimization()

    # Run solver
    status = solver.Solve()
    
    # print(objective.Value(), "Watts\n")
  
    # cluster allocation result
    print("== Optimal cluster assignment result ==")
    res = []
    for i in range(m):
        print("model %02d"%i,end=" ")
        res.append([int(x[i,j].solution_value()) for j in range(g_type)])
        rate = sum([r[i][j]*res[-1][j] for j in range(g_type)])
        print("[",", ".join(f"{j:>4}"for j in res[-1]),"]\t",rate," / ",q[i])        

    server_ = [sum([res[i][j] for i in range(m)]) for j in range(g_type)]
    print("\nmod_all  ","[",", ".join(f"{j:>4}"for j in server_),
                                    "]\t%d / %d"%(sum(server_),sum(server)))
    print("="*39+"\n")

    return objective.Value(), res

# Heuristic cluster assignment
def CA_heuristic(server, m, g_type, r, peak, order, priority):
    # Container of assingment result
    res=[[0]*g_type for j in range(m)]

    # Allocate servers in descending order of peak rate
    free_map = server.copy()
    for i in order:
        get_server(res[i], r[i], priority, peak[i], free_map)

    # cluster allocation result
    print("== Heuristic cluster assignment result ==")
    for i in range(m):
        print("model %02d"%i,end=" ")
        rate = sum([r[i][j]*res[i][j] for j in range(g_type)])
        print("[",", ".join(f"{j:>4}"for j in res[i]),"]\t",rate," / ",peak[i]) 
    
    server_ = [sum([res[i][j] for i in range(m)]) for j in range(g_type)]
    print("\nmod_all  ","[",", ".join(f"{j:>4}"for j in server_),
                                    "]\t%d / %d"%(sum(server_),sum(server)))
    print("="*39+"\n")

    return res

# Subroutine of CA_heruistic - assginment free server
def get_server(alloc, max_rate, priority, peak, free_server):
    # Convert priority to index
    rank = [priority.index(i) for i in sorted(priority)]
    
    # Find free server
    idx = 0
    g = [0]*len(rank)
    if peak == 0: return g
    while(idx < len(rank)):
        if free_server[rank[idx]] == 0:
            idx+=1
            continue

        alloc[rank[idx]]+=1
        free_server[rank[idx]]-=1
        g[rank[idx]]+=1

        rate = sum([alloc[i]*max_rate[i] for i in range(len(rank))])

        if rate >= peak: break
    if sum([alloc[i]*max_rate[i] for i in range(len(rank))]) < peak:
        print(alloc)
        print(sum([alloc[i]*max_rate[i] for i in range(len(rank))]), peak)
        print("Not feasible!")
        exit()
    
def read_model(name):
    print(name)
    f = open("./model_config/scale/%s.conf"%name,'r')
    
    num = int(f.readline())
    
    # rate
    rate = []
    for _ in range(num):
        rate.append(list(map(int,f.readline().split())))

    pwr = []
    for _ in range(num):
        pwr.append(list(map(float,f.readline().split())))

    return [pwr, rate]

def req_dist_opt(pwr, rate, gpu, q, raw, idx):
    # Create the solver instance
    solver = pywraplp.Solver.CreateSolver("SCIP")

    n = int(sum(gpu))
   
    g_list = []
    for i in range(len(gpu)):
        g_list += [i]*int(gpu[i])
    p=[]
    r=[]
    for i in range(n):
        p+=[pwr[g_list[i]]]
        r+=[rate[g_list[i]]]

    # Create the variable
    x={}
    for i in range(n):
        for j in range(5):
            x[(i,j)] = solver.IntVar(0,1,"g%d_s%d"%(i,j))

    # Constarint
    for i in range(n):
        solver.Add(sum(x[i,j] for j in range(5)) == 1)

    solver.Add(sum( sum(x[i,j]*r[i][j] for j in range(5)) for i in range(n)) >= q) 

    # Objective function
    objective = solver.Objective()

    for i in range(n):
        for j in range(5):
            objective.SetCoefficient(x[i,j], p[i][j])
    
    status = solver.Solve()
    
    # if status != pywraplp.Solver.OPTIMAL:
    
    for i in range(n):
        tmp = [x[i,j].solution_value() for j in range(5)]

    dist_r = [0] * len(gpu)
    for i in range(n):
        g = g_list[i]
        dist_r[g] += sum([x[i,j].solution_value()*r[i][j] for j in range(5)])
    # if sum(dist_r)==0 and idx ==4 : 
    #     print(idx,raw, gpu, dist_r, q)
    #     print(pwr, rate, gpu, q, raw, idx)
    #     print('\n')
    div_ = sum(dist_r)
    dist_r = [i/div_*q for i in dist_r]

    return dist_r

def line(x1,y1,x2,y2):
    return [(y2-y1)/(x2-x1), (x2*y1-x1*y2)/(x2-x1)]

def read_power_data_s(model,gpu):
    # Generate dummy data
    m = len(model)
    idle_pwr = {'titan':89, '3090':155, 'v100':100}
    func_ = [[] for i in range(m)]
    for i in range(m):
        for j in range(len(gpu)):
            name="./model_config/scale/%s_%s"%(model[i],gpu[j])
            f=open(name,'r')
            pwr={}
            for line in f.read().split('\n'):
                if line == "": break
                r,p=list(map(float,line.split()))
                pwr[int(r)]=p-idle_pwr[gpu[j]]
            func_[i].append(pwr)
    
    for i in range(m):
        if i == 'bert': continue
        for j in range(len(gpu)):
            unit = list(func_[i][j])
            # print(unit)
            for idx in range(1,len(unit)):
                if (unit[idx]-unit[idx-1])!=5:
                    func_[i][j][unit[idx]-5] = (func_[i][j][unit[idx]]+func_[i][j][unit[idx-1]])/2
    for i in range(m):
        for j in range(len(gpu)):
            max_rate = max(list(func_[i][j]))
            for idx in func_[i][j]:
                func_[i][j][idx]+=10+ max_dynamic*idx/max_rate
    return func_

def read_power_data_no_dvfs(model,gpu):
    # Generate dummy data
    m = len(model)
    idle_pwr = {'titan':89, '3090':155, 'v100':100}
    func_ = [[] for i in range(m)]
    for i in range(m):
        for j in range(len(gpu)):
            name="./model_config/dvfs/%s_%s"%(model[i],gpu[j])
            f=open(name,'r')
            pwr={}
            for line in f.read().split('\n'):
                if line == "": break
                r,p=list(map(float,line.split()))
                pwr[int(r)]=p-idle_pwr[gpu[j]]
            func_[i].append(pwr)
    
    for i in range(m):
        if i == 'bert': continue
        for j in range(len(gpu)):
            unit = list(func_[i][j])
            # print(unit)
            for idx in range(1,len(unit)):
                if (unit[idx]-unit[idx-1])!=5:
                    func_[i][j][unit[idx]-5] = (func_[i][j][unit[idx]]+func_[i][j][unit[idx-1]])/2
    for i in range(m):
        for j in range(len(gpu)):
            max_rate = max(list(func_[i][j]))
            for idx in func_[i][j]:
                func_[i][j][idx]+=10+ max_dynamic*idx/max_rate
    return func_

def read_power_data_default(model,gpu):
    # Generate dummy data
    m = len(model)
    idle_pwr = {'titan':89, '3090':155, 'v100':100}
    func_ = [[] for i in range(m)]
    for i in range(m):
        for j in range(len(gpu)):
            name="./model_config/default/%s_%s"%(model[i],gpu[j])
            f=open(name,'r')
            pwr={}
            for line in f.read().split('\n'):
                if line == "": break
                r,p=list(map(float,line.split()))
                pwr[int(r)]=p-idle_pwr[gpu[j]]
            func_[i].append(pwr)
    
    for i in range(m):
        if i == 'bert': continue
        for j in range(len(gpu)):
            unit = list(func_[i][j])
            # print(unit)
            for idx in range(1,len(unit)):
                if (unit[idx]-unit[idx-1])!=5:
                    func_[i][j][unit[idx]-5] = (func_[i][j][unit[idx]]+func_[i][j][unit[idx-1]])/2
    for i in range(m):
        for j in range(len(gpu)):
            max_rate = max(list(func_[i][j]))
            for idx in func_[i][j]:
                func_[i][j][idx]+=10+ max_dynamic*idx/max_rate
    return func_

def power_module(rate, server, func_):

    n = len(rate) #server type

    P = [0] * n
    for i in range(n):
        if server[i] == 0: continue
        
        r = rate[i]/server[i]
        x1=math.floor(r)//5*5

        try:
            if r <= 5:
                pwr = func_[i][5]
            else:
                pwr = ( func_[i][x1+5] - func_[i][x1] ) * (r-x1) / 5 + func_[i][x1]
        except KeyError:
            pwr = func_[i][x1]
        P[i] += pwr*server[i]
    return P

def free_server(alloc_, max_r_, pr_, q_):
    # select lowest pri gpu server
    n = len(alloc_)
    
    g=-1
    f_server=[0]*n
    while(1):
        t = [pr_[i]*[9999,1][alloc_[i]>0] for i in range(n)]
        g = t.index(min(t))
        
        # Test release
        alloc_[g]-=1

        rate = sum([alloc_[i]*max_r_[i] for i in range(n)])
        if rate < q_:
            alloc_[g]+=1
            break
        else: f_server[g]+=1
    return f_server

def node_selection_opt(q, cluster, pwr, rate, node_conf, func_, idx):
    # Create the solver instance
    solver = pywraplp.Solver.CreateSolver("SCIP")

    n = len(cluster)

    # Create the variable
    x = {}
    for j in range(n):
        x[j] = solver.IntVar(0,cluster[j],"x_%d"%j)

    # Constraint
    solver.Add(sum(x[j] * rate[j] for j in range(n)) >= q)

    # Objective function
    objective = solver.Objective()

    for j in range(n):
        objective.SetCoefficient(x[j], pwr[j]+ max_dynamic)
    objective.SetMinimization()

    status = solver.Solve() #FIXME Handle solver error
    
    # if sum([x[i].solution_value() for i in range(n)])==0:
    #     print(q, cluster, rate, pwr, [x[i].solution_value() for i in range(n)])

    selected =  [x[i].solution_value() for i in range(n)]

    is_homogen = False
    for i in selected:
        if int(sum(selected)) == i: 
            is_homogen = True   
    #FIXME Is it right? Alternative method: count zero
    # if idx==4 and is_homogen==False: 
    #     print(idx,[x[i].solution_value() for i in range(n)],["Hetero","Homo"][is_homogen] ,q )
    if is_homogen:
        dist=[abs((selected[i]>0)*q) for i in range(len(cluster))]
    else: # heterogeneous
        dist=req_dist_opt(*node_conf, selected, q, cluster, idx)
    
    p_res = power_module(dist, selected, func_)
    return selected, p_res, dist

def option1(param):
    t, thread, idx ,cluster, p, r, trace, node_conf, func_ = param

    if idx == 0: print("Start option1 %d with %dthreads"%(idx, thread))

    pbar=[False,True][idx==0]
    result = parmap.map(node_selection_opt, trace[:t], cluster, p, r, node_conf,
                                func_, idx, pm_pbar=pbar,pm_processes=thread)

    if idx == 0: print("Done option1 %d"%idx)

    sim_server = []
    sim_rate = []
    sim_pwr = []
    for i in range(t):
        sim_server.append(result[i][0])
        sim_pwr.append(result[i][1])
        sim_rate.append(result[i][2])

    if idx == 0: print("Done power_module %d\n"%idx)
    return [sim_server, sim_pwr, sim_rate]

def option2(param):
    t, thread, idx ,cluster, p, r, trace, node_conf, func_ = param

    if idx == 0: print("Start option2 %d with %dthreads"%(idx, thread))

    pbar=[False,True][idx==0]
    result = parmap.map(node_selection_opt, trace[:t], cluster, p, r, node_conf,
                                func_, idx, pm_pbar=pbar,pm_processes=thread)

    if idx == 0: print("Done option2 %d"%idx)

    sim_server = []
    sim_pwr = []
    sim_rate = []
    for i in range(t):
        sim_server.append(result[i][0])
        sim_pwr.append(result[i][1])
        sim_rate.append(result[i][2])

    if idx == 0: print("Done power_module %d\n"%idx)
    return [sim_server, sim_pwr, sim_rate]

def node_selection_h(q, alloc_gpu, max_r, func_, pri):
    free_map = alloc_gpu.copy()
    alloc = [0]*len(gpu)
    get_server(alloc, max_r, pri, q, free_map)

    dist = [alloc[j] * max_r[j] for j in range(len(gpu))]
    tot_r = sum(dist)
    if tot_r == 0: tot_r = 1
    dist_ = [q*dist[j] / tot_r for j in range(len(gpu))]
    # print(alloc_gpu, alloc, dist, dist_, q)
    p_res = power_module(dist_, alloc, func_)
    return alloc, p_res, dist_

def option3(param):
    t,num_process,idx ,alloc_gpu, r, trace, func_=param

    res_heu = []
    heu_pwr = []
    pri = [2,3,1]
    
    pbar=[False,True][idx==0]
    result = parmap.map(node_selection_h,trace[:t], alloc_gpu,r,func_,pri, 
                                    pm_pbar=pbar,pm_processes=num_process)

    sim_server = []
    sim_pwr = []
    sim_rate = []
    for i in range(t):
        sim_server.append(result[i][0])
        sim_pwr.append(result[i][1])
        sim_rate.append(result[i][2])
    return [sim_server, sim_pwr, sim_rate]

def static_active_server(s, m):
    idle = [89,155,100]
    g_type = len(idle)
    non_gpu_ = static_power*(sum([sum([sum(j)for j in i]) for i in s]))
    # for i in s: print(i)
    gpu_ = 0
    for t in s:
        for i in range(m):
            gpu_+= sum([idle[k]*t[i][k]for k in range(g_type)])
    # print(non_gpu_)
    # print(gpu_)
    return (non_gpu_ + gpu_)/j_to_kwh

# def idle_GPU_pwr(server, cluster):
#     idle = [x-y for x,y in zip(cluster, server)]
#     power = [89,155,100]
#     return sum([idle[i]*power[i] for i in range(len(power))])

def save_log(name, server, power, rate):
    f=open(name+".log",'w')

    l = len(server)
    # print(server[0])
    # print(power[0])
    # print(rate[0])
    for i in range(l):
        for j in server[i]:
            for k in j:
                if k == 0: f.write("0 ")
                else: f.write("%d "%int(k))
        f.write("| ")
        for j in power[i]:
            for k in j:
                if k == 0: f.write("0 ")
                else: f.write("%lf "%k)
        f.write("| ")
        for j in rate[i]:
            for k in j:
                if k == 0: f.write("0 ")
                else: f.write("%lf "%k)
        f.write("\n")

def CA_ideal(server, m, g_type, p, r, q):
    # print("a4 ",server)

    # Create the solver instance
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Create the variable    
    x={}
    for i in range(m):
        for j in range(g_type):
            x[(i,j)] = solver.IntVar(0,server[j],"x_%d_%d"%(i,j))
    
    # Def. constraint
    for i in range(m):
        solver.Add(sum(x[i,j]*r[i][j] for j in range(g_type))>=q[i])
    
    for j in range(g_type):
        solver.Add(sum(x[i,j] for i in range(m))<=server[j])

    # Objective function
    objective = solver.Objective()
    for i in range(m):
        for j in range(g_type):
            objective.SetCoefficient(x[i,j], p[i][j]+ max_dynamic)
    objective.SetMinimization()

    # Run solver
    status = solver.Solve()
    
    # print(objective.Value(), "Watts\n")
  
    # cluster allocation result
    res = []
    for i in range(m):
        res.append([int(x[i,j].solution_value()) for j in range(g_type)])
        rate = sum([r[i][j]*res[-1][j] for j in range(g_type)])

    server_ = [sum([res[i][j] for i in range(m)]) for j in range(g_type)]

    return objective.Value(), res    

def option_ideal(q, server, m, g_type, pwr_par, rate_par, node_conf, func_):
    # server, m, g_type, pwr_par, rate_par
    # print(q)

    pk_pwr, i_clust = CA_ideal(server, m, g_type, pwr_par, rate_par, q)

    
    pwr = 0
    raw_pwr = 0
    res = []
    for i in range(m):
        tmp=node_selection_opt(q[i],i_clust[i], pwr_par[i], rate_par[i], node_conf[i], func_[i], 0)
        raw_pwr += sum(tmp[1])
        res.append(tmp)
    
    
    
    # pwr += sum(tmp[1]) + static_power * sum(tmp[0]) + sum([idle[k]*tmp[2][k]for k in range(g_type)])

    idle = [89,155,100]
    idle_server = []
    for g in range(g_type):
        idle_server.append(sum([res[i][0][g] for i in range(m)]))
    # print(idle_server)
    
    for i in range(m):
        pwr += sum(res[i][1]) + static_power * sum(res[i][0])
    pwr += sum([idle[k]*idle_server[k]for k in range(g_type)])

    raw_pwr += static_power * sum(server) + sum([idle[k]*server[k]for k in range(g_type)])
    return pwr, raw_pwr

if __name__ == '__main__':
    # Input log file name
    output_name = input("Log file name: ")

    # Load initial configration
    #   model : Served model
    #   gpu : GPU type
    #   server : Number of servers by GPU type
    #   p, r : Max power consumption(p) when arrival rate is r 
    #   service : Model for each service
    #   t_name : input trace file name(with path?)
    model, gpu, server, p, r, service, t_name, priority = read_config()
    
    # Number of service
    m = len(service)

    # Total number of server nodes
    n = sum(server)
    
    # Number of GPU type
    g_type = len(gpu) 

    # Read trace file - peak rate
    trace_file = open(t_name,"r")
    peak = list(map(float, trace_file.readline().split()))
    
    if len(peak) != m:
        print("Peak request rate doesn't match.")
        exit()
    
    print("Peak rates :", peak)
    
    # Read trace file - total time
    trace_l = int(trace_file.readline())
    
    # Read trace file - request rate for every unit time
    trace_raw = [[] for i in range(m)]
    for i in range(trace_l):
        qn = list(map(float,trace_file.readline().split()))
        for j in range(m):
            trace_raw[j].append(qn[j])
    trace_file.close()
    
    # Determine heuristic order
    order=sorted([[i,peak[i]] for i in range(len(peak))],key=lambda x: -x[1])
    order=sorted([order[i]+[i] for i in range(len(peak))],key=lambda x: x[0])
    order=[i[2] for i in order]
    print("Heuristic order :", order, "\n")

    # Power and request rate parameter
    pwr_par = []
    rate_par = []

    for idx in service:
        j = model.index(idx)
        pwr_par.append([p[i][j] for i in range(g_type)])
        rate_par.append([r[i][j] for i in range(g_type)])

    # [Approach 1] 
    # Cluster Assignment at Cloud-level
    #   pk_pwr : Maximum power consumption at peak rate
    #   x : ??
    #   clus_opt : Result of cluster assignment 
    pk_pwr, clus_opt = CA_opt(server, m, g_type, pwr_par, rate_par, peak)

    # Heuristic cluster assignment with fixed priority
    # clus_h : 
    clus_h = CA_heuristic(server, m, g_type, rate_par, peak, order, priority)
    
    # Idle server node
    '''
    a1_res = [server[i] - sum(clus_opt[j][i] for j in range(m)) 
                                        for i in range(len(server))]
    a1_res_h = [server[i] - sum(clus_h[j][i] for j in range(m)) 
                                        for i in range(len(server))]
    print(a1_res)
    print(a1_res_h)
    '''

    
    #
    a3g = {}
    for g in model:
        a3g[g] = read_model(g) # GPU 개수별 power and rate
    
    func_scale=read_power_data_s(service, gpu)
    func_dvfs=read_power_data_no_dvfs(service, gpu)
    func_default=read_power_data_default(service, gpu)
    
    # Length of simulation trace
    l = trace_l
    # l=2

    # #ideal
    # trace_ideal = []

    # for i in range(l):
    #     tmp = [trace_raw[j][i] for j in range(m)]
    #     trace_ideal.append(tmp) 

    # print()

    # res=parmap.map(option_ideal,trace_ideal, server, m, g_type, pwr_par, rate_par, [a3g[service[i]] for i in range(m)], func_scale, pm_pbar=True)

    # res1 = [i[0] for i in res]
    # res2 = [i[1] for i in res]
    # print(sum(res1)/j_to_kwh)
    # print(sum(res2)/j_to_kwh)

    
    trace = trace_raw

    # Option1 - Optimal with all approach
    param = [[l,max(os.cpu_count()//m,1),i,clus_opt[i],pwr_par[i],
            rate_par[i],trace[i],a3g[service[i]],func_scale[i]] for i in range(m)]
    
    pool = MyPool(m)
    result = pool.map(option1, param)
    pool.close()
    pool.join()

    op1_server = []
    op1_pwr = []
    op1_rate = []
    for i in range(len(result[0][0])):
        op1_server.append([result[idx][0][i] for idx in range(m)])
        op1_pwr.append([result[idx][1][i] for idx in range(m)])
        op1_rate.append([result[idx][2][i] for idx in range(m)])
    
    # Option2 - without optimal cluster assignment 
    param = [[l,max(os.cpu_count()//m,1),i,clus_h[i],pwr_par[i],
            rate_par[i],trace[i],a3g[service[i]],func_scale[i]] for i in range(m)]
    
    pool = MyPool(m)
    result = pool.map(option2, param)
    pool.close()
    pool.join()

    op2_server = []
    op2_pwr = []
    op2_rate = []
    for i in range(len(result[0][0])):
        op2_server.append([result[idx][0][i] for idx in range(m)])
        op2_pwr.append([result[idx][1][i] for idx in range(m)])
        op2_rate.append([result[idx][2][i] for idx in range(m)])
    
    # Option3
    param = [[l,max(os.cpu_count()//m,1),i,clus_h[i],rate_par[i],trace[i],
                                                func_scale[i]] for i in range(m)]

    pool = MyPool(m)
    result = pool.map(option3, param)
    pool.close()
    pool.join()    
    
    op3_server = []
    op3_pwr = []
    op3_rate = []
    for i in range(len(result[0][0])):
        op3_server.append([result[idx][0][i] for idx in range(m)])
        op3_pwr.append([result[idx][1][i] for idx in range(m)])
        op3_rate.append([result[idx][2][i] for idx in range(m)])

    

    # Option4
    param = [[l,max(os.cpu_count()//m,1),i,clus_h[i],rate_par[i],trace[i],
                                                func_dvfs[i]] for i in range(m)]

    pool = MyPool(m)
    result = pool.map(option3, param)
    pool.close()
    pool.join()    
    
    op4_server = []
    op4_pwr = []
    op4_rate = []
    for i in range(len(result[0][0])):
        op4_server.append([result[idx][0][i] for idx in range(m)])
        op4_pwr.append([result[idx][1][i] for idx in range(m)])
        op4_rate.append([result[idx][2][i] for idx in range(m)])
    
    # Option5
    param = [[l,max(os.cpu_count()//m,1),i,clus_h[i],rate_par[i],trace[i],
                                                func_default[i]] for i in range(m)]

    pool = MyPool(m)
    result = pool.map(option3, param)
    pool.close()
    pool.join()    
    
    op5_server = []
    op5_pwr = []
    op5_rate = []
    for i in range(len(result[0][0])):
        op5_server.append([result[idx][0][i] for idx in range(m)])
        op5_pwr.append([result[idx][1][i] for idx in range(m)])
        op5_rate.append([result[idx][2][i] for idx in range(m)])
    
    
    # Result

    # Only dynamic power of activated server.
    P_op1 = (sum([sum([sum(j) for j in i]) for i in op1_pwr]))/j_to_kwh
    P_op2 = (sum([sum([sum(j) for j in i]) for i in op2_pwr]))/j_to_kwh
    P_op3 = (sum([sum([sum(j) for j in i]) for i in op3_pwr]))/j_to_kwh
    P_op4 = (sum([sum([sum(j) for j in i]) for i in op4_pwr]))/j_to_kwh
    P_op5 = (sum([sum([sum(j) for j in i]) for i in op5_pwr]))/j_to_kwh
    # print("%.3f %.3f %.3f %.3f %.3f"%(P_op1, P_op2, P_op3, P_op4, P_op5))

    # + static power of activated server.
    op1_tmp = static_active_server(op1_server,m)
    op2_tmp = static_active_server(op2_server,m)
    op3_tmp = static_active_server(op3_server,m)
    op4_tmp = static_active_server(op4_server,m)
    op5_tmp = static_active_server(op5_server,m)
    
    print("%.3f %.3f %.3f %.3f %.3f"
        %(P_op1+op1_tmp, P_op2+op2_tmp, P_op3+op3_tmp, P_op4+op4_tmp, P_op5+op5_tmp))
    E_res = [[P_op1+op1_tmp, P_op2+op2_tmp, P_op3+op3_tmp, P_op4+op4_tmp, P_op5+op5_tmp]]

    idle_pwr=[89,155,100]
    # + static power of allocated server in cluster
    static_opt = (static_power*(l*sum([sum(i)for i in clus_opt]))
                + l*sum([sum([x*y for x,y in zip(i, idle_pwr)]) for i in clus_opt]))/j_to_kwh
    static_h = (static_power*(l*sum([sum(i)for i in clus_h]))
            + l*sum([sum([x*y for x,y in zip(i, idle_pwr)]) for i in clus_h]))/j_to_kwh

    print("%.3f %.3f %.3f %.3f %.3f"
        %(P_op1+static_opt, P_op2+static_h, P_op3+static_h, P_op4+static_h, P_op5+static_h))
    E_res+=[P_op1+static_opt, P_op2+static_h, P_op3+static_h, P_op4+static_h, P_op5+static_h]
    # + static power of entire servers
    idle = [89,155,100]
   
    p_tmp = (l*(sum([server[i]*(static_power+idle[i]) for i in range(g_type)])))/j_to_kwh
    print("%.3f %.3f %.3f %.3f %.3f"
        %(P_op1+p_tmp, P_op2+p_tmp, P_op3+p_tmp, P_op4+p_tmp, P_op5+p_tmp))
    E_res+=[P_op1+p_tmp, P_op2+p_tmp, P_op3+p_tmp, P_op4+p_tmp, P_op5+p_tmp]

    print(n)
    for i in range(len(E_res)):
        for j in range(5):
            E_res[i][j]/=E_res[i][-1]
            E_res[i][j] = round(E_res[i][j],1)
        for j in E_res[i]:
            print("%.3f "%j, end="")
        print()

    save_log("../res/"+output_name+"_op1", op1_server, op1_pwr, op1_rate)
    save_log("../res/"+output_name+"_op2", op2_server, op2_pwr, op2_rate)
    save_log("../res/"+output_name+"_op3", op3_server, op3_pwr, op3_rate)
    save_log("../res/"+output_name+"_op4", op4_server, op4_pwr, op4_rate)
    save_log("../res/"+output_name+"_op5", op5_server, op5_pwr, op5_rate)
    
