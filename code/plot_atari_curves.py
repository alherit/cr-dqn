#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dqn_zoo.atari_data import get_human_normalized_score


import matplotlib


import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import pandas as pd

import seaborn as sns

plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.titlesize'] = 18

matplotlib.rcParams['pdf.fonttype'] = 42



def plotCurveEvalTrain(basedir,n_trials, algorithms, base_dqnzoo_dir, dqnzoo_algos, games, 
              dico=None , color_dico = None, human = False, smooth=False, 
              window=5, legendOnlyFor="all", max_x = None, eval=False):
    print("n_trials",n_trials)

    if games is None:
        games = next(os.walk(basedir+"/" + algorithms[0]))[1]
        games.sort()
        print("games: " ,games)
    
    if eval:
        dqnzoo_dir = base_dqnzoo_dir + "/eval_all_seeds"
        score = "eval_episode_return"
        evaltrain = "eval"
        if human:
            perf = "eval_normalized_return"
            y_label = "Human Normalized Score (eval)"
        else:
            perf = "eval_episode_return"
            y_label = "Score (Eval)"
    else:
        dqnzoo_dir = base_dqnzoo_dir + "/train_all_seeds"
        score = "train_episode_return"
        evaltrain = "train"
        if human:
            perf = "train_normalized_return"
            y_label = "Human Normalized Score (training)"
        else:
            perf = "train_episode_return"
            y_label = "Score (training)"

        
    

    
    for game in games:
        dfs = []
        for trial in range(1,n_trials+1):
        
            for algo in algorithms:
                if algo in ["qr_dqn_0","nc_qr_dqn","nc_qr_dqn_0_adj_eps_scaled_grad"]: 
                    fname = "resultsnoCramer_ncLR0.00005_s" + str(trial) + ".csv"
                else:
                    fname = "resultsCramer_ncLR0.00005_s" + str(trial) + ".csv"

                try:
                    df = pd.read_csv(basedir +"/" + algo + "/" + game + "/" +  fname, sep = ",")
                    df["algorithm"] = algo
                    df["trial"] = trial
                    
                    if human:
                        df[perf] = get_human_normalized_score(game ,df[score])
                   
                     
                    dfs.append(df)    
    
                except Exception as e:
                    print("error: " + str(e))
            
        
        df_zoo = pd.read_csv(dqnzoo_dir+ "/curves_" + game + ".csv", sep = ",")
        for trial in range(1,n_trials+1):
            for algo in dqnzoo_algos:
                colname = " "+algo+"_"+str(trial)
                df = df_zoo[["iteration",colname]].copy()
                df[score] = pd.to_numeric(df[colname],errors='coerce')
                if human:
                    df[perf] = get_human_normalized_score(game ,df[score])
                    
                df["algorithm"] = algo
                df["trial"] = trial
                dfs.append(df)    

        df = pd.concat(dfs)
        print(game)
        print("number of started runs: ", sum(df["iteration"]==0))
        print("number of finished runs: ", sum(df["iteration"]==200))
        df0 = df[df["iteration"]==0][["algorithm","trial"]]
        df200 = df[df["iteration"]==200][["algorithm","trial"]]
        print(pd.concat([df0,df200]).drop_duplicates(keep=False))
        

        if dico is not None:
            df["algorithm"] = df["algorithm"].replace(dico, regex=True)

        if max_x is not None:
            df = df[df["iteration"]<=max_x]

        if smooth:
            # seaborn uses numpy and ddof=0 by default while pandas uses ddof=1 by default
            df = df.groupby(['algorithm','iteration']).agg(['mean','std'])[perf]
            rolled = df.groupby(['algorithm']).rolling(window,min_periods =1).mean().reset_index(0,drop=True)

            fig = plt.figure()  # create a figure object
            ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
            

            algos = algorithms+dqnzoo_algos
            algos = [dico[alg] if alg in dico else alg for alg in algos]


            for algo in sorted(algos,reverse=True):
                rolled_algo = rolled.query("algorithm=='"+algo+"'")
                ax.plot(rolled_algo.index.get_level_values('iteration'), rolled_algo["mean"], color=color_dico[algo], label=algo);
                ax.fill_between(rolled_algo.index.get_level_values('iteration'), rolled_algo["mean"] - rolled_algo["std"], rolled_algo["mean"] + rolled_algo["std"], color=color_dico[algo], alpha=0.2);

        else:
            plt.figure()
            if False:
                ax = sns.lineplot(x="iteration", y=perf,
                             hue="algorithm", ci="sd",
                             data=df, palette = color_dico, hue_order = sorted(df["algorithm"].unique(), reverse=True))

            else:
                ax = sns.lineplot(x="iteration", y=perf,
                             hue="algorithm", units="trial",     estimator=None, lw=1,
                             data=df, palette = color_dico, hue_order = sorted(df["algorithm"].unique(), reverse=True))
                # Put the legend out of the figure
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    
    
        if False: #remove axis labels
            ax.set_ylabel('')    
            ax.set_xlabel('')
        else:
            ax.set_ylabel(y_label)    
            ax.set_xlabel('Million of Samples')
            

        if legendOnlyFor=="all" or legendOnlyFor==game:
            ax.legend().set_title('')
            handles, labels = ax.get_legend_handles_labels()
            # sort both labels and handles by labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            ax.legend(handles, labels)

        ax.set_title(game)
        
        
        plt.savefig(game+"_"+evaltrain+"_perf_ntrials_" + str(n_trials)+".pdf", bbox_inches = 'tight')




def plotCurveAll(basedir,n_trials, n_trials_us, algorithms, dqnzoo_dir, dqnzoo_algos, games, dico=None, color_dico=None , n_cols=None, smooth=True, window=5):
    print("n_trials",n_trials)
    print("n_trials_us",n_trials_us)


    if games is None:
        games = next(os.walk(basedir+"/" + algorithms[0]))[1]
        games.sort()
        print("games: " ,games)
        
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(len(games))))
       
    n_rows =  int(np.ceil(len(games)/n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(16.6,23.4))


    i = 0
    for game in games:
        missing_seed=""
        dfs = []
        for trial in range(1,n_trials_us+1):
            fname = "resultsCramer_ncLR0.00005_s" + str(trial) + ".csv"
        
            for algo in algorithms:
                try:
                    df = pd.read_csv(basedir +"/" + algo + "/" + game + "/" +  fname, sep = ",")
                    df["algorithm"] = algo
                    df["trial"] = trial
                    
                   
                    dfs.append(df)    
    
                except Exception as e:
                    print("error: " + str(e))
                    missing_seed="*"
                    
            
        df_zoo = pd.read_csv(dqnzoo_dir+ "/train_all_seeds/curves_" + game + ".csv", sep = ",")
        for trial in range(1,n_trials+1):
            for algo in dqnzoo_algos:
                colname = " "+algo+"_"+str(trial)
                df = df_zoo[["iteration",colname]].copy()
                df["train_episode_return"] = pd.to_numeric(df[colname],errors='coerce')
                df["algorithm"] = algo
                df["trial"] = trial

                dfs.append(df)    

        df = pd.concat(dfs)

        if dico is not None:
            df["algorithm"] = df["algorithm"].replace(dico, regex=True)

        ax = axs[int(np.ceil(i//n_cols)), i%n_cols]


        if smooth:
            # seaborn uses numpy and ddof=0 by default while pandas uses ddof=1 by default
            df = df.groupby(['algorithm','iteration']).agg(['mean','std'])["train_episode_return"]
            rolled = df.groupby(['algorithm']).rolling(window,min_periods =1).mean().reset_index(0,drop=True)

         
            # put our algorithms last to draw curves on top
            algos = dqnzoo_algos+algorithms
            algos = [dico[alg] if alg in dico else alg for alg in algos]

            for algo in algos:
                rolled_algo = rolled.query("algorithm=='"+algo+"'")
                ax.plot(rolled_algo.index.get_level_values('iteration'), rolled_algo["mean"], color=color_dico[algo], label=algo);
                ax.fill_between(rolled_algo.index.get_level_values('iteration'), rolled_algo["mean"] - rolled_algo["std"], rolled_algo["mean"] + rolled_algo["std"], color=color_dico[algo], alpha=0.2);

        else:
            break

        ax.set_title(game+missing_seed)
        ax.set_ylabel('')    
        ax.set_xlabel('')
        ax.yaxis.set_ticklabels([])   
        ax.legend().set_visible(False)


        i += 1
 
    for j in range(i,n_rows*n_cols):
        axs[int(np.ceil(j//n_cols)), j%n_cols].axis('off')   
 
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower right", fontsize=30, frameon=False, ncol=2) #(0.85, 0.65)
    
    for legobj in leg.legendHandles:
        legobj.set_linewidth(10.0)
    
    plt.tight_layout()
    #plt.show()
    plt.savefig("all_trainperf_ntrials_" + str(n_trials)+"_ntrials_us_" + str(n_trials_us)+".pdf", bbox_inches = 'tight')





def best_agent(basedir, n_trials, n_trials_us, algorithms, dqnzoo_dir, dqnzoo_algos, games=None):
 #For n_trials_us seeds for us and for n_trials seeds for the others:
#	Take max evaluation score over iterations for each game
#	Take median/mean over games
        
 
    if games is None:
        games = next(os.walk(basedir+"/" + algorithms[0]))[1]
        games.sort()
        print("games: " ,games)

    print("Number of games: ", len(games))

    i=1
    dfs = []
    for game in games:
        for trial in range(1,n_trials_us+1):
            for algo in algorithms: #= "symm_cramer"
    
                fname = "resultsCramer_ncLR0.00005_s" + str(trial) + ".csv"


                try:
                    df = pd.read_csv(basedir +"/" + algo + "/" + game + "/" +  fname, sep = ",")
                    df["algo"] = algo
                    df["trial"] = trial
                    df["game"] = game
                    
                    df["normalized_return"] = get_human_normalized_score(game ,df["eval_episode_return"])
                   
                    dfs.append(df)    
    
                except Exception as e:
                    print("error: " + str(e))
                    missing_seed="*"
                    


       
        df_zoo = pd.read_csv(dqnzoo_dir+ "/eval_all_seeds/curves_"+ game + ".csv", sep = ",")  #eval_normalized_all_seeds
        for trial in range(1,n_trials+1):
            for algo in dqnzoo_algos:
                
                colname = " "+algo+"_"+str(trial)
                df = df_zoo[["iteration",colname]].copy()
                df["eval_episode_return"] = pd.to_numeric(df[colname],errors='coerce')
                df["normalized_return"] = get_human_normalized_score(game ,df["eval_episode_return"])

                
                
                df["game"] = game
                df["algo"] = algo
                df["trial"] = trial
                df.drop(columns=colname, inplace=True)

                dfs.append(df)   
        print(game)
        print(len(pd.concat(dfs)))
        print(i*201*(n_trials*len(dqnzoo_algos)+n_trials_us*len(algorithms)))
        i=i+1
        


    df = pd.concat(dfs)

    grouped = df.groupby(['game','algo','trial'])	
    df_max = grouped.max().filter(['normalized_return']).reset_index()	
    grouped = df_max.groupby(['algo','trial'])	
    df_median= grouped.median().reset_index()	
    	

    df_mean= grouped.mean().reset_index()	
    
    
    return df, df_median, df_mean
    




dico3 = {"nc_cr_dqn_adj_eps":"NC-CR-DQN","nc_qr_dqn_0_adj_eps_scaled_grad":"NC-QR-DQN-0", 
         "nc_qr_dqn": "NC-QR-DQN-1",
         "cr_dqn_adj_eps":"CR-DQN","qrdqn":"QR-DQN-1", "qr_dqn_0":"QR-DQN-0", "dqn":"DQN", 
         "c51":"C51", "iqn":"IQN"
         } 



pal = sns.color_palette("colorblind")

color_dico3 = { 'QR-DQN-1':pal[0], 'QR-DQN-0':pal[2],'CR-DQN':'k',
                "DQN":pal[7],"NC-CR-DQN":"tab:cyan", "NC-QR-DQN-1":pal[6],"NC-QR-DQN-0":pal[3] ,
                'C51':pal[5], "NC-QR-DQN-0-SC":"k", "IQN":pal[4],
                "NC-CR-DQN-SC": pal[8] ,
                }



### DQN_ZOO results in csv format
base_dqn_zoo_folder = "../dqn_zoo_curves"


### here one folder per games is expected
our_results_folder = "../our_results"


if True:

    df, df_median, df_mean = best_agent(our_results_folder, 
               5, 3, ["cr_dqn_adj_eps"], 
               base_dqn_zoo_folder, 
               ["dqn","qrdqn","iqn","c51"], 
               games= None
               )


    print("Best agent protocol (median): ", df_median)

    print("Best agent protocol (median): ",     df_median.groupby(['algo']).mean().reset_index()	)



if True: 
    plotCurveAll(our_results_folder,5,3,
                ["cr_dqn_adj_eps"],
               base_dqn_zoo_folder,
               ["qrdqn","iqn", "c51"],
               None,dico3, color_dico3, n_cols=6, smooth=True, window=5)



if True: 
    plotCurveEvalTrain(our_results_folder,3,
              ["cr_dqn_adj_eps", 
               "qr_dqn_0",
               "nc_qr_dqn",
               "nc_cr_dqn_adj_eps",
               "nc_qr_dqn_0_adj_eps_scaled_grad",
               ] ,       
              base_dqn_zoo_folder,
              ["qrdqn","dqn"], 
              ["breakout","frostbite","asterix",
               "seaquest"] ,dico3, color_dico3, 
              human=True, smooth=True, window=5, legendOnlyFor="asterix", eval=False) 



