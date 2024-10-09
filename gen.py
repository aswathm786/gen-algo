#importing libraries
import pandas as pd
import openpyxl as xl
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import chart_studio.plotly as py
import plotly.figure_factory as ff
import datetime
import time
import copy


def Data_excel_json(excel_sheet):
    '''
    convert excel into json 
    
    excel_sheet: take excel sheet as input
    '''

    data_excel = xl.load_workbook(excel_sheet)
    data = {}
    sheet_name = data_excel.sheetnames
    for sheet in sheet_name:
        wb_sheet = data_excel[sheet]
        cell_values = wb_sheet.values
        df = pd.DataFrame(cell_values, columns=next(cell_values))
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x.strip())
        df.index = df.iloc[:, 0]
        df.drop(columns=df.columns[0], inplace=True)
        data[sheet] = df.T.to_dict()
    return data


def Json_to_df(json_data):
    '''
    convert json into df
    
    json_data: take json_data as input
    '''

    dict_data = {}
    for key in json_data.keys():
        dict_data[key] = pd.DataFrame(json_data.get(key)).T
    return dict_data


def Initialize(data_dict):
    '''
    returns all necessary information for the scheduling 
    
    data_dict: dictionary containing machine sequence and processing time for every job
    '''

    data_json = Json_to_df(data_dict)
    machine_sequence_tmp = data_json['Machines Sequence']
    process_time_tmp = data_json['Processing Time']

    df_shape = process_time_tmp.shape
    num_machines = df_shape[1]  # number of machines
    num_job = df_shape[0]  # number of jobs
    num_gene = num_machines * num_job  # number of genes in a chromosome

    process_time = [list(map(int, process_time_tmp.iloc[i]))
                    for i in range(num_job)]
    machine_sequence = [list(map(int, machine_sequence_tmp.iloc[i]))
                        for i in range(num_job)]

    return process_time, machine_sequence, num_gene, num_job, num_machines


def Generate_initial_population(population_size, num_machines, num_job):
    '''
    Generate initial population for the GA
    
    population_size: the number of individuals
    num_machines: the number of machines
    num_job: the number of jobs
    '''
    #operation based representation proposed by Gen et al. (Gen M, Tsujimura Y, and Kubota E 1994 Solving job-shop scheduling problems by genetic algorithm Systems,Man, and Cybernetics. 2 1577-1582. )
    #Example:
    # Chromosome: 3 | 2 | 2 | 1 | 1 | 2 | 3 | 1 | 3
    # decoding: 1st operation job 3 | 1st operation job 2 | 2nd operation job 2 | 1st operation job 1 | 2nd operation job 1 | etc..

    num_gene = num_machines * num_job
    population_list = []
    for i in range(population_size):
        # generate a random permutation of 0 to num_job*num_mc-1
        nxm_random_num = list(np.random.permutation(num_gene))
        population_list.append(nxm_random_num)  # add to the population_list
        for j in range(num_gene):
            # convert to job number format, every job appears m times
            population_list[i][j] = population_list[i][j] % num_job
    return population_list


def Two_point_crossover(population_size, num_gene, num_job, num_machines, process_time, machine_sequence, crossover_rate, population_list):
    '''
    Two-point crossover genetic operator
    
    population_size: the number of individuals
    num_gene: the number of genes for each chromosome
    num_job: the number of jobs
    num_machines: the number of machines
    process_time: duration of every operation
    machine_sequence: machine sequence for every job
    crossover_rate: probability of mating
    population_list: list of current population
    '''
    #Example:
    # Parent 1: [2,1,3|1,1,2,3|2,3]
    # Parent 2: [1,2,2|3,1,3,2|1,3]

    # Offspring 1: [2,1,3|3,1,3,2|2,3]
    # Offspring 2: [1,2,2|1,1,2,3|1,3]

    parent_list = copy.deepcopy(population_list)
    offspring_list = copy.deepcopy(population_list)
    # generate a random sequence to select the parent chromosome to crossover
    pop_random_size = list(np.random.permutation(population_size))

    for size in range(int(population_size/2)):
        crossover_prob = np.random.rand()
        if crossover_rate >= crossover_prob:
            parent_1 = population_list[pop_random_size[2*size]][:]
            parent_2 = population_list[pop_random_size[2*size+1]][:]

            child_1 = parent_1[:]
            child_2 = parent_2[:]
            cutpoint = list(np.random.choice(num_gene, 2, replace=False))
            cutpoint.sort()

            child_1[cutpoint[0]:cutpoint[1]
                    ] = parent_2[cutpoint[0]:cutpoint[1]]
            child_2[cutpoint[0]:cutpoint[1]
                    ] = parent_1[cutpoint[0]:cutpoint[1]]
            offspring_list[pop_random_size[2*size]] = child_1[:]
            offspring_list[pop_random_size[2*size+1]] = child_2[:]

    for pop in range(population_size):
        '''
        Repairment
        
        The number of occurrences of each job in a chromosome sequence is 10 times (since this is a 10X10 job shop problem)
        Due to crossover, the number of occurrences can be less or greater than 10(not feasible solution),
        the following part fixes the problem creating a feasible solution
        '''
        #Example:
        # Chromosome sequence for 3X3 job shop [2,1,3,3,1,3,2,2,3] Here 1 appears only two times while 3 appears four times
        # In order to fix this sequence we can substitute a 3 for a 1 resulting in this now feasible sequence: [2,1,1,3,1,3,2,2,3]
        # Now each occurrence appears 3 times

        job_count = {}
        # 'larger' record jobs appear in the chromosome more than pop times, and 'less' records less than pop times.
        larger, less = [], []
        for job in range(num_job):
            if job in offspring_list[pop]:
                count = offspring_list[pop].count(job)
                pos = offspring_list[pop].index(job)
                # store the above two values to the job_count dictionary
                job_count[job] = [count, pos]
            else:
                count = 0
                job_count[job] = [count, 0]

            if count > num_machines:
                larger.append(job)
            elif count < num_machines:
                less.append(job)

        for large in range(len(larger)):
            change_job = larger[large]
            while job_count[change_job][0] > num_machines:
                for les in range(len(less)):
                    if job_count[less[les]][0] < num_machines:
                        offspring_list[pop][job_count[change_job]
                                            [1]] = less[les]
                        job_count[change_job][1] = offspring_list[pop].index(
                            change_job)
                        job_count[change_job][0] = job_count[change_job][0]-1
                        job_count[less[les]][0] = job_count[less[les]][0]+1
                    if job_count[change_job][0] == num_machines:
                        break

    return offspring_list, parent_list, population_list


def Mutations(offspring_list, mutation_rate, mutation_selection_rate, num_gene):
    '''
    Mutation through gene displacement genetic operator

    offspring_list: list of new individuals generated through crossover
    mutation_rate: determines the percentage of individuals(chromosomes) to be mutated
    mutation_selection_rate: determines the percentage of genes in the chromosome to be mutated
    num_gene: number of genes in a chromosome
    '''
    #Example:
    # exchange order: 5 <- 2 <- 6
    # position:             1  2  3  4  5  6
    # chromosome:           3  5  1  6  2  4
    # mutated chromosome:   3  4  1  6  5  2

    num_mutation_jobs = round(num_gene * mutation_selection_rate)

    for off_spring in range(len(offspring_list)):

        mutation_prob = np.random.rand()

        if mutation_rate >= mutation_prob:
            # chooses the positions to mutate
            m_change = list(np.random.choice(
                num_gene, num_mutation_jobs, replace=False))
            # save the value which is on the first mutation position
            t_value_last = offspring_list[off_spring][m_change[0]]
            for i in range(num_mutation_jobs-1):
                # displacement
                offspring_list[off_spring][m_change[i]
                                           ] = offspring_list[off_spring][m_change[i+1]]
            # move the value of the first mutation position to the last mutation position
            offspring_list[off_spring][m_change[num_mutation_jobs-1]
                                       ] = t_value_last

    return offspring_list


def Fitness(offspring_list, parent_list, population_size, num_job, num_machines, process_time, machine_sequence):
    '''
    Calculate the fitness for every individual. Fitness is calculated as the completition time of each chromosome (makespan).
    Minimization problem, the lower the time the better

    offspring_list: list of new individuals generated through crossover
    parent_list: list of parents
    population_size: the number of individuals
    num_job: the number of jobs
    num_machines: the number of machines
    process_time: duration of every operation
    machine_sequence: machine sequence for every job
    '''

    #total_chromosome is a list containing all parents and children
    #chrom_fit: completition time of each chromosome
    #chrom_fitness: 1 / chrom_fit

    total_chromosome = copy.deepcopy(
        parent_list) + copy.deepcopy(offspring_list)
    chrom_fitness, chrom_fit = [], []
    total_fitness = 0
    for pop_size in range(population_size*2):
        j_keys = [j for j in range(num_job)]
        key_count = {key: 0 for key in j_keys}
        j_count = {key: 0 for key in j_keys}
        m_keys = [j+1 for j in range(num_machines)]
        m_count = {key: 0 for key in m_keys}

        for i in total_chromosome[pop_size]:
            gen_t = int(process_time[i][key_count[i]])
            gen_m = int(machine_sequence[i][key_count[i]])
            j_count[i] = j_count[i] + gen_t
            m_count[gen_m] = m_count[gen_m] + gen_t

            if m_count[gen_m] < j_count[i]:
                m_count[gen_m] = j_count[i]
            elif m_count[gen_m] > j_count[i]:
                j_count[i] = m_count[gen_m]

            key_count[i] = key_count[i] + 1

        makespan = max(j_count.values())
        chrom_fitness.append(1/makespan)
        chrom_fit.append(makespan)
        total_fitness = total_fitness + chrom_fitness[pop_size]

    return total_fitness, total_chromosome, chrom_fitness, chrom_fit


def Selection(population_size, total_chromosome, total_fitness, chrom_fitness, population_list):
    '''
    Standard roulette wheel operator for selecting parents to generate new offspring.
    Fitness is used to associate a probability of selection with each individual chromosome.
    From total_chromosome (parent+children) select individuals with better fitness based on roulette wheel approach
    and update population_list

    population_size: the number of individuals
    total_chromosome: list containing all parents and children
    total_fitness: list of the fitness of all parents and children 
    chrom_fitness: 1 / completition time (for every individual)
    population_list: the selected best individuals
    '''
    #Example:
    # if we have a population size of 10, after crossover we have 10 new individuals(children) for a total of 20 individuals
    # and now we select 10 individuals based on roulette approach for the new generation

    # pk is a list of probabilites proportional to the fitness of each chromosome
    # qk same list as before but cumulative

    # pk = [0.2, 0.1, 0.4, 0.1, 0.2]
    # qk = [0.2, 0.3, 0.7, 0.8, 1]

    pk, qk = [], []

    for size in range(population_size * 2):
        pk.append(chrom_fitness[size] / total_fitness)
    for size in range(population_size * 2):
        cumulative = 0

        for j in range(0, size+1):
            cumulative = cumulative + pk[j]
        qk.append(cumulative)

    selection_rand = [np.random.rand() for i in range(population_size)]

    for pop_size in range(population_size):
        if selection_rand[pop_size] <= qk[0]:
            population_list[pop_size] = copy.deepcopy(total_chromosome[0])
        else:
            for j in range(0, population_size * 2 - 1):
                if selection_rand[pop_size] > qk[j] and selection_rand[pop_size] <= qk[j+1]:
                    population_list[pop_size] = copy.deepcopy(
                        total_chromosome[j+1])
                    break

def Comparison(population_size, chrom_fit, total_chromosome, Tbest_now, Tbest, makespan_record, sequence_best):
    '''
    Compare completition time for each chromosome, select best solution found in this generation and compare it
    with best solution found so far. If the solution found in this generation is better than the previous best
    solution, then replace Tbest and save result.

    population_size: the number of individuals
    chrom_fit: completition time of each chromosome
    total_chromosome: list containing all parents and children
    Tbest_now: fitness of best solution found in current generation
    Tbest: fitness of best solution found so far
    makespan_record: list containing the best individual from each generation
    sequence_best: best solution found
    ''' 
    
    for pop_size in range(population_size * 2):
        if chrom_fit[pop_size] < Tbest_now:
            Tbest_now = chrom_fit[pop_size]
            sequence_now = copy.deepcopy(total_chromosome[pop_size])
    if Tbest_now <= Tbest:
        Tbest = Tbest_now
        sequence_best = copy.deepcopy(sequence_now)

    makespan_record.append(Tbest)
    
    return sequence_best, makespan_record, Tbest


def Results_makespan(sequence_best, makespan_record, Tbest, start_time):
    '''
    After evolutuionary process, output best scheduling found 
    and plot best solution at each generation
    
    sequence_best: best solution found
    makespan_record: list containing the best individual from each generation
    Tbest: fitness of best solution found so far
    start_time: the time the algorithm started
    '''

    # print("optimal sequence", sequence_best)
    print("optimal value: %.2f seconds" % Tbest)
    print("the elapsed time: %.2f seconds" % (time.time() - start_time))

    print("\n")

    plt.plot([i for i in range(len(makespan_record))], makespan_record, 'b')
    plt.ylabel('makespan', fontsize=15)
    plt.xlabel('generation', fontsize=15)
    plt.show()


def Plot_gantt(num_machines, num_job, sequence_best, process_time, machine_sequence):
    '''
    Take as input the best sequence found and create Gantt chart
    
    num_machines: the number of machines
    num_job: the number of jobs
    sequence_best: best solution found
    process_time: duration of every operation
    machine_sequence: machine sequence for every job
    ''' 

    m_keys = [j+1 for j in range(num_machines)]
    j_keys = [j for j in range(num_job)]
    key_count = {key:0 for key in j_keys}
    j_count = {key:0 for key in j_keys}
    m_count = {key:0 for key in m_keys}
    j_record = {}
    
    for i in sequence_best:
        gen_t = int(process_time[i][key_count[i]])
        gen_m = int(machine_sequence[i][key_count[i]])
        j_count[i] = j_count[i] + gen_t
        m_count[gen_m] = m_count[gen_m] + gen_t

        if m_count[gen_m] < j_count[i]:
            m_count[gen_m] = j_count[i]
        elif m_count[gen_m] > j_count[i]:
            j_count[i] = m_count[gen_m]

        start_time = str(datetime.timedelta(seconds = j_count[i] - process_time[i][key_count[i]])) # convert seconds to hours, minutes and seconds
        end_time = str(datetime.timedelta(seconds = j_count[i]))

        j_record[(i, gen_m)] = [start_time, end_time]

        key_count[i] = key_count[i] + 1

    df = []
    for m in m_keys:
        for j in j_keys:
            df.append(dict(Task='Machine %s'%(m), Start='2021-05-01 %s'%(str(j_record[(j,m)][0])), \
                            Finish='2021-05-01 %s'%(str(j_record[(j,m)][1])), Resource='Job %s'%(j+1)))
    
    df_ = pd.DataFrame(df)
    
    df_.Start = pd.to_datetime(df_['Start'])
    df_.Finish = pd.to_datetime(df_['Finish'])
    start = df_.Start.min()
    end = df_.Finish.max()
    df_['Duration'] = (df_.Finish - df_.Start).dt.total_seconds()
    
    

    df_.Start = df_.Start.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    df_.Finish = df_.Finish.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
    data = df_.to_dict('records')
    machine_durations = {}

    # Calculate total duration for each machine
    for entry in data:
        machine = entry['Task']
        duration = entry['Duration']
        if machine in machine_durations:
            machine_durations[machine] += duration
        else:
            machine_durations[machine] = duration

    # Print the total duration for each machine
    for machine, total_duration in machine_durations.items():
        print(f"{machine} -> {total_duration:.0f} seconds")

    # Load the Excel file with energy consumption rates
    energy_data = pd.read_excel('machine_energy_rates.xlsx')

    # Operation times for each machine (in seconds)


    # Calculate energy consumption for each machine
    energy_consumption = {}
    total_energy_consumed = 0
    for machine, time in machine_durations.items():
        rate = energy_data.loc[energy_data['Machine'] == machine, 'Energy Rate (kWh/sec)'].values[0]
        energy_consumption[machine] = time * rate
        total_energy_consumed += time*rate

    # Convert the results to a DataFrame
    energy_consumption_df = pd.DataFrame(list(energy_consumption.items()), columns=['Machine', 'Energy Consumption (kWh)'])

    # Save the results to a new Excel file
    print(energy_consumption_df)
    print(f"Total energy consumed: {total_energy_consumed:.2f} kWh")


    

    final_data = {
        'start': start.strftime('%Y-%m-%dT%H:%M:%S'),
        'end': end.strftime('%Y-%m-%dT%H:%M:%S'),
        'data': data
    }
        
    fig = ff.create_gantt(df_, index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True, title='Job Shop Schedule')
    fig.show()
    
    
    return final_data, df_


def GA(data, population_size, crossover_rate, mutation_rate, mutation_selection_rate, num_iteration):
    '''
    Main of Genetic Algorithm
    
    data: the data of the JSS problem 
    population_size: the number of individuals
    crossover_rate: probability of mating
    mutation_rate: determines the percentage of individuals(chromosomes) to be mutated
    mutation_selection_rate: determines the percentage of genes in the chromosome to be mutated
    num_iteration: the number of generations
    '''
    start_time = time.time()

    process_time, machine_sequence, num_gene, num_job, num_machines = Initialize(
        data)

    Tbest = 999999999999999
    makespan_record = []
    sequence_best = []

    population_list = Generate_initial_population(
        population_size, num_machines, num_job)

    for iteration in range(num_iteration):

        Tbest_now = 99999999999

        offspring_list, parent_list, population_list = Two_point_crossover(
            population_size, num_gene, num_job, num_machines, process_time, machine_sequence, crossover_rate, population_list)
        offspring_list = Mutations(
            offspring_list, mutation_rate, mutation_selection_rate, num_gene)
        total_fitness, total_chromosome, chrom_fitness, chrom_fit = Fitness(
            offspring_list, parent_list, population_size, num_job, num_machines, process_time, machine_sequence)
        Selection(population_size, total_chromosome,
                  total_fitness, chrom_fitness, population_list)
        sequence_best, makespan_record, Tbest = Comparison(
            population_size, chrom_fit, total_chromosome, Tbest_now, Tbest, makespan_record, sequence_best)

    Results_makespan(sequence_best, makespan_record, Tbest, start_time)
    final_data, df = Plot_gantt(
        num_machines, num_job, sequence_best, process_time, machine_sequence)
    return sequence_best, Tbest


data = Data_excel_json('./JSP_dataset.xlsx')


# Main

population_size = 30
crossover_rate = 0.8
mutation_rate = 0.2
mutation_selection_rate = 0.2
generation = 1000

#930 is the optimal value of this benchmark
sequence_best, time_best = GA(
    data, population_size, crossover_rate, mutation_rate, mutation_selection_rate, generation)

