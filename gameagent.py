#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('cd', 'gymretro')


# In[2]:


#pip install -r requirements.txt


# In[3]:


import retro
import numpy as np
import cv2
import neat
import pickle


# In[4]:


env = retro.make(game='Airstriker-Genesis')
obs = env.reset()
done = False
imgarray = []


# In[17]:


def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        obs = env.reset()
        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        counter = 0
        score = 0
        score_max = 0
        exitcounter = 0

        done = False
        flag = True
        #cv2.namedWindow('main', cv2.WINDOW_NORMAL)
        while not done:

            #env.render()
            obs = cv2.resize(obs, (inx, iny))
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            #windows
            obs1 = cv2.resize(obs, (224, 320))
            
            
           # cv2.imshow('main', obs)
           # cv2.waitKey(1)
            obs = np.reshape(obs, (inx, iny))

            imgarray = np.ndarray.flatten(obs)
            nnOutput = net.activate(imgarray)
            #print(nnOutput)
            #print(len(imgarray), nnOutput)
            obs, reward, done, info = env.step(nnOutput)
            
            score = info['score']
            lives = info['lives']
            
            if lives  < 3 and flag == True:
                reward -= 10
                flag = False
                done = True
            if score > score_max:
                fitness_current += 1
                score_max = score
            fitness_current += reward / 10

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
                exitcounter = 0
            else:
                counter += 1
                exitcounter += 1
                
            if counter >= 300:
                reward -= 20
                counter = 0
            #print(counter)
            if done or exitcounter == 1500:
                done = True
                reward -= 30
                print(genome_id, fitness_current)
            genome.fitness = fitness_current
        flag = True


# In[18]:


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')


# In[19]:


p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-62')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))


# In[20]:


model = p.run(eval_genomes)




