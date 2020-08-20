The bad seed environment is broken into several trials. 

The most important files are in SULI/src/:
combined.py : this is where the combined system is. This system is also the most heavily commented. It is parameterized so that you can say how many bad seeds there are, as well as controlling confidence interval, the minimum number of measurements to look at confidence and so on.
          The state here was:
                [the index of the sample with the highest standard deviation, a history of the past badSeedCount - 1 samples chosen, a list of bad seeds (where if null, represented by the highest sample index + 1)]
                please note that the learning suffers when the badSeedCount is 5 or above
tensorForceEnv.py : this was focused on solving the no repeats problem, in which the state was [history of the past 2 samples chosen]
simulation.py :  this was focused on solving the standard deviation problem, in which the state was [indices of samples with top 3 standard deviations]


The other files:
a2c_test.py : built my own a2c agent for cartpole
bad_Seed_env.py : first attempt at environment building, using gym, failed
extraCalc.py : just some arrays I needed to run
goodSeed.py : unfinnished idea about going through the good seeds first, checking them off as good once they have reached confidence
refraction.py : file that helped me set up cookie cutter format
rlpytEnv.py :  another failed attempt using rlpyt as the environment library
standardDev.py : my attempt at an environment before dividing the problem. Minimal learning happens here.

Please note that my front end design concept is here:
https://www.figma.com/file/zcFOWq9VgoGqo0tZ4Gb0ns/SULI_Design?node-id=0%3A1
    (Note that the prototyping is not fully done, but has some use in present mode)
