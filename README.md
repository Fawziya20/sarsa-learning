ddsss# SARSA Learning Algorithm


To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the 


## SARSA LEARNING FUNCTION
```pythonxx
def sarsa(env,ssssann
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):ss
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state,Q,epsilon: 
    			np.argmax(Q[state]) 
    			if np.random.random() > epsilon 
                else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)

    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)

    for e in tqdm(range(n_episodes),leave=False):
        state, done = env.reset(), False
        action = select_action(state,Q,epsilons[e])

        while not done:
            next_state,reward,done,_ = env.step(action)
            next_action = select_action(next_state,Q,epsilons[e])

            td_target = reward+gamma*Q[next_state][next_action]*(not done)

            td_error = td_target - Q[state][action]

            Q[state][action] = Q[state][action] + alphas[e] * td_error

            state, action = next_state,next_action

        Q_track[e] = Q
        pi_track.append(np.argmax(Q,axis=1))

    V = np.max(Q,axis=1)
    pi = lambda s: {s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:

![image](https://github.com/Fawziya20/sarsa-learning/assets/75235022/da87a9a0-4742-4472-99e8-8b07468fd9a2)

![image](https://github.com/Fawziya20/sarsa-learning/assets/75235022/7e854404-0cd2-4740-8e04-8c5a480b8f6f)

![image](https://github.com/Fawziya20/sarsa-learning/assets/75235022/8a4d4de3-2611-4883-a97f-46e12bef3ca9)

![image](https://github.com/Fawziya20/sarsa-learning/assets/75235022/838be182-8078-4949-b730-ccdcfb1e8e7b)

![image](https://github.com/Fawziya20/sarsa-learning/assets/75235022/7d90a6a9-d49a-4147-9d81-d571ec134b27)



## RESULT:
Thus the optimal policy for the given RL environment is found using SARSA-Learning and the state values are compared with the Monte Carlo method.
