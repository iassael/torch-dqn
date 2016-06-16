--[[

    DQN GridWorld
    by Yannis Assael

]] --

-- Configuration
cmd = torch.CmdLine()
cmd:text()
cmd:text('DQN GridWorld')
cmd:text()
cmd:text('Options')

-- general options:
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 4, 'number of threads')

-- gpu
cmd:option('-cuda', 0, 'cuda')

-- game
cmd:option('-puddles', 1, 'PuddleWorld')

-- model
cmd:option('-gamma', 0.99, 'discount factor')
cmd:option('-eps_start', 0.5, 'start ε-greedy policy')
cmd:option('-eps_end', 0.05, 'final ε-greedy policy')
cmd:option('-eps_endt', 100, 'final ε-greedy policy episode')
cmd:option('-learn_start', 1, 'start learning episode')
cmd:option('-replay_memory', 1e+5, 'experience replay memory')
cmd:option('-action_gap', 1, 'increase the action gap')
cmd:option('-action_gap_alpha', 0.9, 'action gap alpha parameter')

-- training
cmd:option('-bs', 32, 'batch size')
cmd:option('-nepisodes', 1000, 'number of episodes')
cmd:option('-nsteps', 1000, 'number of steps')
cmd:option('-target_gamma', 1e-2, 'target network updates')
cmd:option('-target_step', 100, 'target network updates')

cmd:option('-step', 10, 'print every episodes')

cmd:option('-plot', 0, 'plot q values')

cmd:text()

opt = cmd:parse(arg)

opt.bs = math.min(opt.bs, opt.replay_memory)

-- Requirements
require 'nn'
require 'optim'
local kwargs = require 'include.kwargs'
local log = require 'include.log'

-- Set float as default type
math.randomseed(opt.seed)
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

-- Cuda initialisation
if opt.cuda > 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.cuda)
    opt.dtype = 'torch.CudaTensor'
    print(cutorch.getDeviceProperties(opt.cuda))
else
    opt.dtype = 'torch.FloatTensor'
end

-- Initialise game
local env = (require 'game.GridWorld') {
    puddles = opt.puddles == 1
}
local a_space = env:getActionSpec()[3]
local r_space = { env:getRewardSpec() }
local s_space = env:getStateSpec()

-- Initialise model
local exp = (require 'model.model') {
    a_size = a_space[2],
    s_size = #s_space,
    dtype = opt.dtype
}

local model = exp.model
local model_target = exp.model:clone()
local params, gradParams = model:getParameters()
local params_target, _ = model_target:getParameters()

-- Initialise criterion
local criterion = nn.MSECriterion():type(opt.dtype)
criterion.sizeAverage = false

-- Optimisation function
local optim_func, optim_config = exp.optim()
local optim_state = {}

-- Initialise aux vectors
local td_err = torch.Tensor(opt.bs, a_space[2]):type(opt.dtype)

local train_r_episode = torch.zeros(opt.nsteps)
local train_q_episode = torch.zeros(opt.nsteps)

local train_r = 0
local train_r_avg = 0

local train_q = 0
local train_q_avg = 0

local test_r = 0
local test_r_avg = 0
local test_r_all = torch.zeros(opt.nepisodes)

local step_count = 0
local replay = {}

-- Training values storage
local train = {
    s_t = torch.Tensor(opt.bs, #s_space):type(opt.dtype),
    s_t1 = torch.Tensor(opt.bs, #s_space):type(opt.dtype),
    r_t = torch.Tensor(opt.bs):type(opt.dtype),
    a_t = torch.Tensor(opt.bs):type(opt.dtype),
    terminal = torch.Tensor(opt.bs):type(opt.dtype)
}

-- start time
local beginning_time = torch.tic()

for e = 1, opt.nepisodes do

    -- ε-greedy annealing
    opt.eps = (opt.eps_end +
                math.max(0, (opt.eps_start - opt.eps_end) * (opt.eps_endt -
                math.max(0, e - opt.learn_start)) / opt.eps_endt))

    -- Initial state
    local episode = {}
    episode.s_t = torch.Tensor(env:start())
    episode.terminal = false

    -- Initialise clock
    local time = sys.clock()

    -- Run for N steps
    local step = 1
    while step <= opt.nsteps and not episode.terminal do

        -- Compute Q values
        local q = model:forward(episode.s_t:type(opt.dtype)):clone()

        -- Pick an action (ε-greedy)
        if torch.uniform() < opt.eps then
            episode.a_t = torch.random(a_space[2])
        else
            local max_q, max_a = torch.max(q, 2)
            episode.a_t = max_a:squeeze()
        end

        --compute reward for current state-action pair        
        episode.r_t, episode.s_t1, episode.terminal = env:step(episode.a_t)
        episode.s_t1 = torch.Tensor(episode.s_t1)

        -- Store rewards
        train_r_episode[step] = episode.r_t

        -- Store current step
        local r_id = (step_count % opt.replay_memory) + 1
        replay[r_id] = {
            r_t = episode.r_t,
            a_t = episode.a_t,
            s_t = episode.s_t,
            s_t1 = episode.s_t1,
            terminal = episode.terminal and 1 or 0
        }

        -- Fetch from experiences
        local q_next, q_next_max
        if #replay >= opt.bs then

            for b = 1, opt.bs do
                local exp_id = torch.random(#replay)
                train.r_t[b] = replay[exp_id].r_t
                train.a_t[b] = replay[exp_id].a_t
                train.s_t[b] = replay[exp_id].s_t
                train.s_t1[b] = replay[exp_id].s_t1
                train.terminal[b] = replay[exp_id].terminal
            end

            -- Compute Q
            q = model:forward(train.s_t):clone()

            -- Use target network to predict q_max
            q_next = model_target:forward(train.s_t1)
            q_next_max = q_next:max(2):squeeze(2)

            -- Check if terminal state
            for b = 1, opt.bs do
                if train.terminal[b] == 1 then
                    q_next[b] = 0
                    q_next_max[b] = 0
                end
            end

            -- Q learnt value
            td_err:zero()
            for b = 1, opt.bs do
                td_err[{ { b }, { train.a_t[b] } }] = train.r_t[b] + opt.gamma * q_next_max[b] - q[b][train.a_t[b]]
            end

            -- Increase the action gap
            if opt.action_gap == 1 then
                local q_target = model_target:forward(train.s_t):clone()
                local V_s = q_target:max(2):squeeze()
                local V_s_1 = q_next:max(2):squeeze()
                for b = 1, opt.bs do
                    -- Advantage Learning (AL) operator
                    local Q_s_a = q_target[b][train.a_t[b]]
                    local AL = -opt.action_gap_alpha * (V_s[b] - Q_s_a)

                    -- Persistent Advantage Learning (PAL) operator
                    local Q_s_1_a = q_next[b][train.a_t[b]]
                    local PAL = -opt.action_gap_alpha * (V_s_1[b] - Q_s_1_a)

                    td_err[{ { b }, { train.a_t[b] } }]:add(math.max(AL, PAL))
                end
            end

            -- Backward pass
            local feval = function(x)

                -- Reset parameters
                gradParams:zero()

                -- Backprop
                train_q_episode[step] = td_err:clone():pow(2):mean() * 0.5
                model:backward(train.s_t, -td_err)

                -- Clip Gradients
                gradParams:clamp(-5, 5)

                return 0, gradParams
            end

            optim_func(feval, params, optim_config, optim_state)

            -- Update target network
            -- params_target:mul(1 - opt.target_gamma):add(opt.target_gamma, params)
            if step_count % opt.target_step == 0 then
                params_target:copy(params)
            end
        end

        -- next state
        episode.s_t = episode.s_t1:clone()
        step = step + 1

        -- Total steps
        step_count = step_count + 1
    end

    -- Compute statistics
    train_q = train_q_episode:narrow(1, 1, step - 1):mean()
    train_r = train_r_episode:narrow(1, 1, step - 1):sum()

    test_r = exp.test(opt, env, model)

    -- Compute moving averages
    if e == 1 then
        train_q_avg = train_q
        train_r_avg = train_r
        test_r_avg = test_r
    else
        train_q_avg = 0.99 * train_q_avg + 0.01 * train_q
        train_r_avg = 0.99 * train_r_avg + 0.01 * train_r
        test_r_avg = 0.99 * test_r_avg + 0.01 * test_r
    end
    test_r_all[e] = test_r

    -- Print statistics
    if e == 1 or e % opt.step == 0 then
        log.infof('e=%d, train_q=%.3f, train_q_avg=%.3f, train_r=%.3f, train_r_avg=%.3f, test_r=%.3f, test_r_avg=%.3f, t/e=%.2f sec, t=%d min.',
            e, train_q, train_q_avg, train_r, train_r_avg, test_r, test_r_all:narrow(1, 1, e):mean(),
            sys.clock() - time, torch.toc(beginning_time) / 60)
    end
end

-- Plot Q values
if opt.plot == 1 then
    require 'image'
    local s_t = torch.Tensor(2):zero()
    local q_board = torch.Tensor(100, 100)
    for i = 1, 100 do
        for j = 1, 100 do
            s_t[1] = i / 100
            s_t[2] = j / 100
            q_board[i][j] = model:forward(s_t):max()
        end
    end
    print(q_board)
    local q_board_min = q_board:min()
    local q_board_max = q_board:max()
    q_board:add(-q_board_min):div(q_board_max - q_board_min)
    if opt.action_gap == 1 then
        image.save('q_board_gap.png', q_board)
    else
        image.save('q_board.png', q_board)
    end
end