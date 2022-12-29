#! /usr/bin/env python3

#TODO:
# HIGH PRIORITY
# -add mode for displaying stats for particular bot
# -better argparse help
# -add a nice way of renaming bots (requires updating games history :/)
# -create github repo
# -create psyleague pypi package 
# -choose_match: update matchmaking (more priority to top bots 
# -add more ranking models (openskill)
# -find a good ranking model for fixed-skill bots
# -add comments to config file

# LOW PRIORITY
# -worker error shouldn't immediately interrupt main thread (small chance for corrupting results)
# -add support for n-player games 
# -wrapper for \r printing
# -add a nice way of permanently deleting a bot (requires recalculating of the whole ranking :/ )

# ???
# -add ability to rerun games under a different model?
# -show: add --persistent mode to constantly refresh results?
# -switch to JSON for db/games/msg?
# -add an option to update default config? (psyleague config -> psyleague config new)



__version__ = '0.1.0'

import signal
import copy
import time
import shutil
import random
import datetime
import re
import argparse
import os.path
import subprocess
import queue
import traceback
from threading import Thread
from typing import List, Dict, Tuple, Any

import tabulate
import portalocker
import trueskill as ts
import openskill 
import toml

CONFIG_FILE = 'psyleague.cfg'

args = None
cfg = None

games_queue = queue.Queue()
results_queue = queue.Queue()



class RollingStat:
    def __init__(self, window_length: int):
        self.window_length = window_length
        self.data = []
        
    def add(self) -> None:
        self.data.append(time.time())
        
    def get_count(self) -> int:
        t = time.time()
        while len(self.data) and self.data[0] + self.window_length < t:
            del self.data[0]
        return len(self.data)
    

# [Section] Container classes

class Bot:
    def __init__(self, name, description, mu=25.0, sigma=8.333, games=0, active=1):
        self.name = name
        self.description = description
        self.mu = mu
        self.sigma = sigma
        self.games = games
        self.active = active
        
    @classmethod
    def from_str(cls, input):
        a = [s.strip() for s in input.split(':')]
        return cls(a[0], a[1], float(a[2]), float(a[3]), int(a[4]), int(a[5]))
        
    def __repr__(self):
        return f'{self.name} : {self.description} : {self.mu} : {self.sigma} : {self.games} : {self.active}'
        
    def to_ts(self) -> ts.Rating:
        return ts.Rating(mu=self.mu, sigma=self.sigma)
        
    def update(self, data) -> None:
        if isinstance(data, ts.Rating):
            self.mu = data.mu
            self.sigma = data.sigma
        else:
            assert False, 'Unknown rating type'

    
class Game:
    def __init__(self, p1=None, p2=None, rank1=None, rank2=None, *, str=None):
        self.p1 = p1
        self.p2 = p2
        self.rank1 = rank1
        self.rank2 = rank2
        if str:
            self.p1, self.p2, self.rank1, self.rank2 = str.split()
            self.rank1 = int(self.rank1)
            self.rank2 = int(self.rank2)
            
    def __repr__(self):
        return f'{self.p1} {self.p2} {self.rank1} {self.rank2}'
    
    
# [Section] "DB" functions

def load_config() -> Dict[str, Any]:
    with open(CONFIG_FILE, 'r') as f:
        return toml.load(f);
    
lock_args = {'timeout': 2.0, 'check_interval': 0.02}
    
def log(*args) -> None:
    with portalocker.Lock(cfg['file_log'], 'a', **lock_args) as f:
        dt = datetime.datetime.utcnow().replace(microsecond=0)
        f.write('[' + str(dt) + '] ' + ' '.join(map(str, args)) + '\n')

def send_msg(msg: str) -> None:
    with portalocker.Lock(cfg['file_msg'], 'a', **lock_args) as f:
        f.write(msg + '\n')
    
def receive_msgs() -> List[str]:
    if not os.path.exists(cfg['file_msg']):
        return []
    with portalocker.Lock(cfg['file_msg'], 'r+', **lock_args) as f:
        msgs = f.readlines()
        f.truncate(0)
        return [s.strip() for s in msgs]
        
def add_game(game: Game) -> None:
    with portalocker.Lock(cfg['file_games'], 'a', **lock_args) as f:
        f.write(str(game) + '\n')
        
def load_all_games() -> List[Game]:
    if not os.path.exists(cfg['file_games']):
        return []
    with portalocker.Lock(cfg['file_games'], 'r', **lock_args) as f:
        data = f.readlines()
    return [Game(str=s.strip()) for s in data]

# TODO: write to file first and then output with a single call?
def save_db(bots: Dict[str, Bot]) -> None:
    with portalocker.Lock(cfg['file_db'], 'w', **lock_args) as f:
        for b in bots.values():
            f.write(f'{str(b)}\n')

def load_db() -> Dict[str, Bot]:
    if not os.path.exists(cfg['file_db']):
        return {}
    with portalocker.Lock(cfg['file_db'], 'r', **lock_args) as f:
        data = f.readlines()
    bots = [Bot.from_str(line.strip()) for line in data]
    return {b.name: b for b in bots}    


# [Section] helper functions
    
def update_ranking(bots: Dict[str, Bot], game: Game) -> None:
    if cfg['model'] == 'trueskill':
        ts.setup(tau=cfg['tau'], draw_probability=cfg['draw_prob'])
        ratings = ts.rate([[bots[game.p1].to_ts()], [bots[game.p2].to_ts()]], ranks=[game.rank1, game.rank2])
        bots[game.p1].update(ratings[0][0])
        bots[game.p2].update(ratings[1][0])
    else:
        assert False, f'Invalid rating model: {cfg["model"]}'
        
    bots[game.p1].games += 1
    bots[game.p2].games += 1
    

def play_game(bots: List[str], verbose: bool=False) -> Game:
    # TODO: add error handling?
    cmd = cfg['cmd_play_game']
    cmd = cmd.replace('%DIR%', cfg['dir_bots'])
    cmd = cmd.replace('%P1%', bots[0])
    cmd = cmd.replace('%P2%', bots[1])
    
    if verbose:
        print(f'Playing Game: {cmd}')
    
    task = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            
    if task.returncode:
        print(f'Fatal Error: Play Game command {cmd} returned with return code {task.returncode}')
        os._exit(1)
        
    output = task.stdout.decode('UTF-8').strip()
    if verbose:
        print(f'{cmd} produced output: {output}')
    
    ranks = [int(v) for v in output.split()]
    return Game(bots[0], bots[1], ranks[0], ranks[1])


# TODO: restore proper matchmaking 
def choose_match(bots: Dict[str, Bot]) -> List[str]:
    l_bots = [b for b in bots.values() if b.active]

    if len(l_bots) < 2: 
        return None
        
    # find p1
    min_bots = [b for b in l_bots if b.games < cfg['mm_min_matches']]
    selected_bot = None
    if len(min_bots) and random.random() < cfg['mm_min_matches_preference']:
        p1 = random.choice(min_bots)
    else:
        p1 = random.choice(l_bots)
        
    # find p2
    while True:
        p2 = random.choice(l_bots)
        if p1.name != p2.name:
            break
   
    return [p1.name, p2.name] if random.random() < 0.5 else [p2.name, p1.name]


# [Section] Mode functions

def mode_config() -> None:
    source_path = os.path.join(os.path.dirname(__file__), 'psyleague.cfg')
    target_path = os.path.join(os.getcwd(), 'psyleague.cfg')
    if os.file.exists(target_path):
        print('Config file already exist')
        return
    print(f'Creating new config file at psyleague.cfg')
    shutil.copy(source_path, target_path)


def mode_run() -> None:
    log('[Action] Run')
    
    if not args.silent:
        print('Starting Psyleague server, press Ctrl+C to kill it')
    
    bots = load_db()
    games = load_all_games()
    
    games_played = 0
    games_stat = RollingStat(60.0)
    last_msg_time = time.time()
    
    try:
        start_time = time.time()
        
        def worker_loop() -> None:
            while True:
                try:
                    players = games_queue.get(block=False)
                    if players is None: 
                        break
                    game = play_game(players, args.verbose)
                    results_queue.put(game)
                except queue.Empty:
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    print('Worker Interrupted, this should never happen!')
                    break
                except:
                    print('Fatal error in one of the workers')
                    traceback.print_exc()
                    os._exit(1)
                    
                    
        workers = [Thread(target=worker_loop) for _ in range(cfg['n_workers'])]
        for worker in workers:
            worker.start()
            
        while True:
            time.sleep(0.1)
            
            if not args.silent and not args.verbose:
                active_bots = sum([1 for b in bots.values() if b.active])
                print(f'\rActive Bots: {active_bots}  Games since launch: {games_played}  Games in the last 60s: {games_stat.get_count()}                    \r', end='')
            
            # retrieve and process all messages
            for msg in receive_msgs():
                if not args.silent:
                    print(f'{msg}' + ' '*(70-len(msg))) # XXX: Ugly, wrapper for this would be better
                a = [s.strip() for s in msg.split(':')]
                msg_type = a[0]
                if msg_type == 'ADD_BOT':
                    name = a[1]
                    description = a[2]
                    # TODO: add some error checking
                    bots[name] = Bot(name, description)
                    save_db(bots)
                elif msg_type == 'STOP_BOT':
                    name = a[1]
                    # TODO: add some error checking
                    bots[name].active = 0
                    save_db(bots)
                elif msg_type == 'UPDATE_BOT':
                    name = a[2]
                    new_name = a[3] # XXX: OPERATION NOT SUPPORTED
                    description = a[4]
                    bots[name].description = description
                    save_db(bots)
                else:
                    assert False, f'Unknown message type: {msg_type}'
                
            # add new games to the queue
            while games_queue.qsize() < cfg['n_workers'] * 2:
                players = choose_match(bots)
                if players is None: 
                    break
                games_queue.put(players)
            
            # process results
            try:
                while True:
                    game = results_queue.get(block=False)
                    if args.verbose:
                        print(f'Processing result: {game.p1} vs {game.p2}, outcome: {game.rank1} {game.rank2}')
                    add_game(game)
                    update_ranking(bots, game)
                    save_db(bots) 
                    games_played += 1
                    games_stat.add()
            except queue.Empty:
                pass
            
    except KeyboardInterrupt:
        print('\nInterrupted by user, waiting for all workers to finish!')
        print('If this doesn\'t happen, press Ctrl+C again to kill all of the workers')
        try:
            while True:
                games_queue.get(block=False)
        except:
            pass
            
        try:
            for _ in workers:
                games_queue.put(None)
            for worker in workers:
                worker.join()
        except:
            os._exit(1)
        
    except:
        print('\nFatal Error in the main thread')
        traceback.print_exc()
        os._exit(1)


def mode_bot() -> None:
    if args.cmd == 'add':
        bots = load_db()
        if args.name in bots:
            print(f'Bot {args.name} already exists, ignoring command')
            return
            
        if not args.src:
            args.src = args.name
        
        cmd = cfg['cmd_bot_setup']
        cmd = cmd.replace('%DIR%', cfg['dir_bots'])
        cmd = cmd.replace('%NAME%', args.name)
        cmd = cmd.replace('%SRC%', args.src)
        
        os.makedirs(cfg['dir_bots'], exist_ok=True)
        print(f'Running Setup: {cmd}')
        proc = subprocess.run(cmd, shell=True)
        
        # XXX: is this the best way to handle failure?
        if proc.returncode:
            print(f'Setup failed with exit code {proc.returncode}')
            return
        
        log(f'[Action] Add Bot {args.name}')
        send_msg(f'ADD_BOT : {args.name} : {args.description}')
    elif args.cmd == 'stop':
        bots = load_db()
        if args.name not in bots:
            print(f'Bot {args.name} doesn\'t exist, ignoring command')
            return
            
        log(f'[Action] Remove Bot {args.name}')
        send_msg(f'STOP_BOT : {args.name}')
    elif args.cmd == 'update':
        bots = load_db()
        if args.name not in bots:
            print(f'Bot {args.name} doesn\'t exist, ignoring command')
            return
            
        log(f'[Action] Update Bot {args.name}')
        send_msg(f'UPDATE_BOT : {args.name} : {args.name} : {args.description}')
    else:
        assert False, f'Uknown bot command: {args.cmd}'


def mode_show() -> None:
    log('[Action] Show')
    bots = load_db()

    ranking = sorted(bots.values(), key=lambda b: b.mu-3*b.sigma, reverse=True)

    if hasattr(tabulate, 'MIN_PADDING'):
        tabulate.MIN_PADDING = 0

    headers = ['Pos', 'Name', 'Score', 'Games', 'Mu', 'Sigma', 'Description']
    table = []
    if args.limit is None:
        args.limit = len(ranking)
    for i, b in enumerate(ranking[0:args.limit]):
        table.append([i+1, b.name, b.mu-3*b.sigma, b.games, b.mu, b.sigma, b.description])
    print(tabulate.tabulate(table, headers=headers, floatfmt=f'.3f'))


def _main() -> None:
    parser = argparse.ArgumentParser(description='Local league system for bot contests\nMore help available at https://github.com/FakePsyho/psyleague \nYou can type psyleague mode --help for more information about specific mode', formatter_class=argparse.RawTextHelpFormatter)
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(title='modes')
    
    parser_config = subparsers.add_parser('config', aliases=['c'], help='creates a new config file in the current directory')
    parser_config.set_defaults(func=mode_config)
    
    parser_run = subparsers.add_parser('run', aliases=['r'], help='start psyleague and spin-up all of the workers')
    parser_run.set_defaults(func=mode_run)
    parser_run.add_argument('-s', '--silent', action='store_true', help='turns off all of the messages')
    parser_run.add_argument('-v', '--verbose', action='store_true', help='shows extra information, good for debugging')

    parser_bot = subparsers.add_parser('bot', aliases=['b'], help='commands related to adding/stopping/updating bots')
    parser_bot.set_defaults(func=mode_bot)
    parser_bot.add_argument('cmd', choices=['add', 'stop', 'update'], help='add a new bot/stop currently active bot/update bot')
    parser_bot.add_argument('name', help='name of the bot')
    parser_bot.add_argument('-s', '--src', type=str, default=None, help='source file, if not provided defaults to NAME (used in add)')
    parser_bot.add_argument('-d', '--description', type=str, default='n/a', help='description of the bot (used in add/update)')

    parser_show = subparsers.add_parser('show', aliases=['s'], help='shows the current ranking for all bots')
    parser_show.set_defaults(func=mode_show)
    parser_show.add_argument('-l', '--limit', type=int, default=None, help='limits ranking to top X bots')
    parser_show.add_argument('-a', '--active', action='store_true', help='shows only active bots')

    # load config
    global cfg
    cfg = load_config()
    
    assert cfg['version'] == __version__, 'Version of the config file doesn\'t match psyleague version'
    
    global args
    args = parser.parse_args()

    if not args.func:
        parser.print_help()
        return
    
    args.func()


if __name__ == '__main__':
    _main()
