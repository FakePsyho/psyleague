#!/usr/bin/env python3

# Author: Psyho
# Twitter: https://twitter.com/fakepsyho

#TODO:
# HIGH PRIORITY
# -add mode for displaying stats for particular bot
# -better argparse help
# -add a nice way of renaming bots (requires updating games history :/)
# -choose_match: update matchmaking (more priority to top bots)
# -add more ranking models (openskill)
# -find a good ranking model for fixed-skill bots
# -add comments to config file
# -add a way to add information about the game (info about generated map), similar system to [DATA] in psytester
# -add a way to filer results by test type
# -add a bot having only an executable? (allows for bots without source code / in a different language)
# -change psyleague.db format to csv?

# LOW PRIORITY
# -worker error shouldn't immediately interrupt main thread (small chance for corrupting results)
# -add support for n-player games 
# -wrapper for \r printing
# -add a nice way of permanently deleting a bot (requires recalculation of the whole ranking :/ )

# ???
# -add ability to rerun games under a different model?
# -show: add --persistent mode to constantly refresh results?
# -switch to JSON for db/games/msg?
# -add an option to update default config? (psyleague config -> psyleague config new)
# -add option to use a different config? 



__version__ = '0.3.0'

import signal
import time
import shutil
import random
import json
import argparse
import sys
import os.path
import subprocess
import queue
import traceback
from datetime import datetime
from threading import Thread
from typing import List, Dict, Tuple, Any, Union

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
    

#region Container classes

class Bot:
    def __init__(self, name, description, mu=25.0, sigma=25/3, games=0, errors=0, active=1, cdate=None):
        self.name = name
        self.description = description
        self.mu = mu
        self.sigma = sigma
        self.games = games
        self.errors = errors
        self.active = active
        self.cdate = cdate or datetime.now()
        
    @classmethod
    def from_str(cls, input):
        a = [s.strip() for s in input.split(':')]
        return cls(a[0], a[1], float(a[2]), float(a[3]), int(a[4]), int(a[5]), int(a[6]), datetime.strptime(a[7], '%Y-%m-%d %H-%M-%S'))
        
    def __repr__(self):
        return f'{self.name} : {self.description} : {self.mu} : {self.sigma} : {self.games} : {self.errors} : {self.active} : {self.cdate.strftime("%Y-%m-%d %H-%M-%S")}'
        
    def to_ts(self) -> ts.Rating:
        return ts.Rating(mu=self.mu, sigma=self.sigma)
        
    def update(self, data) -> None:
        if isinstance(data, ts.Rating):
            self.mu = data.mu
            self.sigma = data.sigma
        else:
            assert False, 'Unknown rating type'

    
class Game:
    def __init__(self, p1=None, p2=None, rank1=None, rank2=None, error1=0, error2=0, *, str=None):
        self.players = [p1, p2]
        self.ranks = [rank1, rank2]
        self.errors = [error1, error2]
        self.test_data = {}
        self.player_data = [{}, {}]
        if str:
            if str[0] == '{':
                data = json.loads(str)
                self.players = data['players']
                self.ranks = data['ranks']
                self.errors = data['errors']
                self.test_data = data['test_data']
                self.player_data = data['player_data']
            else:
                v = str.split()
                self.players = [v[0], v[1]]
                self.ranks = [int(v[2]), int(v[3])]
                if len(v) >= 5:
                    self.errors = [int(v[4]), int(v[5])]
            
    def __repr__(self):
        return json.dumps(self.__dict__)
        
#endregion    

#region "DB" functions

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return toml.load(f);
    
lock_args = {'timeout': 2.0, 'check_interval': 0.02}
    
def log(*args) -> None:
    with portalocker.Lock(cfg['file_log'], 'a', **lock_args) as f:
        dt = datetime.now().replace(microsecond=0)
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

def save_all_game(games: List[Game]) -> None:
    with portalocker.Lock(cfg['file_games'], 'w', **lock_args) as f:
        for g in games:
            f.write(str(g) + '\n')

# TODO: write to string first and then output with a single call?
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

#endregion

#region helper functions
    
def update_ranking(bots: Dict[str, Bot], games: Union[List[Game], Game]) -> None:
    if cfg['model'] == 'trueskill':
        ts.setup(tau=cfg['tau'], draw_probability=cfg['draw_prob'])
    else:
        assert False, f'Invalid rating model: {cfg["model"]}'

    if not isinstance(games, list):
        games = [games]

    for game in games:
        if cfg['model'] == 'trueskill':
            ratings = ts.rate([[bots[game.players[0]].to_ts()], [bots[game.players[1]].to_ts()]], ranks=[game.ranks[0], game.ranks[1]])
            bots[game.players[0]].update(ratings[0][0])
            bots[game.players[1]].update(ratings[1][0])
            
        bots[game.players[0]].games += 1
        bots[game.players[1]].games += 1
        bots[game.players[0]].errors += game.errors[0]
        bots[game.players[1]].errors += game.errors[1]


def recalculate_ranking(bots: Dict[str, Bot], games: List[Game]) -> Dict[str, Bot]:
    new_bots = {b.name: Bot(b.name, b.description) for b in bots.values()}
    update_ranking(new_bots, games)
    return new_bots


def play_games(bots: List[str], verbose: bool=False) -> Union[Game, List[Game]]:
    # TODO: add error handling?
    cmd = cfg['cmd_play_game']
    cmd = cmd.replace('%DIR%', cfg['dir_bots'])
    cmd = cmd.replace('%P1%', bots[0])
    cmd = cmd.replace('%P2%', bots[1])
    
    if verbose:
        print(f'Playing Game: {cmd}')
    
    if os.name == 'nt':
        task = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        task = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
            
    if task.returncode:
        print(f'Fatal Error: Play Game command {cmd} returned with return code {task.returncode}')
        os._exit(1)
        
    output = task.stdout.decode('UTF-8').strip()
    if verbose:
        print(f'{cmd} produced output: {output}')
    
    if output[0] == '{':
        data = json.loads(output)
        data['players'] = bots
        return Game(str=json.dumps(data))
    elif output[0] == '[':
        data = json.loads(output)
        for d in data:
            d['players'] = bots
        return [Game(str=json.dumps(d)) for d in data]
    else:
        data = [int(v) for v in output.split()]
        return Game(bots[0], bots[1], *data)


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

#endregion

#region Mode functions

def mode_config() -> None:
    source_path = os.path.join(os.path.dirname(__file__), 'psyleague.cfg')
    target_path = os.path.join(os.getcwd(), 'psyleague.cfg')
    if os.path.exists(target_path):
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
    games_total = args.games or sys.maxsize
    games_left = games_total
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
                    games = play_games(players, args.verbose)
                    if isinstance(games, Game):
                        games = [games]
                    for game in games:
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
                    
                    
        workers = [Thread(target=worker_loop) for _ in range(args.workers if args.workers is not None else cfg['n_workers'])]
        for worker in workers:
            worker.start()
            
        while True:
            time.sleep(0.1)
            
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
                elif msg_type == 'UPDATE_BOT':
                    name = a[1]
                    new_name = a[2]
                    description = a[3]
                    bots[name].description = description
                    if new_name != name:
                        bots[new_name] = bots[name]
                        bots[new_name].name = new_name
                        del bots[name]
                        games = load_all_games()
                        for g in games:
                            if g.players[0] == name:
                                g.players[0] = new_name
                            if g.players[1] == name:
                                g.players[1] = new_name
                        save_all_game(games)
                    save_db(bots)
                elif msg_type == 'STOP_BOT':
                    name = a[1]
                    # TODO: add some error checking
                    bots[name].active = 0
                    save_db(bots)
                elif msg_type == 'REMOVE_BOT':
                    games = load_all_games()
                    games_no = len(games)
                    name = a[1]
                    games = [g for g in games if g.players[0] != name and g.players[1] != name]
                    print(f'Removed {games_no - len(games)} games')
                    del bots[name]
                    print('Recalculating ranking')
                    bots = recalculate_ranking(bots, games)
                    save_all_game(games)
                    save_db(bots)
                else:
                    assert False, f'Unknown message type: {msg_type}'
                
            # add new games to the queue
            while games_queue.qsize() < cfg['n_workers'] * 2 and games_left > 0:
                players = choose_match(bots)
                if players is None: 
                    break
                games_left -= 1
                games_queue.put(players)
            # TODO: add games_left == 0 handling
                
            
            # process results
            try:
                while True:
                    game = results_queue.get(block=False)
                    if args.verbose:
                        print(f'Processing result: {game.players[0]} vs {game.players[1]}, outcome: {game.ranks[0]} {game.ranks[1]}')
                    if game.players[0] not in bots or game.players[1] not in bots:
                        print(f'Warning: Unknown bot in game: {game.players[0]} vs {game.players[1]}; skipping game')
                        continue
                    add_game(game)
                    update_ranking(bots, game)
                    save_db(bots) 
                    games_played += 1
                    games_stat.add()
            except queue.Empty:
                pass
                
            if not args.silent and not args.verbose:
                active_bots = sum([1 for b in bots.values() if b.active])
                print(f'\rActive Bots: {active_bots}  Games since launch: {games_played}{f" / {games_total}" if args.games else ""}  Games in the last 60s: {games_stat.get_count()}                    \r', end='')
            
            if games_played >= games_total:
                break
            
    except KeyboardInterrupt:
        print('\nInterrupted by user, waiting for all workers to finish!')
        print('If this doesn\'t happen, press Ctrl+C again to kill all of the workers')
        
    except:
        print('\nFatal Error in the main thread')
        traceback.print_exc()
        os._exit(1)

    # stop running any new games
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
    elif args.cmd == 'update':
        bots = load_db()
        if args.name not in bots:
            print(f'Bot {args.name} doesn\'t exist, ignoring command')
            return
            
        log(f'[Action] Update Bot {args.name}')
        send_msg(f'UPDATE_BOT : {args.name} : {args.new_name or args.name} : {args.description}')
    elif args.cmd == 'stop':
        bots = load_db()
        for bot in args.name.split(','):
            if bot not in bots:
                print(f'Bot {bot} doesn\'t exist, ignoring command')
                return
        for bot in args.name.split(','):
            log(f'[Action] Stop Bot {bot}')
            send_msg(f'STOP_BOT : {bot}')
    elif args.cmd == 'remove':
        bots = load_db()
        for bot in args.name.split(','):
            if bot not in bots:
                print(f'Bot {bot} doesn\'t exist, ignoring command')
                return
        for bot in args.name.split(','):
            log(f'[Action] Remove Bot {bot}')
            send_msg(f'REMOVE_BOT : {bot}')
    else:
        assert False, f'Uknown bot command: {args.cmd}'


def mode_show() -> None:
    log('[Action] Show')
    bots = load_db()

    if args.resample:
        games = load_all_games()
        if args.resample:
            random.seed(datetime.now())
            games = random.choices(games, k=args.resample)
        print(f'Recalculating ranking using {args.resample} games')
        bots = recalculate_ranking(bots, games)
        print()
    
    ranking = sorted(bots.values(), key=lambda b: b.mu-3*b.sigma, reverse=True)

    if args.active:
        ranking = [b for b in ranking if b.active]
        
    if args.best:
        ranking = ranking[:args.best]
        
    if args.recent:
        ranking = sorted(ranking, key=lambda b: b.cdate, reverse=True)
        ranking = ranking[:args.recent]
    
    columns = {}
    columns['pos'] = ('Pos', list(range(1, 1+len(ranking))))
    columns['name'] = ('Name', [b.name for b in ranking])
    columns['score'] = ('Score', [b.mu-3*b.sigma for b in ranking])
    columns['games'] = ('Games', [b.games for b in ranking])
    columns['percentage'] = ('%', [f'{min(100, b.games * 100 // cfg["mm_min_matches"])}%' for b in ranking])
    columns['mu'] = ('Mu', [b.mu for b in ranking])
    columns['sigma'] = ('Sigma', [b.sigma for b in ranking])
    columns['errors'] = ('Errors', [b.errors for b in ranking])
    columns['active'] = ('Active', [b.active for b in ranking])
    columns['description'] = ('Description', [b.description for b in ranking])
    try:
        columns['date'] = ('Created', [b.cdate.strftime(cfg['date_format']) for b in ranking])
    except ValueError:
        print(f'Your date_format: "{cfg["date_format"]}" is invalid')
        sys.exit(1)
        
    headers = []
    table = []
    for column_name in cfg['leaderboard'].split(','):
        column_name = column_name.lower()
        optional = False
        if column_name[-1] == '?':
            optional = True
            column_name = column_name[:-1]
        if column_name not in columns:
            print(f'Unknown column name: {column_name}, please correct the leaderboard option')
            sys.exit(1)
        h, c = columns[column_name]
        if optional and c.count(c[0]) == len(c):
            continue
        headers.append(h)
        table.append(c)
        
    table = list(zip(*table)) #transpose
        
    if hasattr(tabulate, 'MIN_PADDING'):
        tabulate.MIN_PADDING = 0

    print(tabulate.tabulate(table, headers=headers, floatfmt=f'.3f'))


def mode_test() -> None:
    bots = load_db()
    games = load_all_games()
    recalculate_ranking(bots, games)
    for bot in bots:
        print(f'{bot}: {bots[bot].games} {bots[bot].mu} {bots[bot].sigma}')

#endregion
    


def _main() -> None:
    parser = argparse.ArgumentParser(description='Local league system for bot contests\nMore help available at https://github.com/FakePsyho/psyleague \nYou can type psyleague mode --help for more information about specific mode', formatter_class=argparse.RawTextHelpFormatter)
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(title='modes')
    
    parser_config = subparsers.add_parser('config', aliases=['c'], help='creates a new config file in the current directory')
    parser_config.set_defaults(func=mode_config)
    
    parser_run = subparsers.add_parser('run', aliases=['r'], help='start psyleague and spin-up all of the workers')
    parser_run.set_defaults(func=mode_run)
    parser_run.add_argument('-g', '--games', type=int, default=None, help='if specified, number of games to run after which psyleague should finish running')
    parser_run.add_argument('-w', '--workers', type=int, default=None, help='number of workers to use (overrides config file)')
    parser_run.add_argument('-s', '--silent', action='store_true', help='turns off all of the messages')
    parser_run.add_argument('-v', '--verbose', action='store_true', help='shows extra information, good for debugging')

    parser_bot = subparsers.add_parser('bot', aliases=['b'], help='commands related to adding/stopping/updating bots')
    parser_bot.set_defaults(func=mode_bot)
    parser_bot.add_argument('cmd', choices=['add', 'update', 'stop', 'remove'], help='add a new bot/update bot/stop currently active bot (no more played games)/remove a bot along with games (recalculates ranking)')
    parser_bot.add_argument('name', help='name of the bot')
    parser_bot.add_argument('-s', '--src', type=str, default=None, help='source file, if not provided defaults to NAME (used in add)')
    parser_bot.add_argument('-d', '--description', type=str, default='n/a', help='description of the bot (used in add/update)')
    parser_bot.add_argument('-n', '--new-name', type=str, default=None, help='new name of the bot (used in update)')

    parser_show = subparsers.add_parser('show', aliases=['s'], help='shows the current ranking for all bots')
    parser_show.set_defaults(func=mode_show)
    parser_show.add_argument('-a', '--active', action='store_true', help='shows only active bots')
    parser_show.add_argument('-m', '--model', choices=['trueskill'], default=None, help='recalculates ranking using a different model')
    parser_show.add_argument('-s', '--resample', type=int, default=None, help='recalculates ranking using bootstrapping')
    parser_show_xgroup = parser_show.add_mutually_exclusive_group()
    parser_show_xgroup.add_argument('-b', '--best', type=int, default=None, help='limits ranking to the best X bots')
    parser_show_xgroup.add_argument('-r', '--recent', type=int, default=None, help='limits ranking to the most recent X bots')

    # parser_test = subparsers.add_parser('test', aliases=['t'], help='test')
    # parser_test.set_defaults(func=mode_test)

    global args
    args = parser.parse_args()
    
    if not args.func:
        parser.print_help()
        return 
    
    # load config
    if args.func != mode_config:
        if not os.path.exists(CONFIG_FILE):
            print('Missing config file, please run "psyleague config" to create a config file in you current directory')
            return
    
        global cfg
        cfg = load_config(CONFIG_FILE)
        assert cfg['version'] == __version__, 'Version of the config file doesn\'t match psyleague version'
    
    args.func()


if __name__ == '__main__':
    _main()
